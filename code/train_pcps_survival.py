import argparse
import json
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.pcps import PrototypeCalibratedPatchSampling



def nll_survival_loss(hazards, time_bin, censorship, alpha=0.0, eps=1e-7):
    time_bin = time_bin.long().view(-1, 1)
    censorship = censorship.float().view(-1, 1)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    survival_pad = torch.cat([torch.ones_like(censorship), survival], dim=1)
    s_prev = torch.gather(survival_pad, 1, time_bin).clamp_min(eps)
    h_this = torch.gather(hazards, 1, time_bin).clamp_min(eps)
    s_this = torch.gather(survival_pad, 1, time_bin + 1).clamp_min(eps)
    event_loss = -(1.0 - censorship) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = -censorship * torch.log(s_this)
    return ((1.0 - alpha) * (event_loss + censored_loss) + alpha * event_loss).mean()


def concordance_index(event, time, risk):
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    concordant = 0.0
    comparable = 0
    for i in range(len(time)):
        for j in range(i + 1, len(time)):
            if time[i] < time[j] and event[i]:
                shorter, longer = i, j
            elif time[j] < time[i] and event[j]:
                shorter, longer = j, i
            elif time[i] == time[j] and event[i] != event[j]:
                shorter, longer = (i, j) if event[i] else (j, i)
            else:
                continue
            comparable += 1
            if risk[shorter] > risk[longer]:
                concordant += 1.0
            elif risk[shorter] == risk[longer]:
                concordant += 0.5
    return concordant / comparable if comparable else float("nan")


def survival_risk(survival):
    return -survival.sum(dim=1)


def load_fold_data(split_csv, feature_dir):
    df = pd.read_csv(split_csv)
    required = {"wsi_id", "fold", "event", "survival_months", "time_bin"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    def load_group(frame):
        result = []
        for _, row in frame.iterrows():
            pt_path = os.path.join(feature_dir, f"{row.wsi_id}.pt")
            if not os.path.exists(pt_path):
                print(f"WARNING: {pt_path} not found, skipping")
                continue
            feats = torch.load(pt_path, map_location="cpu", weights_only=True).float()
            result.append({
                "wsi_id": row.wsi_id,
                "features": feats,
                "time_bin": int(row.time_bin),
                "event": int(row.event),
                "censorship": int(1 - int(row.event)),
                "time": float(row.survival_months),
                "case_id": getattr(row, "case_id", row.wsi_id),
            })
        return result

    return (
        load_group(df[df["fold"] > 0]),
        load_group(df[df["fold"] == 0]),
        load_group(df[df["fold"] < 0]),
    )


class PCPSSurvivalTaskHead(nn.Module):
    def __init__(self, in_dim, n_bins, hidden_dim=256, attn_dim=128, dropout=0.25):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn_a = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attn_b = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attn_c = nn.Linear(attn_dim, 1)
        self.head = nn.Linear(hidden_dim, n_bins)

    def _attention_raw(self, features):
        h = self.feature(features.float())
        raw = self.attn_c(self.attn_a(h) * self.attn_b(h)).squeeze(1)
        return h, raw

    def compute_patch_evidence(self, features):
        return self._attention_raw(features)

    def aggregate_with_mask(self, encoded_features, raw, mask):
        gate = mask.reshape(-1).float().clamp_min(1e-6)
        attention = torch.softmax(raw + gate.log(), dim=0)
        bag = torch.sum(attention.unsqueeze(1) * encoded_features, dim=0)
        logits = self.head(bag.unsqueeze(0))
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        return hazards, survival

    def attention_scores(self, features, mask=None):
        _, raw = self._attention_raw(features)
        if mask is not None:
            raw = raw + mask.reshape(-1).float().clamp_min(1e-6).log()
        return torch.softmax(raw, dim=0)


def evaluate(wsis, selector, classifier, args, device):
    metrics = evaluate_detailed(wsis, selector, classifier, args, device)
    return metrics["loss"], metrics["cindex"]


def evaluate_detailed(wsis, selector, classifier, args, device):
    selector.eval()
    classifier.eval()
    total_loss = 0.0
    events, times, risks, hazard_means = [], [], [], []
    with torch.no_grad():
        for item in wsis:
            feats = item["features"].to(device)
            keep_ratio = min(1.0, args.selected_count / max(1, feats.shape[0]))
            encoded, task_logits = classifier.compute_patch_evidence(feats)
            out = selector(feats, task_logits, target_keep_ratio=keep_ratio)
            hazards, survival = classifier.aggregate_with_mask(
                encoded, task_logits, out["variational_keep_mask"]
            )
            time_bin = torch.tensor([item["time_bin"]], device=device)
            censorship = torch.tensor([item["censorship"]], device=device)
            total_loss += nll_survival_loss(
                hazards, time_bin, censorship, alpha=args.survival_alpha
            ).item()
            events.append(item["event"])
            times.append(item["time"])
            risks.append(float(survival_risk(survival).detach().cpu()[0]))
            hazard_means.append(float(hazards.detach().mean().cpu()))
    n = max(1, len(wsis))
    risks_arr = np.asarray(risks, dtype=float)
    events_arr = np.asarray(events, dtype=bool)
    event_risks = risks_arr[events_arr]
    censored_risks = risks_arr[~events_arr]
    return {
        "loss": total_loss / n,
        "cindex": concordance_index(events, times, risks),
        "risk_mean": float(np.mean(risks_arr)) if risks else float("nan"),
        "risk_std": float(np.std(risks_arr)) if risks else float("nan"),
        "event_risk_mean": float(np.mean(event_risks)) if event_risks.size else float("nan"),
        "censored_risk_mean": float(np.mean(censored_risks)) if censored_risks.size else float("nan"),
        "hazard_mean": float(np.mean(hazard_means)) if hazard_means else float("nan"),
        "n": int(n),
        "events": int(np.sum(events_arr)) if events else 0,
    }


def _clone_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def train_one_fold(train_wsis, val_wsis, test_wsis, args, fold, device):
    if not train_wsis:
        raise RuntimeError(f"Fold {fold}: no training WSI features were loaded")
    feature_dim = train_wsis[0]["features"].shape[1]
    selector = PrototypeCalibratedPatchSampling(
        feature_dim=feature_dim,
        prototype_count=args.prototype_count,
        assignment_temperature=args.assignment_temperature,
        initial_reputation=args.initial_reputation,
        concrete_temperature=args.concrete_temperature,
        mask_samples=args.mask_samples,
        reputation_contrast=args.reputation_contrast,
        prototype_calibration_max=args.prototype_calibration_max,
        prototype_calibration_init=args.prototype_calibration_init,
    ).to(device)
    selector.initialize_adaptive_codebook(
        [item["features"] for item in train_wsis],
        sample_ratio=args.kmeans_sample_ratio,
    )
    if args.reputation_jitter_std > 0:
        with torch.no_grad():
            selector.prototype_reputation_logits.add_(
                torch.randn_like(selector.prototype_reputation_logits)
                * args.reputation_jitter_std
            )
    classifier = PCPSSurvivalTaskHead(
        in_dim=feature_dim, n_bins=args.n_bins,
        hidden_dim=args.hidden_dim, attn_dim=args.attn_dim,
        dropout=args.dropout,
    ).to(device)

    selector_main_params = [
        p for name, p in selector.named_parameters() if name != "prototype_codebook"
    ]
    optimizer = AdamW([
        {"params": selector_main_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": classifier.parameters(), "lr": args.lr * args.task_lr_factor, "weight_decay": args.weight_decay},
        {"params": [selector.prototype_codebook], "lr": args.lr * args.codebook_lr_factor, "weight_decay": 0.0},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_cindex = -float("inf")
    best_val_loss_state = None
    best_val_cindex_state = None
    last_state = None
    history = []

    for epoch in range(args.epochs):
        selector.train()
        classifier.train()
        totals = {"loss": 0.0, "nll": 0.0, "cons": 0.0, "rho": 0.0, "budget": 0.0, "morph": 0.0, "anchor": 0.0}
        events, times, risks = [], [], []
        for idx in np.random.permutation(len(train_wsis)):
            item = train_wsis[idx]
            feats = item["features"].to(device)
            keep_ratio = min(1.0, args.selected_count / max(1, feats.shape[0]))
            encoded, task_logits = classifier.compute_patch_evidence(feats)
            out = selector(feats, task_logits, target_keep_ratio=keep_ratio)
            hazards, survival = classifier.aggregate_with_mask(
                encoded, task_logits, out["variational_keep_mask"]
            )
            time_bin = torch.tensor([item["time_bin"]], device=device)
            censorship = torch.tensor([item["censorship"]], device=device)
            nll = nll_survival_loss(
                hazards, time_bin, censorship, alpha=args.survival_alpha
            )
            loss = (
                nll
                + args.calibration_weight * out["prototype_calibration_divergence"]
                + args.prototype_reputation_weight * out["prototype_reputation_loss"]
                + args.selection_budget_weight * out["selection_budget_loss"]
                + args.codebook_compactness_weight * out["codebook_compactness_loss"]
                + args.codebook_anchor_weight * out["codebook_anchor_loss"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), args.gradient_clip)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.gradient_clip)
            optimizer.step()

            totals["loss"] += loss.item()
            totals["nll"] += nll.item()
            totals["cons"] += out["prototype_calibration_divergence"].item()
            totals["rho"] += out["prototype_reputation_loss"].item()
            totals["budget"] += out["selection_budget_loss"].item()
            totals["morph"] += out["codebook_compactness_loss"].item()
            totals["anchor"] += out["codebook_anchor_loss"].item()
            events.append(item["event"])
            times.append(item["time"])
            risks.append(float(survival_risk(survival).detach().cpu()[0]))

        scheduler.step()
        n_train = max(1, len(train_wsis))
        train_cindex = concordance_index(events, times, risks)
        train_risks = np.asarray(risks, dtype=float)
        train_events = np.asarray(events, dtype=bool)
        train_event_risks = train_risks[train_events]
        train_censored_risks = train_risks[~train_events]
        val_metrics = evaluate_detailed(val_wsis, selector, classifier, args, device)


        test_metrics = {key: float("nan") for key in (
            "loss", "cindex", "risk_mean", "risk_std", "event_risk_mean",
            "censored_risk_mean", "hazard_mean")}
        val_loss = val_metrics["loss"]
        val_cindex = val_metrics["cindex"]
        reputation_prior = selector.compute_prototype_reputation().detach().squeeze(1)
        center = torch.nn.functional.normalize(selector.prototype_codebook.detach(), dim=1)
        center_init = torch.nn.functional.normalize(selector.initial_prototype_codebook.detach(), dim=1)
        codebook_drift = (1.0 - (center * center_init).sum(dim=1)).mean().item()

        record = {
            "epoch": epoch + 1,
            "train_cindex": train_cindex,
            "train_risk_mean": float(np.mean(train_risks)) if risks else float("nan"),
            "train_risk_std": float(np.std(train_risks)) if risks else float("nan"),
            "train_event_risk_mean": float(np.mean(train_event_risks)) if train_event_risks.size else float("nan"),
            "train_censored_risk_mean": float(np.mean(train_censored_risks)) if train_censored_risks.size else float("nan"),
            "val_loss": val_loss,
            "val_cindex": val_cindex,
            "val_risk_mean": val_metrics["risk_mean"],
            "val_risk_std": val_metrics["risk_std"],
            "val_event_risk_mean": val_metrics["event_risk_mean"],
            "val_censored_risk_mean": val_metrics["censored_risk_mean"],
            "val_hazard_mean": val_metrics["hazard_mean"],
            "test_loss": test_metrics["loss"],
            "test_cindex": test_metrics["cindex"],
            "test_risk_mean": test_metrics["risk_mean"],
            "test_risk_std": test_metrics["risk_std"],
            "test_event_risk_mean": test_metrics["event_risk_mean"],
            "test_censored_risk_mean": test_metrics["censored_risk_mean"],
            "test_hazard_mean": test_metrics["hazard_mean"],
            "reputation_prior_mean": reputation_prior.mean().item(),
            "reputation_prior_std": reputation_prior.std(unbiased=False).item(),
            "reputation_prior_min": reputation_prior.min().item(),
            "reputation_prior_max": reputation_prior.max().item(),
            "codebook_cosine_drift": codebook_drift,
            "prototype_calibration_strength": float(selector.compute_prototype_calibration_strength().detach().cpu()),
        }
        record.update({f"train_{key}": value / n_train for key, value in totals.items()})
        history.append(record)
        print(
            f"Fold {fold} Epoch {epoch + 1}/{args.epochs}: "
            f"Loss={record['train_loss']:.4f} NLL={record['train_nll']:.4f} "
            f"TrainC={train_cindex:.4f} ValLoss={val_loss:.4f} ValC={val_cindex:.4f} "
            f"TestLoss={record['test_loss']:.4f} TestC={record['test_cindex']:.4f} "
            f"Budget={record['train_budget']:.4f} "
            f"Risk(val e/c={record['val_event_risk_mean']:.3f}/{record['val_censored_risk_mean']:.3f}, "
            f"test e/c={record['test_event_risk_mean']:.3f}/{record['test_censored_risk_mean']:.3f}) "
            f"reputation_prior={record['reputation_prior_mean']:.4f}+/-{record['reputation_prior_std']:.4f} "
            f"morph_lambda={record['prototype_calibration_strength']:.4f} H_drift={codebook_drift:.6f}"
        )

        current_state = {
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_cindex": val_cindex,
            "test_loss": record["test_loss"],
            "test_cindex": record["test_cindex"],
            "pcps_state_dict": _clone_state(selector),
            "task_head_state_dict": _clone_state(classifier),
        }
        last_state = current_state
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_state = current_state
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_val_cindex_state = current_state

    if best_val_loss_state is None or best_val_cindex_state is None or last_state is None:
        raise RuntimeError(f"Fold {fold}: no valid survival PCPS checkpoint was produced")
    selector.load_state_dict(best_val_loss_state["pcps_state_dict"])
    classifier.load_state_dict(best_val_loss_state["task_head_state_dict"])
    checkpoints = {
        "best_val_loss": best_val_loss_state,
        "best_val_cindex": best_val_cindex_state,
        "last": last_state,
    }
    return selector, classifier, best_val_loss, best_val_loss_state["val_cindex"], history, checkpoints


def export_pcps_outputs(selector, classifier, all_wsis, args, output_dir, fold, device, history):
    os.makedirs(output_dir, exist_ok=True)
    selector.eval()
    classifier.eval()
    result = {}
    patch_diagnostics = {}

    with torch.no_grad():
        for item in all_wsis:
            wsi_id = item["wsi_id"]
            feats = item["features"].to(device)
            keep_ratio = min(1.0, args.selected_count / max(1, feats.shape[0]))
            encoded, task_logits = classifier.compute_patch_evidence(feats)
            out = selector(feats, task_logits, target_keep_ratio=keep_ratio)
            assign = out["prototype_assignment"]
            prior = out["prototype_prior"]
            post = out["task_relevance_posterior"]
            attention = classifier.attention_scores(feats)
            selection_score = post.squeeze(1)
            k = min(args.selected_count, selection_score.shape[0])
            scores, indices = torch.topk(selection_score, k)
            cluster_task = out["prototype_task_relevance"]
            assignment_confidence, prototype_ids = assign.max(dim=1)

            patch_diagnostics[wsi_id] = {
                "task_relevance_posterior": post.squeeze(1).half().cpu(),
                "prototype_prior": prior.squeeze(1).half().cpu(),
                "pcps_abmil_attention": attention.half().cpu(),
                "pcps_selection_score": selection_score.half().cpu(),
                "prototype_ids": prototype_ids.to(torch.int16).cpu(),
                "prototype_assignment_confidence": assignment_confidence.half().cpu(),
                "task_evidence_logits": task_logits.half().cpu(),
                "task_evidence_probability": out["task_evidence_probability"].squeeze(1).half().cpu(),
            }
            result[wsi_id] = {
                "selected_patch_indices": indices.cpu().tolist(),
                "selection_score_name": "task_relevance_posterior",
                "selection_scores": scores.cpu().tolist(),
                "task_relevance_scores": post.squeeze(1)[indices].cpu().tolist(),
                "abmil_attention_scores": attention[indices].cpu().tolist(),
                "total_patches": int(feats.shape[0]),
                "case_id": item["case_id"],
                "event": int(item["event"]),
                "survival_months": float(item["time"]),
                "time_bin": int(item["time_bin"]),
                "slide_prototype_profile": assign.mean(dim=0).cpu().tolist(),
                "prototype_task_relevance": cluster_task.cpu().tolist(),
                "task_relevance_mean": float(post.mean().item()),
                "task_relevance_std": float(post.std(unbiased=False).item()),
                "reputation_prior_mean": float(prior.mean().item()),
                "reputation_prior_std": float(prior.std(unbiased=False).item()),
                "prototype_calibration_strength": float(out["prototype_calibration_strength"].item()),
            }

    json_path = os.path.join(output_dir, f"pcps_selection_fold{fold}.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    patch_diag_path = os.path.join(output_dir, f"pcps_patch_diagnostics_fold{fold}.pt")
    torch.save(patch_diagnostics, patch_diag_path)
    model_path = os.path.join(output_dir, f"pcps_model_fold{fold}.pt")
    torch.save({
        "pcps_state_dict": selector.state_dict(),
        "task_head_state_dict": classifier.state_dict(),
        "selection_score_name": "task_relevance_posterior",
        "selected_count": args.selected_count,
        "prototype_count": args.prototype_count,
        "history": history,
        "config": vars(args),
    }, model_path)
    diag_path = os.path.join(output_dir, f"pcps_training_diagnostics_fold{fold}.json")
    with open(diag_path, "w", encoding="utf-8") as handle:
        json.dump({"history": history}, handle, indent=2)
    print(f"Saved PCPS selection: {json_path}")
    print(f"Saved PCPS patch diagnostics: {patch_diag_path}")
    print(f"Saved PCPS checkpoint: {model_path}")
    print(f"Saved diagnostics: {diag_path}")


def save_checkpoint_variants(checkpoints, args, output_dir, fold, history):
    os.makedirs(output_dir, exist_ok=True)
    for name, state in checkpoints.items():
        path = os.path.join(output_dir, f"pcps_model_fold{fold}_{name}.pt")
        torch.save({
            "pcps_state_dict": state["pcps_state_dict"],
            "task_head_state_dict": state["task_head_state_dict"],
            "selection_score_name": "task_relevance_posterior",
            "selected_count": args.selected_count,
            "prototype_count": args.prototype_count,
            "checkpoint_name": name,
            "checkpoint_epoch": state["epoch"],
            "checkpoint_val_loss": state["val_loss"],
            "checkpoint_val_cindex": state["val_cindex"],
            "checkpoint_test_loss": state["test_loss"],
            "checkpoint_test_cindex": state["test_cindex"],
            "history": history,
            "config": vars(args),
        }, path)
        print(
            f"Saved {name} checkpoint: {path} "
            f"(epoch={state['epoch']}, val_loss={state['val_loss']:.4f}, "
            f"val_cindex={state['val_cindex']:.4f}, test_cindex={state['test_cindex']:.4f})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--split-pattern", type=str, default="luad_survival_split43_fold{fold}.csv")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--survival-alpha", type=float, default=0.0)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attn-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--calibration-weight", type=float, default=0.2)
    parser.add_argument("--prototype-reputation-weight", type=float, default=0.2)
    parser.add_argument("--reputation-jitter-std", type=float, default=0.0)
    parser.add_argument("--reputation-contrast", type=float, default=0.75)
    parser.add_argument("--prototype-calibration-max", type=float, default=0.5)
    parser.add_argument("--prototype-calibration-init", type=float, default=0.1)
    parser.add_argument("--task-lr-factor", type=float, default=0.5)
    parser.add_argument("--selection-budget-weight", type=float, default=1.0)
    parser.add_argument("--codebook-compactness-weight", type=float, default=0.1)
    parser.add_argument("--codebook-anchor-weight", type=float, default=0.05)
    parser.add_argument("--codebook-lr-factor", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--selected-count", type=int, default=2048,
                        help="May be set to 256 to match M; usually has little effect. Keeping the default is also acceptable.")
    parser.add_argument("--prototype-count", type=int, default=16)
    parser.add_argument("--kmeans-sample-ratio", type=float, default=0.05)
    parser.add_argument("--assignment-temperature", type=float, default=0.1)
    parser.add_argument("--initial-reputation", type=float, default=0.3)
    parser.add_argument("--concrete-temperature", type=float, default=0.1)
    parser.add_argument("--mask-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = [args.fold] if args.fold is not None else range(5)

    for fold in folds:
        split_csv = os.path.join(args.split_dir, args.split_pattern.format(fold=fold))
        if not os.path.exists(split_csv):
            print(f"Split not found: {split_csv}, skipping")
            continue
        train_wsis, val_wsis, test_wsis = load_fold_data(split_csv, args.feature_dir)
        print(f"Fold {fold}: {len(train_wsis)} train, {len(val_wsis)} val, {len(test_wsis)} test WSIs")
        selector, classifier, best_loss, best_cindex, history, checkpoints = train_one_fold(
            train_wsis, val_wsis, test_wsis, args, fold, device
        )
        print(f"Fold {fold} selected checkpoint: val_loss={best_loss:.4f}, val_cindex={best_cindex:.4f}")
        export_pcps_outputs(
            selector, classifier, train_wsis + val_wsis + test_wsis,
            args, args.output_dir, fold, device, history
        )
        save_checkpoint_variants(checkpoints, args, args.output_dir, fold, history)
    print("Done!")


if __name__ == "__main__":
    main()
