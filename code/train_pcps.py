#!/usr/bin/env python3
"""Offline training for Prototype-Calibrated Patch Sampling (PCPS)."""

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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.pcps import PrototypeCalibratedPatchSampling


def load_fold_data(split_csv, feature_dir, fold):
    """Load train/validation/test features without crossing fold boundaries."""
    df = pd.read_csv(split_csv)
    train_ids = df[df["fold"] > 0]["wsi_id"].tolist()
    val_ids = df[df["fold"] == 0]["wsi_id"].tolist()
    test_ids = df[df["fold"] < 0]["wsi_id"].tolist()

    def load_group(ids):
        result = []
        for wsi_id in ids:
            pt_path = os.path.join(feature_dir, f"{wsi_id}.pt")
            if not os.path.exists(pt_path):
                print(f"WARNING: {pt_path} not found, skipping")
                continue
            feats = torch.load(pt_path, map_location="cpu").float()
            label = int(df[df["wsi_id"] == wsi_id]["label"].values[0])
            result.append((wsi_id, feats, label))
        return result

    return load_group(train_ids), load_group(val_ids), load_group(test_ids)


class PCPSTaskHead(nn.Module):
    """ABMIL head supervised through the PCPS variational mask.

    PCPS trains its posterior through this differentiable keep-mask gate. At
    export time, the posterior is the sole patch-selection score; ungated ABMIL
    attention is retained only as a diagnostic and comparison signal.
    """

    def __init__(self, in_dim, num_classes, hidden_dim=256, attn_dim=128, dropout=0.25):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn_a = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attn_b = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attn_c = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _attention_raw(self, features):
        h = self.feature(features.float())
        raw = self.attn_c(self.attn_a(h) * self.attn_b(h)).squeeze(1)
        return h, raw

    def compute_patch_evidence(self, features):
        """Return encoded patches and the shared task-evidence logits."""
        return self._attention_raw(features)

    def aggregate_with_mask(self, encoded_features, raw, mask):
        """Classify a bag using the variational gate over shared evidence."""
        gate = mask.reshape(-1).float().clamp_min(1e-6)
        attn = torch.softmax(raw + gate.log(), dim=0)
        bag = torch.sum(attn.unsqueeze(1) * encoded_features, dim=0)
        return self.classifier(bag.unsqueeze(0))

    def forward(self, features, mask):
        h, raw = self._attention_raw(features)
        return self.aggregate_with_mask(h, raw, mask)

    def attention_scores(self, features, mask=None):
        _, raw = self._attention_raw(features)
        if mask is not None:
            gate = mask.reshape(-1).float().clamp_min(1e-6)
            raw = raw + gate.log()
        return torch.softmax(raw, dim=0)


def safe_auc_score(labels, probabilities, num_classes):
    if not labels:
        return float("nan")
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    try:
        if num_classes == 2:
            if np.unique(labels).size < 2:
                return float("nan")
            return float(roc_auc_score(labels, probabilities[:, 1]))
        present = np.unique(labels)
        if present.size < 2:
            return float("nan")
        return float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")

def train_one_fold(train_wsis, val_wsis, args, fold, device):
    if not train_wsis:
        raise RuntimeError(f"Fold {fold}: no training WSI features were loaded")
    feature_dim = train_wsis[0][1].shape[1]
    args.feature_dim = feature_dim
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
        [feats for _, feats, _ in train_wsis],
        sample_ratio=args.kmeans_sample_ratio,
    )
    if args.reputation_jitter_std > 0:
        with torch.no_grad():
            selector.prototype_reputation_logits.add_(
                torch.randn_like(selector.prototype_reputation_logits)
                * args.reputation_jitter_std
            )
    num_classes = args.num_classes
    if num_classes is None:
        num_classes = max(label for _, _, label in train_wsis) + 1
    classifier = PCPSTaskHead(
        in_dim=feature_dim, num_classes=num_classes
    ).to(device)
    if args.task_head_init:
        abmil_checkpoint = torch.load(
            args.task_head_init, map_location="cpu", weights_only=False
        )
        state_dict = abmil_checkpoint.get(
            "model_state_dict", abmil_checkpoint.get("task_head_state_dict")
        )
        if state_dict is None:
            raise KeyError(
                f"No ABMIL state dict found in {args.task_head_init}"
            )
        classifier.load_state_dict(state_dict, strict=True)
        print(f"Initialized PCPS task branch from {args.task_head_init}")

    # H uses a smaller learning rate and no weight decay. It remains trainable,
    # but morphology/anchor losses keep it close to meaningful K-means regions.
    selector_main_params = [
        p for name, p in selector.named_parameters() if name != "prototype_codebook"
    ]
    optimizer = AdamW(
        [
            {
                "params": selector_main_params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": classifier.parameters(),
                "lr": args.lr * args.task_lr_factor,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [selector.prototype_codebook],
                "lr": args.lr * args.codebook_lr_factor,
                "weight_decay": 0.0,
            },
        ]
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_ce = float('inf')
    selected_val_acc = -1.0
    selected_val_auc = float('nan')
    best_selector_state = None
    best_classifier_state = None
    history = []

    for epoch in range(args.epochs):
        selector.train()
        classifier.train()
        task_branch_trainable = epoch >= args.task_warmup_epochs
        for parameter in classifier.parameters():
            parameter.requires_grad_(task_branch_trainable)
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "cons": 0.0,
            "rho": 0.0,
            "budget": 0.0,
            "morph": 0.0,
            "anchor": 0.0,
        }
        train_correct = 0
        train_labels = []
        train_probs = []

        for idx in np.random.permutation(len(train_wsis)):
            _, feats, label = train_wsis[idx]
            feats = feats.to(device)
            label_t = torch.tensor([label], device=device)
            keep_ratio = min(1.0, args.selected_count / max(1, feats.shape[0]))

            encoded, task_logits = classifier.compute_patch_evidence(feats)
            out = selector(feats, task_logits, target_keep_ratio=keep_ratio)
            logits = classifier.aggregate_with_mask(
                encoded, task_logits, out["variational_keep_mask"]
            )
            ce = ce_loss_fn(logits, label_t)
            loss = (
                ce
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
            totals["ce"] += ce.item()
            totals["cons"] += out["prototype_calibration_divergence"].item()
            totals["rho"] += out["prototype_reputation_loss"].item()
            totals["budget"] += out["selection_budget_loss"].item()
            totals["morph"] += out["codebook_compactness_loss"].item()
            totals["anchor"] += out["codebook_anchor_loss"].item()
            train_correct += int(logits.argmax(dim=1).item() == label)
            train_labels.append(label)
            train_probs.append(torch.softmax(logits.detach(), dim=1).squeeze(0).cpu().numpy())

        scheduler.step()
        n_train = max(1, len(train_wsis))
        selector.eval()
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_labels = []
        val_probs = []
        with torch.no_grad():
            for _, feats, label in val_wsis:
                feats = feats.to(device)
                label_t = torch.tensor([label], device=device)
                keep_ratio = min(1.0, args.selected_count / max(1, feats.shape[0]))
                encoded, task_logits = classifier.compute_patch_evidence(feats)
                out = selector(feats, task_logits, target_keep_ratio=keep_ratio)
                logits = classifier.aggregate_with_mask(
                    encoded, task_logits, out["variational_keep_mask"]
                )
                val_loss += ce_loss_fn(logits, label_t).item()
                val_correct += int(logits.argmax(dim=1).item() == label)
                val_labels.append(label)
                val_probs.append(torch.softmax(logits, dim=1).squeeze(0).cpu().numpy())

        n_val = max(1, len(val_wsis))
        val_acc = val_correct / n_val
        train_auc = safe_auc_score(train_labels, train_probs, num_classes)
        val_auc = safe_auc_score(val_labels, val_probs, num_classes)
        reputation_prior = selector.compute_prototype_reputation().detach().squeeze(1)
        center = torch.nn.functional.normalize(
            selector.prototype_codebook.detach(), dim=1
        )
        center_init = torch.nn.functional.normalize(
            selector.initial_prototype_codebook.detach(), dim=1
        )
        codebook_drift = (1.0 - (center * center_init).sum(dim=1)).mean().item()

        record = {
            "epoch": epoch + 1,
            "train_acc": train_correct / n_train,
            "train_auc": train_auc,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_ce": val_loss / n_val,
            "reputation_prior_mean": reputation_prior.mean().item(),
            "reputation_prior_std": reputation_prior.std(unbiased=False).item(),
            "reputation_prior_min": reputation_prior.min().item(),
            "reputation_prior_max": reputation_prior.max().item(),
            "codebook_cosine_drift": codebook_drift,
            "prototype_calibration_strength": float(
                selector.compute_prototype_calibration_strength().detach().cpu()
            ),
        }
        record.update({f"train_{k}": v / n_train for k, v in totals.items()})
        history.append(record)
        print(
            f"Fold {fold} Epoch {epoch + 1}/{args.epochs}: "
            f"Loss={record['train_loss']:.4f} CE={record['train_ce']:.4f} "
            f"KL={record['train_cons']:.4f} Budget={record['train_budget']:.4f} "
            f"TrainAcc={record['train_acc']:.4f} TrainAUC={train_auc:.4f} "
            f"ValAcc={val_acc:.4f} ValAUC={val_auc:.4f} | "
            f"reputation_prior={record['reputation_prior_mean']:.4f}+/-{record['reputation_prior_std']:.4f} "
            f"[{record['reputation_prior_min']:.4f},{record['reputation_prior_max']:.4f}] "
            f"morph_lambda={record['prototype_calibration_strength']:.4f} "
            f"H_drift={codebook_drift:.6f}"
        )

        mean_val_ce = val_loss / n_val
        if mean_val_ce < best_val_ce:
            best_val_ce = mean_val_ce
            selected_val_acc = val_acc
            selected_val_auc = val_auc
            best_selector_state = {
                key: value.detach().cpu().clone()
                for key, value in selector.state_dict().items()
            }
            best_classifier_state = {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            }

    if best_selector_state is None or best_classifier_state is None:
        raise RuntimeError(f"Fold {fold}: no valid PCPS checkpoint was produced")
    selector.load_state_dict(best_selector_state)
    classifier.load_state_dict(best_classifier_state)
    return selector, classifier, selected_val_acc, selected_val_auc, history


def export_pcps_outputs(selector, classifier, all_wsis, args, output_dir, fold, device, history):
    os.makedirs(output_dir, exist_ok=True)
    selector.eval()
    classifier.eval()
    result = {}
    patch_diagnostics = {}

    with torch.no_grad():
        for wsi_id, feats, _ in all_wsis:
            feats = feats.to(device)
            keep_ratio = min(1.0, args.selected_count / max(1, feats.shape[0]))
            encoded, task_logits = classifier.compute_patch_evidence(feats)
            out = selector(feats, task_logits, target_keep_ratio=keep_ratio)
            assign = out["prototype_assignment"]
            prior = out["prototype_prior"]
            post = out["task_relevance_posterior"]
            # PCPS has one deterministic selector at export time: the
            # morphology-calibrated posterior q_i.  ABMIL remains the
            # slide-level task head that trains q_i through the reparameterized
            # Concrete gate, but its ungated attention is diagnostic only.
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
                "slide_prototype_profile": assign.mean(dim=0).cpu().tolist(),
                "prototype_task_relevance": cluster_task.cpu().tolist(),
                "task_relevance_mean": float(post.mean().item()),
                "task_relevance_std": float(post.std(unbiased=False).item()),
                "reputation_prior_mean": float(prior.mean().item()),
                "reputation_prior_std": float(prior.std(unbiased=False).item()),
                "prototype_calibration_strength": float(
                    out["prototype_calibration_strength"].item()
                ),
            }

    json_path = os.path.join(output_dir, f"pcps_selection_fold{fold}.json")
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2)

    patch_diag_path = os.path.join(
        output_dir, f"pcps_patch_diagnostics_fold{fold}.pt"
    )
    torch.save(patch_diagnostics, patch_diag_path)

    checkpoint = {
        "pcps_state_dict": selector.state_dict(),
        "task_head_state_dict": classifier.state_dict(),
        "selection_score_name": "task_relevance_posterior",
        "selected_count": args.selected_count,
        "prototype_count": args.prototype_count,
        "prototype_calibration_strength": selector.compute_prototype_calibration_strength().detach().cpu(),
        "prototype_prior": selector.compute_prototype_reputation().detach().cpu(),
        "prototype_prior_raw": torch.sigmoid(
            selector.prototype_reputation_logits.detach()
        ).cpu(),
        "history": history,
        "config": vars(args),
    }
    model_path = os.path.join(output_dir, f"pcps_model_fold{fold}.pt")
    torch.save(checkpoint, model_path)

    diag_path = os.path.join(output_dir, f"pcps_training_diagnostics_fold{fold}.json")
    with open(diag_path, "w") as handle:
        json.dump(
            {
                "history": history,
                "prototype_prior": checkpoint[
                    "prototype_prior"
                ].squeeze(1).tolist(),
            },
            handle,
            indent=2,
        )
    print(f"Saved PCPS selection: {json_path}")
    print(f"Saved PCPS patch diagnostics: {patch_diag_path}")
    print(f"Saved PCPS checkpoint: {model_path}")
    print(f"Saved diagnostics: {diag_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Directory containing per-fold split CSV files")
    parser.add_argument("--split_csv", type=str, default=None,
                        help="Single official split CSV, e.g. BRACS cohort.csv")
    parser.add_argument("--split-pattern", type=str, default="cptac_label_split{fold}.csv",
                        help="Filename pattern used with --split_dir")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--calibration-weight", type=float, default=0.1)
    parser.add_argument("--prototype-reputation-weight", type=float, default=0.1)
    parser.add_argument("--reputation-separation-weight", type=float, default=0.0,
                        help="Deprecated compatibility argument; not used.")
    parser.add_argument("--reputation-bimodal-weight", type=float, default=0.0,
                        help="Deprecated compatibility argument; not used.")
    parser.add_argument("--reputation-jitter-std", type=float, default=0.15)
    parser.add_argument("--reputation-contrast", type=float, default=0.75)
    parser.add_argument("--prototype-calibration-max", type=float, default=0.5)
    parser.add_argument("--prototype-calibration-init", type=float, default=0.1)
    parser.add_argument("--task-head-init", type=str, default=None)
    parser.add_argument("--task-warmup-epochs", type=int, default=0)
    parser.add_argument("--task-lr-factor", type=float, default=1.0)
    parser.add_argument("--selection-budget-weight", type=float, default=1.0)
    parser.add_argument("--codebook-compactness-weight", type=float, default=0.1)
    parser.add_argument("--codebook-anchor-weight", type=float, default=0.01)
    parser.add_argument("--codebook-lr-factor", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--selected-count", type=int, default=384)
    parser.add_argument("--prototype-count", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=None,
                        help="Deprecated; feature dimension is inferred from loaded .pt files")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Infer from training labels when omitted")
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
    if args.split_csv is None and args.split_dir is None:
        raise ValueError("Either --split_csv or --split_dir must be provided")

    if args.split_csv is not None:
        folds = [args.fold if args.fold is not None else 0]
    else:
        folds = [args.fold] if args.fold is not None else range(5)

    for fold in folds:
        if args.split_csv is not None:
            split_csv = args.split_csv
        else:
            split_csv = os.path.join(
                args.split_dir, args.split_pattern.format(fold=fold)
            )
        if not os.path.exists(split_csv):
            print(f"Split not found: {split_csv}, skipping")
            continue
        train_wsis, val_wsis, test_wsis = load_fold_data(
            split_csv, args.feature_dir, fold
        )
        print(
            f"Fold {fold}: {len(train_wsis)} train, "
            f"{len(val_wsis)} val, {len(test_wsis)} test WSIs"
        )
        selector, classifier, best_acc, best_auc, history = train_one_fold(
            train_wsis, val_wsis, args, fold, device
        )
        print(
            f"Fold {fold} selected-checkpoint validation accuracy: {best_acc:.4f}; "
            f"AUC: {best_auc:.4f}"
        )
        export_pcps_outputs(
            selector,
            classifier,
            train_wsis + val_wsis + test_wsis,
            args,
            args.output_dir,
            fold,
            device,
            history,
        )
    print("Done!")


if __name__ == "__main__":
    main()
