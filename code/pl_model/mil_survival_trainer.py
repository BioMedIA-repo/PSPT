import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


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


def _as_scalar(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    if torch.is_tensor(value):
        return value.detach().cpu().view(-1)[0].item()
    return value


from .streaming import NativePatchMixin


class MilSurvivalModule(NativePatchMixin, pl.LightningModule):
    def __init__(self, backbone, survival_head, transforms, args):
        super().__init__()
        self.automatic_optimization = False
        self.backbone = backbone
        self.classifier = survival_head
        self.transforms_train, self.transforms_eval = transforms
        self.args = args
        self.freeze_backbone = args.transfer_type in ["frozen", "mil"]
        self.batch_size_train = args.batch_size_train
        self.batch_size_eval = args.batch_size_eval
        self.accumulate_grad_batches = args.accumulate_grad_batches
        self.lr = args.lr
        self.val_outputs = {0: [], 1: []}
        self.test_outputs = []
        self.save_hyperparameters("args")

    def split_tensor(self, data, batch_size):
        num_chunks = int(np.ceil(data.shape[0] / batch_size))
        return torch.chunk(data, num_chunks, dim=0)


    def _prepare_wsi_prompt_context(self, data):
        if not getattr(self.backbone, "scpm_enabled", True):
            return
        scpm = getattr(self.backbone, "scpm", None)
        compute_context = getattr(self.backbone, "compute_wsi_initial_token_context", None)
        if scpm is None or compute_context is None:
            return
        chunk_size = self.batch_size_train if self.training else self.batch_size_eval
        context = compute_context(self.split_tensor(data, chunk_size))
        scpm.set_wsi_initial_context(context)

    def backbone_forward(self, data):
        features = []
        chunk_size = self.batch_size_train if self.training else self.batch_size_eval
        with torch.no_grad():
            for data_i in self.split_tensor(data, chunk_size):
                features.append(self.backbone(data_i))
        return torch.cat(features, dim=0)

    def backbone_backward(self, data, features):
        feature_grads = features.grad
        if feature_grads is None:
            raise RuntimeError("Missing feature gradient before backbone recomputation.")
        data_chunks = self.split_tensor(data, self.batch_size_train)
        grad_chunks = self.split_tensor(feature_grads, self.batch_size_train)
        for data_i, grad_i in zip(data_chunks, grad_chunks):
            recomputed_features = self.backbone(data_i)
            recomputed_features.backward(grad_i)

    def forward(self, data, target=None, train=False, loss_scale=1.0):
        self._prepare_wsi_prompt_context(data)
        raw_features = self.backbone_forward(data)
        if train and not self.freeze_backbone:
            raw_features = raw_features.detach().requires_grad_(True)

        features = raw_features
        diffusion_time = None
        if getattr(self.args, "enable_fprd", False) and hasattr(
            self.backbone, "apply_fidelity_preserving_residual_diffusion"
        ):
            features = self.backbone.apply_fidelity_preserving_residual_diffusion(raw_features)
            diffusion_time = getattr(self.backbone, "last_diffusion_time", None)
            if train and not self.freeze_backbone:
                features.retain_grad()

        hazards, survival, risk, attention = self.classifier(features)
        loss = None
        if target is not None:
            loss = nll_survival_loss(
                hazards,
                target["time_bin"].to(self.device),
                target["censorship"].to(self.device),
                alpha=self.args.survival_alpha,
            )
        if train:
            self.manual_backward(loss * loss_scale)
            if not self.freeze_backbone:
                self.backbone_backward(data, raw_features)
            if diffusion_time is not None:
                self.log("train/fprd_diffusion_time", diffusion_time.detach(),
                         on_step=True, on_epoch=True, sync_dist=True)
        return hazards, survival, risk, attention, loss

    def _collect_prediction(self, outputs, target, risk, loss):
        case_id = target["case_id"][0] if isinstance(target["case_id"], list) else target["case_id"]
        wsi_id = target["wsi_id"][0] if isinstance(target["wsi_id"], list) else target["wsi_id"]
        outputs.append({
            "case_id": str(case_id),
            "wsi_id": str(wsi_id),
            "event": int(_as_scalar(target["event"])),
            "time": float(_as_scalar(target["time"])),
            "risk": float(risk.detach().cpu().view(-1)[0]),
            "loss": float(loss.detach().cpu()),
        })

    def _aggregate_metrics(self, outputs, prefix):
        if not outputs:
            return
        eval_level = getattr(self.args, "survival_eval_level", "case")
        if eval_level == "slide":
            events = [item["event"] for item in outputs]
            times = [item["time"] for item in outputs]
            risks = [item["risk"] for item in outputs]
            losses = [item["loss"] for item in outputs]
            n_units = len(outputs)
        else:
            grouped = {}
            for item in outputs:
                group = grouped.setdefault(item["case_id"], {
                    "risks": [], "event": item["event"], "time": item["time"], "losses": [],
                })
                group["risks"].append(item["risk"])
                group["losses"].append(item["loss"])
            events, times, risks, losses = [], [], [], []
            for group in grouped.values():
                events.append(group["event"])
                times.append(group["time"])
                risks.append(float(np.mean(group["risks"])))
                losses.extend(group["losses"])
            n_units = len(grouped)
        cindex = concordance_index(events, times, risks)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        self.log(f"Loss/{prefix}", mean_loss, prog_bar=True, sync_dist=False)
        self.log(f"C-index/{prefix}", cindex, prog_bar=True, sync_dist=False)
        self.log(f"{eval_level}s/{prefix}", n_units, sync_dist=False)

    def training_step(self, batch, batch_idx):
        img, target, _, _ = batch
        img = self.augment_data(img, train=True)
        accumulation = max(1, self.accumulate_grad_batches)
        num_batches = int(self.trainer.num_training_batches)
        group_start = (batch_idx // accumulation) * accumulation
        group_size = min(accumulation, num_batches - group_start)
        should_step = ((batch_idx + 1) % accumulation == 0) or (batch_idx + 1 == num_batches)
        _, _, risk, _, loss = self(img, target=target, train=True, loss_scale=1.0 / group_size)
        optimizer = self.optimizers()
        if should_step:
            optimizer.step()
            optimizer.zero_grad()
        self.log("Loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("risk/train", risk.detach().mean(), on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        img, target, _, _ = batch
        img = self.augment_data(img, train=False)
        _, _, risk, _, loss = self(img, target=target, train=False)
        idx = dataloader_idx if dataloader_idx is not None else 0
        self._collect_prediction(self.val_outputs[idx], target, risk, loss)
        return loss

    def on_validation_epoch_start(self):
        self.val_outputs = {0: [], 1: []}

    def on_validation_epoch_end(self):
        self._aggregate_metrics(self.val_outputs[0], "val")
        if self.val_outputs.get(1):
            self._aggregate_metrics(self.val_outputs[1], "test")

    def test_step(self, batch, batch_idx):
        img, target, _, _ = batch
        img = self.augment_data(img, train=False)
        _, _, risk, _, loss = self(img, target=target, train=False)
        self._collect_prediction(self.test_outputs, target, risk, loss)
        return loss

    def on_test_epoch_start(self):
        self.test_outputs = []

    def on_test_epoch_end(self):
        self._aggregate_metrics(self.test_outputs, "final_test")
        from pathlib import Path
        import pandas as pd
        destination = Path(self.args.output_dir) / self.args.run_name / self.args.tag / "diagnostics"
        destination.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.test_outputs)
        frame.to_csv(destination / "test_slide_risk.csv", index=False)
        if not frame.empty:
            if (frame.groupby("case_id")[["event", "time"]].nunique() > 1).any().any():
                raise ValueError("Inconsistent survival targets within a patient")
            patients = frame.groupby("case_id", as_index=False).agg(
                event=("event", "first"), time=("time", "first"),
                risk=("risk", "mean"), n_slides=("wsi_id", "size"))
            patients.to_csv(destination / "test_patient_risk.csv", index=False)

    def configure_optimizers(self):
        if self.args.adam:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr, weight_decay=self.args.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr, weight_decay=self.args.weight_decay,
            )
        return optimizer
