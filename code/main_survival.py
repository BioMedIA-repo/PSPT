import argparse
import os
import re

import kornia.augmentation as K
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from dataset.survival_wsi_dataset import SurvivalPatchWsiDataModule, summarize_survival_split
from main import get_network
from network.model_clam_survival import CLAMSurvivalHead
from options import get_arguments_additional
from pl_model.mil_survival_trainer import MilSurvivalModule
from pl_model.runtime_checks import precision_options, verify_optimizer_updates
from utils import save_parameters


def get_transforms(args):
    mean = args.data_mean if args.data_mean is not None else [0.485, 0.456, 0.406]
    std = args.data_std if args.data_std is not None else [0.229, 0.224, 0.225]
    if args.data_norm:
        return (
            nn.Sequential(K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))),
            nn.Sequential(K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))),
        )
    return nn.Sequential(), nn.Sequential()


def get_model(args, backbone, num_fts):
    if args.model not in ("clam_sb", "clam_mb"):
        raise NotImplementedError("Survival PSPT currently supports clam_sb/clam_mb heads")
    head = CLAMSurvivalHead(
        size=args.clam_size,
        dropout=True,
        n_bins=args.n_bins,
        gate=True,
    )
    return MilSurvivalModule(backbone, head, get_transforms(args), args)


def _checkpoint_epoch(path):
    match = re.search(r"epoch=(\d+)", path or "")
    return int(match.group(1)) + 1 if match else None


def _score_value(score):
    return None if score is None else float(score.detach().cpu().item())


def save_test_results_to_csv(args, test_results, checkpoint_results=None):
    if not test_results:
        return
    row = {
        "dataset": args.dataset_name,
        "run_name": args.run_name,
        "tag": args.tag,
        "fold": (
            args.survival_split_index
            if args.survival_split_index is not None
            else args.val_fold
        ),
        "model": args.model,
        "network": args.network,
        "transfer_type": args.transfer_type,
    }
    row.update(test_results[0])
    if checkpoint_results:
        row.update(checkpoint_results)
    out_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "all_survival_experiments_results.csv")
    frame = pd.DataFrame([row])
    frame.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
    print(f"Saved survival result: {out_csv}")


def validate_inputs(args):
    if args.survival_test_both_checkpoints:
        raise ValueError("New study tests only one validation-selected checkpoint")
    if not args.scatter_png_dir or not os.path.isdir(args.scatter_png_dir):
        raise FileNotFoundError(
            "Missing --scatter-png-dir. LUAD PSPT end-to-end needs exported patch PNGs."
        )
    if args.pcps_selection_path and not os.path.exists(args.pcps_selection_path):
        raise FileNotFoundError(f"Missing PCPS selection: {args.pcps_selection_path}")
    if not args.pcps_selection_path and args.pcps_eval_mode != "full" and args.pcps_sampling_mode != "uniform":
        raise ValueError("Without PCPS, use full-bag training and evaluation")
    if not os.path.exists(args.dataset_csv):
        raise FileNotFoundError(args.dataset_csv)
    frame = pd.read_csv(args.dataset_csv)
    if not frame.time_bin.between(0, args.n_bins - 1).all():
        raise ValueError("time_bin outside configured survival head range")
    if not (frame.event + frame.censorship == 1).all():
        raise ValueError("event/censorship labels disagree")
    if not frame.event.isin([0, 1]).all() or (frame.survival_months < 0).any():
        raise ValueError("Invalid clinical labels")
    if (frame.groupby('case_id')[['split', 'event', 'survival_months']].nunique() > 1).any().any():
        raise ValueError("Inconsistent patient labels/partitions")
    summary = summarize_survival_split(args.dataset_csv)
    print("\nSurvival split summary:")
    print(summary.to_string(index=False))


def main(args):
    validate_inputs(args)
    pl.seed_everything(args.seed, workers=True)
    data_module = SurvivalPatchWsiDataModule(
        args.dataset_root,
        args.dataset_csv,
        val_fold=args.val_fold,
        num_workers=args.num_workers,
        num_workers_eval=args.num_workers_eval,
        drop_out=args.dropout_inst,
        weighted_sample=False,
        pcps_selection_path=args.pcps_selection_path,
        pcps_selected_count=args.pcps_selected_count,
        pcps_random_count=args.pcps_random_count,
        pcps_sampling_mode=args.pcps_sampling_mode,
        pcps_total_count=args.pcps_total_count,
        pcps_evidence_concentration=args.pcps_evidence_concentration,
        pcps_context_temperature=args.pcps_context_temperature,
        pcps_distribution_concentration=args.pcps_distribution_concentration,
        pcps_eval_mode=args.pcps_eval_mode,
        pcps_eval_sampling_seed=(
            args.seed if args.pcps_eval_sampling_seed is None
            else args.pcps_eval_sampling_seed
        ),
        scatter_png_dir=args.scatter_png_dir,
    )
    print(
        "PCPS evaluation policy: "
        f"mode={args.pcps_eval_mode}, M={args.pcps_total_count}, "
        f"kappa={args.pcps_distribution_concentration}, "
        f"seed={args.seed if args.pcps_eval_sampling_seed is None else args.pcps_eval_sampling_seed}"
    )
    backbone, num_fts = get_network(args)
    model = get_model(args, backbone, num_fts)

    ckpt_dir = os.path.join(args.output_dir, args.run_name, args.tag, "checkpoints")
    checkpoint_val_loss = ModelCheckpoint(
        monitor="Loss/val", dirpath=ckpt_dir,
        filename="best-val-loss-epoch={epoch:02d}-loss={Loss/val:.4f}",
        save_top_k=1, mode="min",
        save_last=not args.survival_disable_last_checkpoint,
        auto_insert_metric_name=False,
    )
    checkpoint_val_cindex = ModelCheckpoint(
        monitor="C-index/val", dirpath=ckpt_dir,
        filename="best-val-cindex-epoch={epoch:02d}-cindex={C-index/val:.4f}",
        save_top_k=1, mode="max", auto_insert_metric_name=False,
    )
    logger = TensorBoardLogger(save_dir=os.path.join(args.output_dir, args.run_name), name=args.tag)
    trainer = pl.Trainer(
        default_root_dir=os.path.join(args.output_dir, args.run_name),
        max_epochs=args.epochs,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        **precision_options(args),
        accelerator="gpu",
        devices=args.gpu_id,
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="epoch"), checkpoint_val_loss, checkpoint_val_cindex],
    )
    trainer.fit(model, data_module)
    verify_optimizer_updates(trainer, model, args)
    print(f"Best val-loss checkpoint: {checkpoint_val_loss.best_model_path}")
    print(f"Best val-cindex checkpoint: {checkpoint_val_cindex.best_model_path}")
    criterion = getattr(args, "survival_checkpoint_metric", "cindex")
    chosen = checkpoint_val_cindex if criterion == "cindex" else checkpoint_val_loss
    if not chosen.best_model_path:
        raise RuntimeError("No validation-selected checkpoint; test was not run")
    test_results = trainer.test(model, data_module, ckpt_path=chosen.best_model_path)
    checkpoint_results = {
        "checkpoint/best_val_loss_epoch": _checkpoint_epoch(checkpoint_val_loss.best_model_path),
        "Loss/best_val_loss": _score_value(checkpoint_val_loss.best_model_score),
        "checkpoint/best_val_cindex_epoch": _checkpoint_epoch(checkpoint_val_cindex.best_model_path),
        "C-index/best_val_cindex": _score_value(checkpoint_val_cindex.best_model_score),
    }
    checkpoint_results["checkpoint/test_selection_criterion"] = criterion
    save_test_results_to_csv(args, test_results, checkpoint_results)


def add_argument_fun(parser):
    parser.add_argument("--survival-checkpoint-metric", choices=["cindex", "loss"], default="cindex")
    parser.add_argument("--clam-size", type=lambda s: [int(item) for item in s.split(",")],
                        default=[1024, 256, 128])
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--survival-alpha", type=float, default=0.0)
    parser.add_argument("--survival-eval-level", choices=["slide", "case"], default="slide",
                        help="slide computes C-index over WSI predictions; case averages WSI risks within each patient.")
    parser.add_argument(
        "--survival-split-index", type=int, default=None,
        help="Reported outer split index. This is separate from --val_fold when a CSV encodes train/val/test roles.",
    )
    parser.add_argument(
        "--survival-test-both-checkpoints", action="store_true",
        help="Also test and record the checkpoint selected by validation C-index.",
    )
    parser.add_argument(
        "--survival-disable-last-checkpoint", action="store_true",
        help="Keep only best-val-loss and best-val-cindex checkpoints to reduce disk usage.",
    )
    return parser


def process_argument_fun(opts):
    return opts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_arguments_additional(parser, add_argument_fun, process_argument_fun)
    save_parameters(args)
    main(args)
