"""Command-line evaluation for PSPT inference checkpoints."""

import argparse
import json
from pathlib import Path

import torch

from .data import (
    load_scatter_patches,
    read_bright_split,
)
from .metrics import summarize
from .models import load_pspt_model
from .pcps import PCPSArtifacts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--scatter-png-dir", required=True, type=Path)
    parser.add_argument("--pcps-scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conch-backbone-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-count", type=int)
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--sampling-seed", type=int)
    return parser.parse_args()


def evaluate_split(
    model,
    table,
    artifacts,
    scatter_root,
    split_name,
    patch_count,
    kappa,
    sampling_seed,
    device,
):
    offset = 0 if split_name == "validation" else 10_000_019
    records = []
    with torch.inference_mode():
        for item_index, row in table.iterrows():
            wsi_id = str(row["wsi_id"])
            indices = artifacts.sample(
                wsi_id=wsi_id,
                patch_count=patch_count,
                kappa=kappa,
                sampling_seed=sampling_seed,
                split_offset=offset,
                item_index=item_index,
            )
            images = load_scatter_patches(
                scatter_root, wsi_id, indices
            ).to(device)
            images = model.normalize(images)
            if str(device).startswith("cuda"):
                context = torch.autocast(
                    device_type="cuda", dtype=torch.float16
                )
            else:
                context = torch.autocast(
                    device_type="cpu", enabled=False
                )
            with context:
                _, probability = model(images)
            records.append(
                {
                    "wsi_id": wsi_id,
                    "label": int(row["label"]),
                    "probability": (
                        probability.detach()
                        .float()
                        .cpu()
                        .reshape(-1)
                        .tolist()
                    ),
                    "selected_patch_count": len(indices),
                }
            )
            print(
                f"{split_name}: {item_index + 1}/{len(table)} "
                f"{wsi_id}",
                flush=True,
            )
    return records


def main():
    args = parse_args()
    model, config = load_pspt_model(
        args.checkpoint,
        conch_backbone_checkpoint=args.conch_backbone_checkpoint,
        device=args.device,
    )
    protocol = config["pcps"]
    patch_count = (
        protocol["patch_count"]
        if args.patch_count is None
        else args.patch_count
    )
    kappa = (
        protocol["kappa"] if args.kappa is None else args.kappa
    )
    sampling_seed = (
        protocol["evaluation_sampling_seed"]
        if args.sampling_seed is None
        else args.sampling_seed
    )
    artifacts = PCPSArtifacts(args.pcps_scores)

    outputs = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        table = read_bright_split(
            args.split_csv,
            split,
            validation_fold=config["data"]["validation_fold"],
        )
        records = evaluate_split(
            model=model,
            table=table,
            artifacts=artifacts,
            scatter_root=args.scatter_png_dir,
            split_name=split,
            patch_count=patch_count,
            kappa=kappa,
            sampling_seed=sampling_seed,
            device=args.device,
        )
        (args.output_dir / f"{split}_predictions.json").write_text(
            json.dumps(records, indent=2)
        )
        outputs[split] = summarize(
            records,
            class_bias=config["inference"].get(
                "validation_class_bias"
            ),
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "encoder": config["encoder"],
        "protocol": {
            "patch_count": patch_count,
            "kappa": kappa,
            "sampling_seed": sampling_seed,
        },
        **outputs,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
