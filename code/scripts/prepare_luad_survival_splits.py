#!/usr/bin/env python3
"""Convert locked CLAM LUAD survival splits to PSPT WSI-level CSVs."""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def assign_time_bins(times, edges):
    # Null values are serialized infinite endpoints; strip endpoints once.
    if len(edges) < 3:
        raise ValueError("Expected endpoints and interior thresholds")
    thresholds = np.asarray(edges[1:-1], dtype=float)
    if not np.isfinite(thresholds).all() or not (np.diff(thresholds) > 0).all():
        raise ValueError("Interior thresholds must be finite and strictly increasing")
    values = np.asarray(times, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Survival times must be finite and nonnegative")
    return np.searchsorted(thresholds, values, side="right").astype(int)


def count_patches(patch_dir, slide_id):
    path = patch_dir / f"{slide_id}.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        return int(handle["coords"].shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clam-split-dir",
        required=True,
    )
    parser.add_argument(
        "--patch-dir",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    split_dir = Path(args.clam_split_dir)
    patch_dir = Path(args.patch_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for fold in range(args.folds):
        split_csv = split_dir / f"split_{fold}.csv"
        bins_json = split_dir / f"split_{fold}_bins.json"
        if not split_csv.exists():
            raise FileNotFoundError(split_csv)
        if not bins_json.exists():
            raise FileNotFoundError(bins_json)

        cases = pd.read_csv(split_csv)
        with open(bins_json, "r", encoding="utf-8") as handle:
            edges = json.load(handle)

        rows = []
        for _, row in cases.iterrows():
            slide_ids = str(row["slide_ids"]).split(";")
            time_bin = int(assign_time_bins([float(row["survival_months"])], edges)[0])
            for slide_id in slide_ids:
                rows.append({
                    "case_id": row["case_id"],
                    "slide_id": slide_id,
                    "wsi_id": slide_id,
                    "event": int(row["event"]),
                    "censorship": int(1 - int(row["event"])),
                    "survival_months": float(row["survival_months"]),
                    "time_bin": time_bin,
                    "split": row["split"],
                    "fold": 1 if row["split"] == "train" else (0 if row["split"] == "val" else -1),
                    "is_test": 1 if row["split"] == "test" else 0,
                    "label": time_bin,
                    "len_img": count_patches(patch_dir, slide_id),
                })

        out_csv = output_dir / f"luad_survival_split43_fold{fold}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        out_bins = output_dir / f"luad_survival_split43_fold{fold}_bins.json"
        with open(out_bins, "w", encoding="utf-8") as handle:
            json.dump(edges, handle)

        frame = pd.DataFrame(rows)
        for split, group in frame.groupby("split"):
            summary_rows.append({
                "fold": fold,
                "split": split,
                "slides": int(len(group)),
                "cases": int(group["case_id"].nunique()),
                "events": int(group.drop_duplicates("case_id")["event"].sum()),
                "mean_patches": float(group["len_img"].mean()),
                "median_patches": float(group["len_img"].median()),
            })
        print(f"Saved {out_csv} ({len(rows)} slides)")

    summary = pd.DataFrame(summary_rows).sort_values(["fold", "split"])
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
