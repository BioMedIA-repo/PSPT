"""BRACS scatter-PNG reader."""

from pathlib import Path

import cv2
import kornia as K
import numpy as np
import pandas as pd
import torch


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_bright_split(split_csv, split, validation_fold=0):
    table = pd.read_csv(split_csv)
    required = {"wsi_id", "label", "fold"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"Split CSV is missing columns: {sorted(missing)}"
        )
    if split == "validation":
        table = table[table["fold"] == validation_fold]
    elif split == "test":
        table = table[table["fold"] < 0]
    else:
        raise ValueError("Inference split must be validation or test.")
    return table.reset_index(drop=True)


def load_scatter_patches(scatter_root, wsi_id, indices):
    root = Path(scatter_root) / str(wsi_id)
    images = []
    for index in sorted(indices):
        images.append(
            read_rgb(root / f"{wsi_id}_patch_{index}.png")
        )
    if not images:
        raise ValueError(f"{wsi_id}: PCPS selected an empty bag.")
    array = np.stack(images)
    images = K.utils.image_to_tensor(array)
    return K.enhance.normalize(
        images,
        torch.tensor(0.0),
        torch.tensor(255.0),
    )
