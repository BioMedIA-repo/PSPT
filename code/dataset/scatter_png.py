"""Individual patch PNG reader with the recovered RGB and [0,1] convention."""
from pathlib import Path

import cv2
import kornia as K
import numpy as np
import torch


def load_scatter_patches(scatter_root, wsi_id, indices):
    root = Path(scatter_root) / str(wsi_id)
    indices = sorted(indices)
    if not indices:
        raise ValueError(f'{wsi_id}: empty patch bag')
    images = []
    for index in indices:
        path = root / f'{wsi_id}_patch_{index}.png'
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if images and image.shape != images[0].shape:
            raise ValueError(f'{wsi_id}: inconsistent patch dimensions')
        images.append(image)
    tensor = K.utils.image_to_tensor(np.stack(images))
    return K.enhance.normalize(tensor, torch.tensor(0.), torch.tensor(255.))
