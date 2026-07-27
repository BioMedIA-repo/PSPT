"""PSPT evaluation graph."""

import math

import kornia.augmentation as K
import torch
import torch.nn as nn


class PSPTInferenceModel(nn.Module):
    def __init__(
        self,
        backbone,
        classifier,
        evaluation_chunk_size=32,
        normalization_mean=None,
        normalization_std=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.evaluation_chunk_size = int(evaluation_chunk_size)
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

    def _split(self, tensor):
        chunk_count = int(
            math.ceil(tensor.shape[0] / self.evaluation_chunk_size)
        )
        return torch.chunk(tensor, chunk_count, dim=0)

    def normalize(self, images):
        if self.normalization_mean is None:
            return images
        transform = K.Normalize(
            mean=torch.tensor(
                self.normalization_mean, device=images.device
            ),
            std=torch.tensor(
                self.normalization_std, device=images.device
            ),
        )
        for start in range(
            0, images.shape[0], self.evaluation_chunk_size
        ):
            end = start + self.evaluation_chunk_size
            images[start:end] = transform(images[start:end])
        return images

    def forward(self, images):
        chunks = self._split(images)
        context = self.backbone.compute_wsi_initial_token_context(chunks)
        self.backbone.scpm.set_wsi_initial_context(context)
        features = torch.cat(
            [self.backbone(chunk) for chunk in self._split(images)],
            dim=0,
        )
        features = (
            self.backbone.apply_fidelity_preserving_residual_diffusion(
                features
            )
        )
        return self.classifier(features)
