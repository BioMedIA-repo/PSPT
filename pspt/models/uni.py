"""UNI backbone with PSPT prompts."""

import torch
import torch.nn as nn
import timm

from .scpm_fprd import (
    PSPTCoreMixin,
    SlideAwareCrossLayerPromptModulation,
)


class UNIPSPTBackbone(nn.Module, PSPTCoreMixin):
    def __init__(
        self,
        num_prompt_tokens=1,
        prompt_dropout=0.0,
        scpm_latent_dim=128,
        fprd_enabled=True,
        fprd_neighbors=16,
        fprd_temperature=0.2,
        fprd_time_max=0.5,
        fprd_iterations=3,
    ):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.scpm_enabled = True
        self.fprd_enabled = fprd_enabled
        self.fprd_neighbors = fprd_neighbors
        self.fprd_temperature = fprd_temperature
        self.fprd_time_max = fprd_time_max
        self.fprd_iterations = fprd_iterations

        self.vit = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.num_features = self.vit.embed_dim
        self.num_layers = len(self.vit.blocks)

        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, num_prompt_tokens, self.num_features)
        )
        self.deep_prompt_embeddings = nn.Parameter(
            torch.zeros(
                self.num_layers - 1,
                num_prompt_tokens,
                self.num_features,
            )
        )
        nn.init.trunc_normal_(self.prompt_embeddings, std=0.02)
        nn.init.trunc_normal_(self.deep_prompt_embeddings, std=0.02)
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.scpm = SlideAwareCrossLayerPromptModulation(
            prompt_dim=self.num_features,
            latent_dim=scpm_latent_dim,
            num_tokens=num_prompt_tokens,
            num_layers=self.num_layers,
        )

        import math

        self.fprd_time_logit = nn.Parameter(
            torch.tensor(math.log(0.2 / 0.8))
        )
        self.last_diffusion_time = torch.tensor(0.1)

    @torch.no_grad()
    def compute_wsi_initial_token_context(self, data_chunks):
        token_sum = None
        token_count = 0
        for images in data_chunks:
            tokens = self.vit._pos_embed(self.vit.patch_embed(images))
            patch_tokens = tokens[:, 1:, :].float()
            current = patch_tokens.sum(dim=(0, 1), keepdim=True)
            token_sum = current if token_sum is None else token_sum + current
            token_count += (
                patch_tokens.shape[0] * patch_tokens.shape[1]
            )
        if token_sum is None or token_count == 0:
            raise RuntimeError("Cannot encode an empty WSI bag.")
        return token_sum / token_count

    def _layer_prompt(self, tokens, layer_index, patch_start):
        base = (
            self.prompt_embeddings[0]
            if layer_index == 0
            else self.deep_prompt_embeddings[layer_index - 1]
        )
        return base + self.scpm(tokens, layer_index, patch_start)

    def forward(self, images):
        batch_size = images.shape[0]
        tokens = self.vit._pos_embed(self.vit.patch_embed(images))
        self.scpm.reset_state()

        prompt = self.prompt_dropout(
            self._layer_prompt(tokens, 0, 1).unsqueeze(0)
        ).expand(batch_size, -1, -1)
        tokens = torch.cat(
            [tokens[:, :1, :], prompt, tokens[:, 1:, :]], dim=1
        )

        for layer_index, block in enumerate(self.vit.blocks):
            if layer_index > 0:
                prompt = self.prompt_dropout(
                    self._layer_prompt(
                        tokens,
                        layer_index,
                        1 + self.num_prompt_tokens,
                    ).unsqueeze(0)
                ).expand(batch_size, -1, -1)
                tokens = torch.cat(
                    [
                        tokens[:, :1, :],
                        prompt,
                        tokens[:, 1 + self.num_prompt_tokens :, :],
                    ],
                    dim=1,
                )
            tokens = block(tokens)
        return self.vit.norm(tokens)[:, 0]

