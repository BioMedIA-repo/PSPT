"""CONCH backbone with PSPT prompts."""

import torch
import torch.nn as nn

from conch.open_clip_custom.factory import create_model_from_pretrained

from .scpm_fprd import PSPTCoreMixin


class CONCHPSPTBackbone(nn.Module, PSPTCoreMixin):
    def __init__(
        self,
        backbone_checkpoint,
        image_size=256,
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
        model, _ = create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint_path=str(backbone_checkpoint),
            force_image_size=image_size,
            return_transform=True,
        )
        self.visual = model.visual
        del model.text
        if hasattr(model, "text_projection"):
            del model.text_projection

        self.trunk = self.visual.trunk
        self.num_features = 512
        self._init_pspt_core(
            prompt_dim=self.trunk.embed_dim,
            num_layers=len(self.trunk.blocks),
            num_tokens=num_prompt_tokens,
            prompt_dropout=prompt_dropout,
            scpm_latent_dim=scpm_latent_dim,
            fprd_enabled=fprd_enabled,
            fprd_neighbors=fprd_neighbors,
            fprd_temperature=fprd_temperature,
            fprd_time_max=fprd_time_max,
            fprd_iterations=fprd_iterations,
        )

    def _embed_without_prompts(self, images):
        tokens = self.trunk.patch_embed(images)
        tokens = self.trunk._pos_embed(tokens)
        tokens = self.trunk.patch_drop(tokens)
        return self.trunk.norm_pre(tokens)

    @torch.no_grad()
    def compute_wsi_initial_token_context(self, data_chunks):
        token_sum = None
        token_count = 0
        prefix = getattr(self.trunk, "num_prefix_tokens", 1)
        for images in data_chunks:
            tokens = self._embed_without_prompts(images)
            patch_tokens = tokens[:, prefix:, :].float()
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

    def _forward_tokens(self, images):
        batch_size = images.shape[0]
        prefix = getattr(self.trunk, "num_prefix_tokens", 1)
        tokens = self._embed_without_prompts(images)
        self.scpm.reset_state()

        prompt = self.prompt_dropout(
            self._layer_prompt(tokens, 0, prefix).unsqueeze(0)
        ).expand(batch_size, -1, -1)
        tokens = torch.cat(
            [
                tokens[:, :prefix, :],
                prompt,
                tokens[:, prefix:, :],
            ],
            dim=1,
        )

        for layer_index, block in enumerate(self.trunk.blocks):
            if layer_index > 0:
                prompt = self.prompt_dropout(
                    self._layer_prompt(
                        tokens,
                        layer_index,
                        prefix + self.num_prompt_tokens,
                    ).unsqueeze(0)
                ).expand(batch_size, -1, -1)
                tokens = torch.cat(
                    [
                        tokens[:, :prefix, :],
                        prompt,
                        tokens[
                            :,
                            prefix + self.num_prompt_tokens :,
                            :,
                        ],
                    ],
                    dim=1,
                )
            tokens = block(tokens)
        return self.trunk.norm(tokens)

    def forward(self, images):
        tokens = self._forward_tokens(images)
        if self.visual.use_attentional_pool_contrast:
            pooled = self.visual.attn_pool_contrast(tokens)[:, 0]
            pooled = self.visual.ln_contrast(pooled)
            return pooled @ self.visual.proj_contrast
        pooled, _ = self.visual._global_pool(tokens)
        return self.visual.forward_project(pooled)

