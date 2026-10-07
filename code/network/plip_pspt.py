import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from .pspt_core import PSPTCoreMixin


def interpolate_pos_embed(model, image_size=256):
    vision_model = model.vision_model
    patch_size = vision_model.embeddings.patch_embedding.weight.shape[2]
    num_patches = (image_size // patch_size) ** 2
    num_positions = num_patches + 1

    old_pos_embed = vision_model.embeddings.position_embedding.weight
    if old_pos_embed.shape[0] == num_positions:
        return

    cls_pos = old_pos_embed[0:1, :]
    grid_pos = old_pos_embed[1:, :]
    orig_grid_size = int(math.sqrt(grid_pos.shape[0]))
    new_grid_size = image_size // patch_size

    grid_pos = grid_pos.reshape(1, orig_grid_size, orig_grid_size, -1).permute(0, 3, 1, 2)
    grid_pos = F.interpolate(
        grid_pos, size=(new_grid_size, new_grid_size),
        mode="bicubic", align_corners=False
    )
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(-1, old_pos_embed.shape[-1])

    new_pos_embed = torch.cat([cls_pos, grid_pos], dim=0)
    vision_model.embeddings.position_embedding = nn.Embedding(
        num_positions, old_pos_embed.shape[-1]
    )
    vision_model.embeddings.position_embedding.weight.data.copy_(new_pos_embed)

    if hasattr(vision_model.embeddings, "position_ids"):
        del vision_model.embeddings.position_ids
    vision_model.embeddings.register_buffer(
        "position_ids", torch.arange(num_positions).expand((1, -1)),
        persistent=False
    )


class PLIP_PSPT(nn.Module, PSPTCoreMixin):


    def __init__(
        self,
        checkpoint_path="",
        image_size=224,
        num_tokens=1,
        drop_out=0.0,
        scpm_latent_dim=128,
        scpm_enabled=True,
        fprd_enabled=False,
        fprd_neighbors=16,
        fprd_temperature=0.2,
        fprd_time_max=0.5,
        fprd_iterations=3,
    ):
        super().__init__()
        if os.path.isfile(checkpoint_path):
            checkpoint_path = os.path.dirname(checkpoint_path)

        model = CLIPModel.from_pretrained(checkpoint_path)
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection
        self.num_features = 512
        self.scpm_enabled = bool(scpm_enabled)
        self.image_size = int(image_size)

        del model.text_model
        del model.text_projection

        if image_size != 224:
            interpolate_pos_embed(model, image_size)

        prompt_dim = self.vision_model.config.hidden_size
        self._init_pspt_core(
            prompt_dim=prompt_dim,
            num_layers=len(self.vision_model.encoder.layers),
            num_tokens=num_tokens,
            drop_out=drop_out,
            scpm_latent_dim=scpm_latent_dim,
            fprd_enabled=fprd_enabled,
            fprd_neighbors=fprd_neighbors,
            fprd_temperature=fprd_temperature,
            fprd_time_max=fprd_time_max,
            fprd_iterations=fprd_iterations,
        )

    def _embed_without_prompts(self, x):
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        hidden_states = self.vision_model.embeddings(x)
        return self.vision_model.pre_layrnorm(hidden_states)

    def _prepare_prompt_for_encoder(self, prompt):
        return self.vision_model.pre_layrnorm(prompt)

    def _reset_prompt_state(self):
        if self.scpm_enabled:
            self.scpm.reset_state()

    def _generate_layer_prompt_from_tokens(self, tokens, layer_idx, patch_start):
        if layer_idx == 0:
            base = self.prompt_embeddings[0]
        else:
            base = self.deep_prompt_embeddings[layer_idx - 1]
        if not self.scpm_enabled:
            return base
        residual = self.scpm(tokens, layer_idx, patch_start)
        return base + residual

    def get_scpm_regularization(self):
        if not self.scpm_enabled:
            return self.prompt_embeddings.new_tensor(0.0)
        return self.scpm.regularization_loss()

    @torch.no_grad()
    def compute_wsi_initial_token_context(self, data_chunks):

        token_sum = None
        token_count = 0
        for data_i in data_chunks:
            tokens = self._embed_without_prompts(data_i)
            patch_tokens = tokens[:, 1:, :].float()
            current_sum = patch_tokens.sum(dim=(0, 1), keepdim=True)
            token_sum = current_sum if token_sum is None else token_sum + current_sum
            token_count += patch_tokens.shape[0] * patch_tokens.shape[1]
        if token_sum is None or token_count == 0:
            raise RuntimeError('Cannot compute WSI context from an empty bag.')
        return token_sum / token_count

    def forward(self, x):
        batch_size = x.shape[0]
        hidden_states = self._embed_without_prompts(x)
        self._reset_prompt_state()

        first_prompt = self.prompt_dropout(
            self._generate_layer_prompt_from_tokens(hidden_states, 0, 1).unsqueeze(0)
        ).expand(batch_size, -1, -1)
        first_prompt = self._prepare_prompt_for_encoder(first_prompt)
        hidden_states = torch.cat(
            [hidden_states[:, :1, :], first_prompt, hidden_states[:, 1:, :]], dim=1
        )

        for layer_idx, encoder_layer in enumerate(self.vision_model.encoder.layers):
            if layer_idx > 0:
                layer_prompt = self.prompt_dropout(
                    self._generate_layer_prompt_from_tokens(
                        hidden_states, layer_idx, 1 + self.num_prompt_tokens
                    ).unsqueeze(0)
                ).expand(batch_size, -1, -1)
                layer_prompt = self._prepare_prompt_for_encoder(layer_prompt)
                hidden_states = torch.cat(
                    [
                        hidden_states[:, :1, :],
                        layer_prompt,
                        hidden_states[:, 1 + self.num_prompt_tokens :, :],
                    ],
                    dim=1,
                )
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )[0]

        if self.scpm_enabled:
            self.scpm.finalize_diagnostics()
        pooled_output = hidden_states[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)
        return self.visual_projection(pooled_output)
