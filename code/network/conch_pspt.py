import torch
import torch.nn as nn

from conch.open_clip_custom.factory import create_model_from_pretrained

from .pspt_core import PSPTCoreMixin


class CONCH_PSPT(nn.Module, PSPTCoreMixin):


    def __init__(
        self,
        checkpoint_path="",
        image_size=256,
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
        model, _ = create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint_path=checkpoint_path,
            force_image_size=image_size,
            return_transform=True,
        )
        self.visual = model.visual
        del model.text
        if hasattr(model, "text_projection"):
            del model.text_projection

        self.trunk = self.visual.trunk
        prompt_dim = self.trunk.embed_dim
        self.num_features = 512
        self.scpm_enabled = bool(scpm_enabled)
        self._init_pspt_core(
            prompt_dim=prompt_dim,
            num_layers=len(self.trunk.blocks),
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
        x = self.trunk.patch_embed(x)
        x = self.trunk._pos_embed(x)
        x = self.trunk.patch_drop(x)
        x = self.trunk.norm_pre(x)
        return x

    @torch.no_grad()
    def compute_wsi_initial_token_context(self, data_chunks):

        token_sum = None
        token_count = 0
        prefix_tokens = getattr(self.trunk, "num_prefix_tokens", 1)
        for data_i in data_chunks:
            tokens = self._embed_without_prompts(data_i)
            patch_tokens = tokens[:, prefix_tokens:, :].float()
            current_sum = patch_tokens.sum(dim=(0, 1), keepdim=True)
            token_sum = current_sum if token_sum is None else token_sum + current_sum
            token_count += patch_tokens.shape[0] * patch_tokens.shape[1]
        if token_sum is None or token_count == 0:
            raise RuntimeError('Cannot compute WSI context from an empty bag.')
        return token_sum / token_count

    def _forward_trunk_with_prompts(self, x):
        batch_size = x.shape[0]
        prefix_tokens = getattr(self.trunk, "num_prefix_tokens", 1)
        x = self._embed_without_prompts(x)
        self._reset_prompt_state()

        first_prompt = self.prompt_dropout(
            self._generate_layer_prompt_from_tokens(x, 0, prefix_tokens).unsqueeze(0)
        ).expand(batch_size, -1, -1)
        x = torch.cat([x[:, :prefix_tokens, :], first_prompt, x[:, prefix_tokens:, :]], dim=1)

        for layer_idx, block in enumerate(self.trunk.blocks):
            if layer_idx > 0:
                layer_prompt = self.prompt_dropout(
                    self._generate_layer_prompt_from_tokens(
                        x, layer_idx, prefix_tokens + self.num_prompt_tokens
                    ).unsqueeze(0)
                ).expand(batch_size, -1, -1)
                x = torch.cat(
                    [
                        x[:, :prefix_tokens, :],
                        layer_prompt,
                        x[:, prefix_tokens + self.num_prompt_tokens :, :],
                    ],
                    dim=1,
                )
            x = block(x)
        if self.scpm_enabled:
            self.scpm.finalize_diagnostics()
        return self.trunk.norm(x)

    def forward(self, x):
        tokens = self._forward_trunk_with_prompts(x)
        if self.visual.use_attentional_pool_contrast:
            pooled = self.visual.attn_pool_contrast(tokens)[:, 0]
            pooled = self.visual.ln_contrast(pooled)
            pooled = pooled @ self.visual.proj_contrast
            return pooled
        pooled, _ = self.visual._global_pool(tokens)
        return self.visual.forward_project(pooled)
