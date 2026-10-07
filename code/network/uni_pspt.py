import torch
import torch.nn as nn
import timm

from .pspt_core import SlideAwareCrossLayerPromptModulation


class PSPTBackbone(nn.Module):
    """UNI VPT-Deep with layer-wise SCPM prompts and FPRD.

    SCPM shares a sampled-WSI token initialization across memory-bounded image
    chunks, then applies layer-wise local corrections within each chunk.
    """

    def __init__(
        self,
        num_tokens=1,
        drop_out=0.0,
        scpm_latent_dim=128,
        scpm_enabled=True,
        fprd_enabled=False,
        fprd_neighbors=16,
        fprd_temperature=0.2,
        fprd_time_max=0.5,
        fprd_iterations=3,
        **kwargs,
    ):
        super().__init__()
        self.num_prompt_tokens = num_tokens
        self.deep_prompt = True
        self.scpm_enabled = bool(scpm_enabled)
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
        self.prompt_dim = self.num_features

        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, num_tokens, self.num_features)
        )
        self.deep_prompt_embeddings = nn.Parameter(
            torch.zeros(self.num_layers - 1, num_tokens, self.num_features)
        )
        nn.init.trunc_normal_(self.prompt_embeddings, std=0.02)
        nn.init.trunc_normal_(self.deep_prompt_embeddings, std=0.02)
        self.prompt_dropout = nn.Dropout(drop_out)

        self.scpm = SlideAwareCrossLayerPromptModulation(
            prompt_dim=self.num_features,
            latent_dim=scpm_latent_dim,
            num_tokens=num_tokens,
            num_layers=self.num_layers,
        )
        import math
        self.fprd_time_logit = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))
        self.last_diffusion_time = torch.tensor(0.1)

    @torch.no_grad()
    def compute_wsi_initial_token_context(self, data_chunks):
        """Streaming mean of pre-block patch tokens over the sampled WSI bag."""
        token_sum = None
        token_count = 0
        for data_i in data_chunks:
            tokens = self.vit._pos_embed(self.vit.patch_embed(data_i))
            patch_tokens = tokens[:, 1:, :].float()
            current_sum = patch_tokens.sum(dim=(0, 1), keepdim=True)
            token_sum = current_sum if token_sum is None else token_sum + current_sum
            token_count += patch_tokens.shape[0] * patch_tokens.shape[1]
        if token_sum is None or token_count == 0:
            raise RuntimeError('Cannot compute WSI context from an empty bag.')
        return token_sum / token_count

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

    def compute_diffusion_time(self):
        return self.fprd_time_max * torch.sigmoid(self.fprd_time_logit)

    def apply_fidelity_preserving_residual_diffusion(self, features):
        from .fprd import apply_fprd
        return apply_fprd(self, features)


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.vit._pos_embed(self.vit.patch_embed(x))
        self._reset_prompt_state()

        first_prompt = self.prompt_dropout(
            self._generate_layer_prompt_from_tokens(x, 0, 1).unsqueeze(0)
        ).expand(batch_size, -1, -1)
        x = torch.cat([x[:, :1, :], first_prompt, x[:, 1:, :]], dim=1)

        for layer_idx, block in enumerate(self.vit.blocks):
            if layer_idx > 0:
                layer_prompt = self.prompt_dropout(
                    self._generate_layer_prompt_from_tokens(
                        x, layer_idx, 1 + self.num_prompt_tokens
                    ).unsqueeze(0)
                ).expand(batch_size, -1, -1)
                x = torch.cat(
                    [
                        x[:, :1, :],
                        layer_prompt,
                        x[:, 1 + self.num_prompt_tokens :, :],
                    ],
                    dim=1,
                )
            x = block(x)

        if self.scpm_enabled:
            self.scpm.finalize_diagnostics()
        x = self.vit.norm(x)
        return x[:, 0]
