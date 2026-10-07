import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from .scpm import SlideAwareCrossLayerPromptModulation


class PSPTCoreMixin:
    def _init_pspt_core(
        self,
        prompt_dim,
        num_layers,
        num_tokens=1,
        drop_out=0.0,
        scpm_latent_dim=128,
        fprd_enabled=False,
        fprd_neighbors=16,
        fprd_temperature=0.2,
        fprd_time_max=0.5,
        fprd_iterations=3,
    ):
        self.num_prompt_tokens = num_tokens
        self.deep_prompt = True
        self.fprd_enabled = fprd_enabled
        self.fprd_neighbors = fprd_neighbors
        self.fprd_temperature = fprd_temperature
        self.fprd_time_max = fprd_time_max
        self.fprd_iterations = fprd_iterations
        self.prompt_dim = prompt_dim
        self.num_layers = num_layers

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
        self.deep_prompt_embeddings = nn.Parameter(
            torch.zeros(num_layers - 1, num_tokens, prompt_dim)
        )
        nn.init.trunc_normal_(self.prompt_embeddings, std=0.02)
        nn.init.trunc_normal_(self.deep_prompt_embeddings, std=0.02)
        self.prompt_dropout = nn.Dropout(drop_out)

        self.scpm = SlideAwareCrossLayerPromptModulation(
            prompt_dim=prompt_dim,
            latent_dim=scpm_latent_dim,
            num_tokens=num_tokens,
            num_layers=num_layers,
        )
        self.fprd_time_logit = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))
        self.last_diffusion_time = torch.tensor(0.1)

    def _reset_prompt_state(self):
        if getattr(self, 'scpm_enabled', True):
            self.scpm.reset_state()

    def _generate_layer_prompt_from_tokens(self, tokens, layer_idx, patch_start):
        if layer_idx == 0:
            base = self.prompt_embeddings[0]
        else:
            base = self.deep_prompt_embeddings[layer_idx - 1]
        if not getattr(self, 'scpm_enabled', True):
            return base
        residual = self.scpm(tokens, layer_idx, patch_start)
        return base + residual

    def get_scpm_regularization(self):
        if not getattr(self, 'scpm_enabled', True):
            return self.prompt_embeddings.new_tensor(0.0)
        if hasattr(self, 'scpm'):
            return self.scpm.regularization_loss()
        return self.prompt_embeddings.new_tensor(0.0)

    def compute_diffusion_time(self):
        return self.fprd_time_max * torch.sigmoid(self.fprd_time_logit)

    def apply_fidelity_preserving_residual_diffusion(self, features):
        from .fprd import apply_fprd
        return apply_fprd(self, features)
