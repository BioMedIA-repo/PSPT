"""SCPM and FPRD inference components from PSPT."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlideAwareCrossLayerPromptModulation(nn.Module):
    """Maintain a WSI-anchored latent state across Transformer depth."""

    def __init__(
        self,
        prompt_dim,
        latent_dim=128,
        num_tokens=1,
        num_layers=24,
        eta_max=0.5,
        prompt_residual_max=0.2,
        transition_rho=0.995,
    ):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.eta_max = eta_max
        self.prompt_residual_max = prompt_residual_max
        self.transition_rho = transition_rho

        self.context_norm = nn.LayerNorm(prompt_dim)
        self.context_projector = nn.Linear(prompt_dim, latent_dim)
        self.transition_delta = nn.Parameter(torch.empty(latent_dim, latent_dim))
        self.correction_logits = nn.Parameter(
            torch.zeros(max(1, num_layers - 1))
        )
        self.state_norm = nn.LayerNorm(latent_dim)
        self.residual_decoder = nn.Linear(
            latent_dim, num_tokens * prompt_dim, bias=False
        )
        residual_ratio = 0.05 / prompt_residual_max
        self.residual_logits = nn.Parameter(
            torch.full(
                (num_layers,),
                math.log(residual_ratio / (1.0 - residual_ratio)),
            )
        )

        nn.init.normal_(self.transition_delta, std=0.02)
        nn.init.normal_(self.residual_decoder.weight, std=1e-3)
        self._wsi_initial_context = None
        self.reset_state()

    def set_wsi_initial_context(self, context):
        self._wsi_initial_context = context.detach()

    def reset_state(self):
        self._state = None
        self._transition_fp32 = None
        self._transition_generator_fp32 = None

    def _transition(self, dtype, device):
        if self._transition_fp32 is None:
            generator = self.transition_delta.float()
            generator = 0.25 * (generator - generator.T)
            self._transition_generator_fp32 = generator
            self._transition_fp32 = (
                self.transition_rho * torch.matrix_exp(generator)
            )
        return self._transition_fp32.to(device=device, dtype=dtype)

    def forward(self, tokens, layer_index, patch_start):
        patch_tokens = tokens[:, patch_start:, :]
        if layer_index == 0 and self._wsi_initial_context is not None:
            context = self._wsi_initial_context.float()
        else:
            context = patch_tokens.detach().float().mean(
                dim=(0, 1), keepdim=True
            )
        context = context.to(device=tokens.device, dtype=tokens.dtype)
        observation = self.context_projector(self.context_norm(context))

        if layer_index == 0 or self._state is None:
            state = observation
        else:
            predicted = self._state @ self._transition(
                dtype=tokens.dtype, device=tokens.device
            )
            eta = self.eta_max * torch.sigmoid(
                self.correction_logits[layer_index - 1]
            )
            state = (1.0 - eta) * predicted + eta * observation
        self._state = state

        residual = self.residual_decoder(self.state_norm(state)).view(
            1, self.num_tokens, self.prompt_dim
        )
        strength = self.prompt_residual_max * torch.sigmoid(
            self.residual_logits[layer_index]
        )
        return (strength * residual).squeeze(0).to(dtype=tokens.dtype)


class PSPTCoreMixin:
    """Shared prompt, SCPM, and FPRD inference implementation."""

    def _init_pspt_core(
        self,
        prompt_dim,
        num_layers,
        num_tokens=1,
        prompt_dropout=0.0,
        scpm_latent_dim=128,
        fprd_enabled=True,
        fprd_neighbors=16,
        fprd_temperature=0.2,
        fprd_time_max=0.5,
        fprd_iterations=3,
    ):
        self.num_prompt_tokens = num_tokens
        self.prompt_dim = prompt_dim
        self.num_layers = num_layers
        self.scpm_enabled = True
        self.fprd_enabled = fprd_enabled
        self.fprd_neighbors = fprd_neighbors
        self.fprd_temperature = fprd_temperature
        self.fprd_time_max = fprd_time_max
        self.fprd_iterations = fprd_iterations

        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, num_tokens, prompt_dim)
        )
        self.deep_prompt_embeddings = nn.Parameter(
            torch.zeros(num_layers - 1, num_tokens, prompt_dim)
        )
        nn.init.trunc_normal_(self.prompt_embeddings, std=0.02)
        nn.init.trunc_normal_(self.deep_prompt_embeddings, std=0.02)
        self.prompt_dropout = nn.Dropout(prompt_dropout)

        self.scpm = SlideAwareCrossLayerPromptModulation(
            prompt_dim=prompt_dim,
            latent_dim=scpm_latent_dim,
            num_tokens=num_tokens,
            num_layers=num_layers,
        )
        self.fprd_time_logit = nn.Parameter(
            torch.tensor(math.log(0.2 / 0.8))
        )
        self.last_diffusion_time = torch.tensor(0.1)

    def compute_diffusion_time(self):
        return self.fprd_time_max * torch.sigmoid(self.fprd_time_logit)

    def apply_fidelity_preserving_residual_diffusion(self, features):
        if not self.fprd_enabled or features.shape[0] <= 1:
            return features

        node_count = features.shape[0]
        neighbors = min(self.fprd_neighbors, node_count - 1)
        with torch.no_grad():
            normalized = F.normalize(features.detach().float(), dim=1)
            similarity = normalized @ normalized.T
            similarity.fill_diagonal_(-torch.inf)
            values, indices = torch.topk(
                similarity, k=neighbors, dim=1
            )
            weights = F.softmax(
                values / self.fprd_temperature, dim=1
            )

        diffusion_time = self.compute_diffusion_time().to(
            device=features.device, dtype=torch.float32
        )
        beta = diffusion_time / (1.0 + diffusion_time)
        source = features.float()
        propagated = source
        for _ in range(max(1, self.fprd_iterations)):
            gathered = propagated[indices]
            neighborhood = (
                gathered * weights.unsqueeze(-1)
            ).sum(dim=1)
            propagated = (
                (1.0 - beta) * source + beta * neighborhood
            )
        self.last_diffusion_time = diffusion_time.detach()
        return propagated.to(dtype=features.dtype)

