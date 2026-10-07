import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SlideAwareCrossLayerPromptModulation(nn.Module):


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
        self.correction_logits = nn.Parameter(torch.zeros(max(1, num_layers - 1)))
        self.state_norm = nn.LayerNorm(latent_dim)
        self.residual_decoder = nn.Linear(latent_dim, num_tokens * prompt_dim, bias=False)
        residual_ratio = 0.05 / prompt_residual_max
        self.residual_logits = nn.Parameter(
            torch.full((num_layers,), math.log(residual_ratio / (1.0 - residual_ratio)))
        )

        nn.init.normal_(self.transition_delta, std=0.02)
        nn.init.normal_(self.residual_decoder.weight, std=1e-3)
        self._wsi_initial_context = None
        self.reset_state()

    def set_wsi_initial_context(self, context):

        self._wsi_initial_context = context.detach()

    def clear_wsi_initial_context(self):
        self._wsi_initial_context = None

    def reset_state(self):
        self._state = None
        self._states = []
        self._residuals = []
        self._context_norms = []
        self._corrections = []


        self._transition_fp32 = None
        self._transition_generator_fp32 = None

    def _transition(self, dtype, device):
        if self._transition_fp32 is None:
            generator = self.transition_delta.float()
            generator = 0.25 * (generator - generator.T)
            self._transition_generator_fp32 = generator
            self._transition_fp32 = self.transition_rho * torch.matrix_exp(generator)
        return (
            self._transition_fp32.to(device=device, dtype=dtype),
            self._transition_generator_fp32,
        )

    def forward(self, tokens, layer_idx, patch_start):
        patch_tokens = tokens[:, patch_start:, :]
        if layer_idx == 0 and self._wsi_initial_context is not None:
            context = self._wsi_initial_context.float()
        else:
            context = patch_tokens.detach().float().mean(dim=(0, 1), keepdim=True)
        context = context.to(device=tokens.device, dtype=tokens.dtype)
        observed = self.context_projector(self.context_norm(context))

        if layer_idx == 0 or self._state is None:
            state = observed
        else:
            transition, _ = self._transition(dtype=tokens.dtype, device=tokens.device)
            predicted = self._state @ transition
            eta = self.eta_max * torch.sigmoid(self.correction_logits[layer_idx - 1])
            state = (1.0 - eta) * predicted + eta * observed
            self._corrections.append(eta.detach().float())
        self._state = state

        residual = self.residual_decoder(self.state_norm(state)).view(
            1, self.num_tokens, self.prompt_dim
        )
        strength = self.prompt_residual_max * torch.sigmoid(self.residual_logits[layer_idx])
        residual = strength.to(dtype=residual.dtype) * residual


        self._states.append(state)
        self._residuals.append(residual)
        self._context_norms.append(context.detach().float().norm())
        return residual.squeeze(0).to(dtype=tokens.dtype)

    def finalize_diagnostics(self):
        if self._residuals:
            residuals = torch.stack(
                [residual.detach().float() for residual in self._residuals], dim=0
            ).flatten(start_dim=1)
            self.last_residual_vectors = residuals.unsqueeze(0)
            self.last_residual_norms = residuals.norm(dim=1).unsqueeze(0)
            if residuals.shape[0] > 1:
                self.last_adjacent_residual_cosine = F.cosine_similarity(
                    residuals[1:], residuals[:-1], dim=1
                ).unsqueeze(0)
            else:
                self.last_adjacent_residual_cosine = residuals.new_zeros(1, 0)
        if self._states:
            states = torch.cat(
                [state.detach().float() for state in self._states], dim=0
            ).flatten(start_dim=1)
            self.last_state_vectors = states.unsqueeze(0)
            if states.shape[0] > 1:
                self.last_state_smoothness = (
                    1.0 - F.cosine_similarity(states[1:], states[:-1], dim=1)
                ).mean().detach()
            else:
                self.last_state_smoothness = states.new_tensor(0.0)
        if self._context_norms:
            self.last_context_norm_mean = torch.stack(self._context_norms).mean().detach()
        if self._corrections:
            self.last_correction_mean = torch.stack(self._corrections).mean().detach()
        else:
            self.last_correction_mean = self.transition_delta.new_tensor(float('nan'))
        transition, _ = self._transition(
            dtype=self.transition_delta.dtype,
            device=self.transition_delta.device,
        )
        identity = torch.eye(self.latent_dim, device=transition.device, dtype=torch.float32)
        self.last_transition_orthogonality_error = (
            (transition.float().T @ transition.float()) - (self.transition_rho ** 2) * identity
        ).norm().detach() / self.latent_dim

    def regularization_loss(self):
        loss = self.transition_delta.new_tensor(0.0)
        if self._residuals:
            residuals = torch.stack(self._residuals, dim=0)
            loss = loss + 1e-4 * residuals.float().square().mean()
        if self._states and len(self._states) > 1:
            states = torch.cat(self._states, dim=0).flatten(start_dim=1)
            flow = 1.0 - F.cosine_similarity(states[1:], states[:-1], dim=1)
            loss = loss + 1e-2 * flow.mean()
        return loss

SCPM = SlideAwareCrossLayerPromptModulation
