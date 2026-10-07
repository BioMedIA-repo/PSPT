import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli


class PrototypeCalibratedPatchSampling(nn.Module):


    def __init__(
        self, feature_dim=1024, prototype_count=16, assignment_temperature=0.1,
        probability_epsilon=1e-6, initial_reputation=0.3,
        concrete_temperature=0.1, mask_samples=10,
        reputation_contrast=1.0,
        prototype_calibration_max=0.5,
        prototype_calibration_init=0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.prototype_count = prototype_count
        self.assignment_temperature = assignment_temperature
        self.probability_epsilon = probability_epsilon
        self.concrete_temperature = concrete_temperature
        self.mask_samples = mask_samples
        self.reputation_contrast = float(reputation_contrast)
        self.prototype_calibration_max = float(prototype_calibration_max)
        self.initial_reputation = initial_reputation
        self.prototype_codebook = nn.Parameter(torch.zeros(prototype_count, feature_dim))
        self.register_buffer("initial_prototype_codebook", torch.zeros(prototype_count, feature_dim))
        initial_reputation = float(min(max(initial_reputation, probability_epsilon), 1.0 - probability_epsilon))
        init_logit = math.log(initial_reputation / (1.0 - initial_reputation))
        self.prototype_reputation_logits = nn.Parameter(torch.full((prototype_count, 1), init_logit))
        correction_ratio = float(prototype_calibration_init) / max(
            self.prototype_calibration_max, probability_epsilon
        )
        correction_ratio = min(max(correction_ratio, probability_epsilon), 1.0 - probability_epsilon)
        self.prototype_calibration_logit = nn.Parameter(torch.tensor(
            math.log(correction_ratio / (1.0 - correction_ratio)), dtype=torch.float32
        ))
        self._initialized = False

    def compute_prototype_reputation(self):

        logits = self.prototype_reputation_logits.squeeze(1)
        centered_logits = logits - logits.mean()
        scale = centered_logits.std(unbiased=False).clamp_min(1e-4)
        normalized_logits = centered_logits / scale
        center_logit = math.log(
            self.initial_reputation / (1.0 - self.initial_reputation)
        )
        reputation = torch.sigmoid(center_logit + self.reputation_contrast * normalized_logits)
        return reputation.unsqueeze(1)

    @torch.no_grad()
    def initialize_adaptive_codebook(self, features_list, sample_ratio=0.05):

        if self._initialized:
            return
        sampled_list = []
        for feats in features_list:
            feats = feats.detach().cpu().float()
            n_sample = max(1, int(feats.shape[0] * sample_ratio))
            sampled_list.append(feats[torch.randperm(feats.shape[0])[:n_sample]])
        all_feats = torch.cat(sampled_list, dim=0)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.prototype_count, random_state=42, n_init=10)
        kmeans.fit(all_feats.numpy())
        centers = torch.as_tensor(
            kmeans.cluster_centers_, dtype=self.prototype_codebook.dtype,
            device=self.prototype_codebook.device,
        )
        self.prototype_codebook.copy_(centers)
        self.initial_prototype_codebook.copy_(centers)
        self._initialized = True
        total_before = sum(f.shape[0] for f in features_list)
        print(
            f"Codebook init: {all_feats.shape[0]}/{total_before} patches "
            f"({100 * sample_ratio:.1f}%) -> KMeans(C={self.prototype_count}); "
            "prototypes are trainable after initialization"
        )

    def compute_prototype_assignment(self, features):

        features_norm = F.normalize(features.float(), dim=1)
        centers_norm = F.normalize(self.prototype_codebook.float(), dim=1)
        return F.softmax((features_norm @ centers_norm.T) / self.assignment_temperature, dim=1)

    def compute_prototype_prior(self, assign):

        prior = assign @ self.compute_prototype_reputation()
        return prior.clamp(self.probability_epsilon, 1.0 - self.probability_epsilon)

    def compute_prototype_calibration_strength(self):

        return self.prototype_calibration_max * torch.sigmoid(
            self.prototype_calibration_logit
        )

    def calibrate_probability_mean(self, probabilities, target_keep_ratio):

        eps = self.probability_epsilon
        logits = torch.logit(probabilities.clamp(eps, 1.0 - eps), eps=eps)
        target = float(min(max(target_keep_ratio, eps), 1.0 - eps))
        with torch.no_grad():
            fixed_logits = logits.detach()
            lower = fixed_logits.new_tensor(-30.0)
            upper = fixed_logits.new_tensor(30.0)
            for _ in range(40):
                midpoint = (lower + upper) * 0.5
                if torch.sigmoid(fixed_logits + midpoint).mean().item() < target:
                    lower = midpoint
                else:
                    upper = midpoint
            shift = (lower + upper) * 0.5
        return torch.sigmoid(logits + shift).clamp(eps, 1.0 - eps)

    def compute_task_relevance_posterior(self, task_logits, prior, target_keep_ratio):

        task_logits = task_logits.float().reshape(-1, 1)
        prior_logits = torch.logit(
            prior.float().clamp(self.probability_epsilon, 1.0 - self.probability_epsilon),
            eps=self.probability_epsilon,
        )
        task_evidence = (
            task_logits - task_logits.mean(dim=0, keepdim=True)
        ) / task_logits.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        prototype_evidence = (
            prior_logits - prior_logits.mean(dim=0, keepdim=True)
        ) / prior_logits.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        mixing = self.compute_prototype_calibration_strength()
        posterior_score = (1.0 - mixing) * task_evidence + mixing * prototype_evidence
        posterior = self.calibrate_probability_mean(
            torch.sigmoid(posterior_score), target_keep_ratio
        )
        return posterior.clamp(self.probability_epsilon, 1.0 - self.probability_epsilon)

    def sample_variational_keep_mask(self, post_prob):

        logits = torch.logit(post_prob, eps=self.probability_epsilon)
        samples = RelaxedBernoulli(
            self.concrete_temperature, logits=logits
        ).rsample((self.mask_samples,))
        return samples.mean(dim=0)

    @staticmethod
    def bernoulli_kl(q, p, eps=1e-6):
        q = q.clamp(eps, 1.0 - eps)
        p = p.clamp(eps, 1.0 - eps)
        return (
            q * torch.log(q / p)
            + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))
        ).mean()

    def prototype_calibration_divergence(self, post, prior):

        return self.bernoulli_kl(post, prior.detach(), self.probability_epsilon)

    def prototype_reputation_loss(self, task_prob, assign):


        assign_fixed = assign.detach()
        task_fixed = task_prob.detach()
        cluster_mass = assign_fixed.sum(dim=0).clamp_min(self.probability_epsilon)
        cluster_task = (assign_fixed.T @ task_fixed).squeeze(1) / cluster_mass
        rho_prob = self.compute_prototype_reputation().squeeze(1)
        loss = self.bernoulli_kl(cluster_task, rho_prob, self.probability_epsilon)
        return loss, cluster_task

    def selection_budget_loss(self, post, target_keep_ratio):
        target = post.new_tensor(float(target_keep_ratio)).clamp(0.0, 1.0)
        return (post.mean() - target).square()

    def prototype_reputation_separation_loss(self):

        rho_prob = self.compute_prototype_reputation().squeeze(1)
        return -rho_prob.var(unbiased=False)

    def prototype_reputation_bimodal_loss(self):

        rho_prob = self.compute_prototype_reputation().squeeze(1)
        return (rho_prob * (1.0 - rho_prob)).mean()

    def codebook_compactness_loss(self, features, assign):

        features_norm = F.normalize(features.float(), dim=1)
        centers_norm = F.normalize(self.prototype_codebook.float(), dim=1)
        reconstruction = F.normalize(assign @ centers_norm, dim=1)
        return (1.0 - (features_norm * reconstruction).sum(dim=1)).mean()

    def codebook_anchor_loss(self):

        centers = F.normalize(self.prototype_codebook.float(), dim=1)
        centers_init = F.normalize(self.initial_prototype_codebook.float(), dim=1)
        return (1.0 - (centers * centers_init).sum(dim=1)).mean()

    @staticmethod
    def compute_slide_prototype_profile(assign):

        return assign.mean(dim=0)

    def forward(self, features, task_logits, target_keep_ratio=None):
        assign = self.compute_prototype_assignment(features)
        raw_prior = self.compute_prototype_prior(assign)
        if target_keep_ratio is None:
            target_keep_ratio = min(1.0, 384.0 / max(1, features.shape[0]))
        prior = self.calibrate_probability_mean(raw_prior, target_keep_ratio)
        task_prob = self.calibrate_probability_mean(
            torch.sigmoid(task_logits.float().reshape(-1, 1)), target_keep_ratio
        )
        post = self.compute_task_relevance_posterior(
            task_logits, prior, target_keep_ratio
        )
        mask = self.sample_variational_keep_mask(post) if self.training else post
        consistency = self.prototype_calibration_divergence(post, prior)
        rho_loss, cluster_task = self.prototype_reputation_loss(task_prob, assign)
        return {
            "task_relevance_posterior": post,
            "prototype_prior": prior,
            "variational_keep_mask": mask,
            "prototype_assignment": assign,
            "prototype_calibration_divergence": consistency,
            "prototype_reputation_loss": rho_loss,
            "selection_budget_loss": self.selection_budget_loss(post, target_keep_ratio),
            "codebook_compactness_loss": self.codebook_compactness_loss(features, assign),
            "codebook_anchor_loss": self.codebook_anchor_loss(),
            "prototype_task_relevance": cluster_task,
            "task_evidence_probability": task_prob,
            "prototype_calibration_strength": self.compute_prototype_calibration_strength(),
            "slide_prototype_profile": self.compute_slide_prototype_profile(assign),
        }

    def select_task_relevant_patches(self, features, task_logits, selected_count=384):
        with torch.no_grad():
            keep_ratio = min(1.0, selected_count / max(1, features.shape[0]))
            assign = self.compute_prototype_assignment(features)
            prior = self.calibrate_probability_mean(
                self.compute_prototype_prior(assign), keep_ratio
            )
            scores = self.compute_task_relevance_posterior(
                task_logits, prior, keep_ratio
            ).squeeze(1)
            k = min(selected_count, scores.shape[0])
            selected_scores, selected_indices = torch.topk(scores, k)
        return selected_indices, selected_scores
