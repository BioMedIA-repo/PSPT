"""PCPS score loading and deterministic evaluation-time sampling."""

from pathlib import Path

import torch


class PCPSArtifacts:
    def __init__(self, scores_pt):
        self.scores_pt = Path(scores_pt)
        self.scores = torch.load(
            self.scores_pt,
            map_location="cpu",
            weights_only=True,
        )

    def sample(
        self,
        wsi_id,
        patch_count,
        kappa,
        sampling_seed,
        split_offset,
        item_index,
    ):
        if wsi_id not in self.scores:
            raise KeyError(f"Missing all-patch PCPS scores for {wsi_id}.")
        item = self.scores[wsi_id]
        if isinstance(item, dict):
            item = item["pcps_selection_score"]
        scores = item.float().reshape(-1)
        total_patches = scores.numel()
        count = min(int(patch_count), total_patches)
        if count == total_patches:
            return list(range(total_patches))

        eps = 1e-6
        evidence = torch.logit(
            scores.clamp(eps, 1.0 - eps), eps=eps
        )
        evidence = (
            evidence - evidence.mean()
        ) / evidence.std(unbiased=False).clamp_min(eps)
        generator = torch.Generator(device="cpu").manual_seed(
            int(sampling_seed)
            + int(split_offset)
            + int(item_index) * 104729
        )
        uniform = torch.rand(
            evidence.shape, generator=generator
        ).clamp_(eps, 1.0 - eps)
        gumbel = -torch.log(-torch.log(uniform))
        keys = float(kappa) * evidence + gumbel
        return sorted(torch.topk(keys, count).indices.tolist())
