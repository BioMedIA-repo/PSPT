import torch
import torch.nn.functional as F


def apply_fprd(backbone, features):


    if not backbone.fprd_enabled or features.shape[0] <= 1:
        return features
    k = min(backbone.fprd_neighbors, features.shape[0] - 1)
    with torch.no_grad():
        normalized = F.normalize(features.detach().float(), dim=1)
        similarity = normalized @ normalized.T
        similarity.fill_diagonal_(-torch.inf)
        values, indices = torch.topk(similarity, k=k, dim=1)
        weights = F.softmax(values / backbone.fprd_temperature, dim=1)
    diffusion_time = backbone.compute_diffusion_time().to(device=features.device, dtype=torch.float32)
    beta = diffusion_time / (1.0 + diffusion_time)
    source = features.float()
    z = source
    for _ in range(max(1, backbone.fprd_iterations)):
        gathered = z[indices]
        smooth = (gathered * weights.unsqueeze(-1)).sum(dim=1)
        z = (1.0 - beta) * source + beta * smooth
    backbone.last_diffusion_time = diffusion_time.detach()
    backbone.last_diffusion_delta = (z - source).detach().norm(dim=1)
    backbone.last_fprd_beta = beta.detach()
    backbone.last_fprd_neighbor_similarity = values.detach().float().mean()
    return z.to(dtype=features.dtype)
