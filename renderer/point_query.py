"""Point-query rendering operator R(f, z): evaluate the 3D Gaussian field at points.

Implements eq. (6) of the proposal directly. For a query point p, the predicted
gene-expression vector is the opacity- and Mahalanobis-weighted average of the k
nearest Gaussians:

    f(p) = sum_i w_i g_i / (sum_i w_i + eps),
    w_i  = alpha_i * exp(-0.5 * (p - mu_i)^T Sigma_i^{-1} (p - mu_i)).

Pure PyTorch, no CUDA. kNN selection is non-differentiable (standard); the weights
and the weighted sum are fully differentiable w.r.t. means, scale, rotation,
opacity and features.
"""
import torch


def render(gaussians, query_points, k=16, chunk=8192, eps=1e-8, return_aux=True):
    means = gaussians.xyz                       # (N,3)
    feats = gaussians.features                  # (N,G)
    opacity = gaussians.opacity                 # (N,1)
    _, Sigma_inv = gaussians.build_covariance()  # (N,3,3)

    N = means.shape[0]
    M = query_points.shape[0]
    G = feats.shape[1]
    k = min(k, N)

    preds = []
    weights_all = []
    idx_all = []
    vis_count = torch.zeros(N, device=means.device)
    max_weight = torch.zeros(N, device=means.device)

    for start in range(0, M, chunk):
        q = query_points[start:start + chunk]               # (m,3)
        # Euclidean kNN selection
        with torch.no_grad():
            d2 = torch.cdist(q, means)                      # (m,N)
            idx = d2.topk(k, dim=1, largest=False).indices  # (m,k)

        mu_k = means[idx]                                   # (m,k,3)
        Sinv_k = Sigma_inv[idx]                             # (m,k,3,3)
        feat_k = feats[idx]                                 # (m,k,G)
        alpha_k = opacity[idx].squeeze(-1)                  # (m,k)

        delta = q[:, None, :] - mu_k                        # (m,k,3)
        maha = torch.einsum("mki,mkij,mkj->mk", delta, Sinv_k, delta)  # (m,k)
        w = alpha_k * torch.exp(-0.5 * maha)                # (m,k)

        wsum = w.sum(dim=1, keepdim=True)                   # (m,1)
        pred = (w[..., None] * feat_k).sum(dim=1) / (wsum + eps)  # (m,G)
        preds.append(pred)

        if return_aux:
            weights_all.append(w)
            idx_all.append(idx)
            flat_idx = idx.reshape(-1)
            vis_count.scatter_add_(
                0, flat_idx, torch.ones_like(flat_idx, dtype=vis_count.dtype))
            max_weight.scatter_reduce_(
                0, flat_idx, w.reshape(-1), reduce="amax", include_self=True)

    out = {"gex": torch.cat(preds, dim=0)}                  # (M,G)
    if return_aux:
        out["weights"] = torch.cat(weights_all, dim=0)      # (M,k)
        out["idx"] = torch.cat(idx_all, dim=0)              # (M,k)
        out["vis_count"] = vis_count                        # (N,)
        out["max_weight"] = max_weight                      # (N,)
    return out
