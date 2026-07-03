"""Gene-expression rendering loss: MSE + optional cosine / sparsity regularizers."""
import torch.nn.functional as F


def gex_loss(pred, target, gaussians=None, cfg=None):
    """Return (total_loss, logs).

    pred, target: (M, G). Targets are signed log-normalized expression, so MSE is
    the right primary objective (no non-negativity assumed).
    """
    w = (cfg["loss"] if (cfg is not None and "loss" in cfg) else (cfg or {}))
    w_cos = float(w.get("w_cos", 0.0))
    w_opacity = float(w.get("w_opacity", 0.0))
    w_scale = float(w.get("w_scale", 0.0))

    mse = F.mse_loss(pred, target)
    total = mse
    logs = {"mse": mse.item()}

    if w_cos > 0:
        cos = 1.0 - F.cosine_similarity(pred, target, dim=1).mean()
        total = total + w_cos * cos
        logs["cos"] = cos.item()

    if gaussians is not None and w_opacity > 0:
        opacity_reg = gaussians.opacity.mean()
        total = total + w_opacity * opacity_reg
        logs["opacity_reg"] = opacity_reg.item()

    if gaussians is not None and w_scale > 0:
        scale_reg = gaussians.scaling.mean()
        total = total + w_scale * scale_reg
        logs["scale_reg"] = scale_reg.item()

    logs["total"] = total.item()
    return total, logs
