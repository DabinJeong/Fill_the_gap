"""Metrics for held-out gene-expression prediction."""
import numpy as np


def _pearson(a, b, axis):
    a = a - a.mean(axis=axis, keepdims=True)
    b = b - b.mean(axis=axis, keepdims=True)
    num = (a * b).sum(axis=axis)
    den = np.sqrt((a ** 2).sum(axis=axis) * (b ** 2).sum(axis=axis))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return r


def per_gene_pearson(pred, target):
    """Correlation across spots, per gene -> (G,). NaN where a gene is constant."""
    return _pearson(pred, target, axis=0)


def spot_pearson(pred, target):
    """Correlation across genes, per spot -> (N,)."""
    return _pearson(pred, target, axis=1)


def mse_mae(pred, target):
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))}


def asign_pcc(pred, target, eps=1e-8):
    """ASIGN `calculate_pcc` (models/losses.py): per-spot Pearson across genes,
    averaged over spots. Correlation is taken over dim=1 (the 250 genes) for each
    row/spot, then meaned. pred/target: (N_spots, G)."""
    x = pred - pred.mean(axis=1, keepdims=True)
    y = target - target.mean(axis=1, keepdims=True)
    cov = (x * y).sum(axis=1)
    vx = (x ** 2).sum(axis=1)
    vy = (y ** 2).sum(axis=1)
    pcc = cov / np.sqrt(vx * vy + eps)
    return float(pcc.mean())


def asign_metrics(pred, target):
    """The exact ASIGN benchmark metrics: PCC (per-spot), MSE, MAE."""
    mm = mse_mae(pred, target)
    return {"pcc": asign_pcc(pred, target), "mse": mm["mse"], "mae": mm["mae"]}


def hvg_subset_metrics(pred, target, hvg_idx):
    pg = per_gene_pearson(pred[:, hvg_idx], target[:, hvg_idx])
    return {
        "hvg_pearson_median": float(np.nanmedian(pg)),
        "hvg_pearson_mean": float(np.nanmean(pg)),
        **{f"hvg_{k}": v for k, v in mse_mae(pred[:, hvg_idx], target[:, hvg_idx]).items()},
    }


def summarize(pred, target):
    pg = per_gene_pearson(pred, target)
    sp = spot_pearson(pred, target)
    out = {
        "pcc": asign_pcc(pred, target),                 # ASIGN headline metric
        "per_gene_pearson_median": float(np.nanmedian(pg)),
        "per_gene_pearson_mean": float(np.nanmean(pg)),
        "spot_pearson_mean": float(np.nanmean(sp)),
        "spot_pearson_median": float(np.nanmedian(sp)),
    }
    out.update(mse_mae(pred, target))
    return out


def aggregate(per_sample_results):
    """Mean/median across samples of each scalar metric."""
    keys = per_sample_results[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in per_sample_results], dtype=float)
        agg[f"{k}_mean"] = float(np.nanmean(vals))
        agg[f"{k}_median"] = float(np.nanmedian(vals))
    return agg
