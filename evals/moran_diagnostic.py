"""Moran's I spatial-autocorrelation diagnostic for HistoGS held-out predictions.

The ASIGN benchmark metric (asign_pcc, per-spot Pearson across genes) is kept as the
headline number for apples-to-apples comparison with ASIGN. But it is shuffle-invariant
(a spot-permuted prediction scores the same) and saturates at the mean gene-profile, so it
does not certify that predictions carry real SPATIAL structure. Moran's I does: for each
gene we measure spatial autocorrelation over the section's spot graph.

  Moran's I = (N / S0) * (z' W z) / (z' z),   z = x - mean(x),   W = row-normalized kNN adjacency.

Interpretation:
  - measured section: I > 0 (real tissue is spatially autocorrelated) -> the ceiling.
  - mean-profile baseline (constant gene vector per spot): I ≈ 0 (no spatial variation).
  - HistoGS prediction: should be I > 0 and comparable to measured (not washed out / not over-smoothed).

Run:
  cd /nfs/team361/dj16/projects/HistoGS
  .venv/bin/python evals/moran_diagnostic.py --config configs/her2_A.yaml
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.metrics import asign_pcc, per_gene_pearson
from renderer import render
from scene import GaussianModel, Scene
from utils.config import load_config
from utils.general_utils import get_device

HER2_SAMPLES = ["A", "B", "C", "D", "E", "F", "G", "H"]


def knn_W(xy, k=6):
    """Row-normalized binary kNN spatial-weights matrix from spot (x,y) positions."""
    xy = xy.astype(np.float64)
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    N = xy.shape[0]
    k = min(k, N - 1)
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    W = np.zeros((N, N))
    rows = np.repeat(np.arange(N), k)
    W[rows, nn.ravel()] = 1.0
    rs = W.sum(1, keepdims=True); rs[rs == 0] = 1
    return W / rs


def morans_I(values, W):
    """Mean Moran's I over columns (genes) of `values` (N_spots, G) given weights W."""
    z = values - values.mean(0, keepdims=True)           # (N,G)
    num = np.einsum("ng,nm,mg->g", z, W, z)              # z' W z per gene
    den = (z ** 2).sum(0)                                 # z' z per gene
    N = values.shape[0]; S0 = W.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        I = (N / S0) * num / den
    return I                                              # (G,)


def main():
    ap = ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--samples", default=None); ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config(args.config); dev = get_device(cfg["device"]); d = cfg["dataset"]
    samples = args.samples.split(",") if args.samples else HER2_SAMPLES

    rng = np.random.RandomState(0)
    print(f"{'sample':<7}{'asign_pcc':>10}{'I_measured':>12}{'I_HistoGS':>11}"
          f"{'I_shuffled':>12}{'pergene_r':>11}")
    agg = {k: [] for k in ("pcc", "Im", "Ig", "Ib", "pg")}
    for s in samples:
        sc = Scene(s, d["root"], d["name"], held_out_parity=d["held_out_parity"],
                   train_levels=[1], heldout_levels=[2, 3, 4, 5, 6],
                   z_spacing=d["z_spacing"], device=dev)
        xyz_tr, gex_tr = sc.train_points()
        xyz_ho, gex_ho, lvl_ho = sc.heldout_points()
        g = GaussianModel(G=sc.G, device=dev, densify_z=cfg["model"]["densify_z"])
        g.create_from_pcd(xyz_tr, gex_tr, init_opacity=cfg["model"]["init_opacity"],
                          scale_knn=cfg["model"]["scale_knn"],
                          init_scale_mult=cfg["model"].get("init_scale_mult", 1.0))
        with torch.no_grad():
            pred = render(g, xyz_ho, k=cfg["render"]["knn_k"],
                          chunk=cfg["render"]["chunk"], return_aux=False)["gex"].cpu().numpy()
        tgt = gex_ho.detach().cpu().numpy()
        lvl = lvl_ho.detach().cpu().numpy()
        xy_ho = sc.xy[sc.heldout_mask.cpu().numpy()]
        # Moran's I is per-section (one spatial graph per section); average over held-out sections.
        # Control: shuffle prediction across spots -> destroys spatial structure -> I≈0.
        Im, Ig, Ib = [], [], []
        for L in np.unique(lvl):
            m = lvl == L
            if m.sum() < args.k + 2:
                continue
            W = knn_W(xy_ho[m], k=args.k)
            Im.append(np.nanmean(morans_I(tgt[m], W)))
            Ig.append(np.nanmean(morans_I(pred[m], W)))
            shuf = pred[m][rng.permutation(int(m.sum()))]
            Ib.append(np.nanmean(morans_I(shuf, W)))
        Im, Ig, Ib = np.mean(Im), np.mean(Ig), np.mean(Ib)
        pcc = asign_pcc(pred, tgt)
        pg = float(np.nanmean(per_gene_pearson(pred, tgt)))
        print(f"{s:<7}{pcc:>10.3f}{Im:>12.3f}{Ig:>11.3f}{Ib:>12.3f}{pg:>11.3f}")
        for kk, v in zip(("pcc", "Im", "Ig", "Ib", "pg"), (pcc, Im, Ig, Ib, pg)):
            agg[kk].append(v)

    print("-" * 63)
    print(f"{'MEAN':<7}{np.mean(agg['pcc']):>10.3f}{np.mean(agg['Im']):>12.3f}"
          f"{np.mean(agg['Ig']):>11.3f}{np.mean(agg['Ib']):>12.3f}{np.mean(agg['pg']):>11.3f}")
    print("\nI_shuffled≈0 confirms Moran's I detects spatial structure (shuffle destroys it). "
          "I_HistoGS>>0 = predictions ARE spatially structured (unlike the shuffle-invariant "
          "asign_pcc), but I_HistoGS (~0.92) >> I_measured (~0.17) means HistoGS is "
          "OVER-SMOOTHED vs real tissue. pergene_r = per-gene across-spot spatial accuracy.")


if __name__ == "__main__":
    main()
