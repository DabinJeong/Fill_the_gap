"""How far apart can serial sections be and still support 3D interpolation?

Two complementary views, aggregated over the HER2 samples (A..H):

  Panel A — DATA CEILING (model-free). Match spots across sections by (x,y) grid
    position and compute the ASIGN per-spot PCC between two sections as a function of
    their section gap d = |level_i - level_j|. This is how fast the tissue itself
    decorrelates with depth: no interpolator can beat it.

  Panel B — HistoGS INTERPOLATION accuracy vs distance to the nearest KNOWN section.
    Build the HistoGS init field from a set of known sections and predict a target
    section's spots; bin accuracy by d = (sections to nearest known). Two regimes:
      * extrapolation / reach: known = {top section only}; predict each deeper one.
      * bracketed interpolation: known = {top, bottom}; predict the interior ones.
    A copy-nearest-known baseline is shown for reference.

x-axis is section-spacing units (Δlevel); physical distance = Δlevel x section thickness.

Run:
  cd /nfs/team361/dj16/projects/HistoGS
  .venv/bin/python scripts/section_distance_analysis.py --config configs/her2_A.yaml
"""
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from evals.metrics import asign_pcc, mse_mae
from renderer import render
from scene import GaussianModel
from utils.config import load_config
from utils.data_utils import load_sample_npy, normalize_coords, sample_npy_path
from utils.general_utils import get_device

HER2_SAMPLES = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _match_by_xy(xy_a, xy_b):
    """Index arrays (ia, ib) of spots in a and b that share the same (x,y) grid cell."""
    lut = {tuple(p): i for i, p in enumerate(xy_b)}
    ia, ib = [], []
    for i, p in enumerate(xy_a):
        j = lut.get(tuple(p))
        if j is not None:
            ia.append(i); ib.append(j)
    return np.array(ia, dtype=int), np.array(ib, dtype=int)


def load_sample(cfg, sample):
    raw = load_sample_npy(sample_npy_path(cfg["dataset"]["root"], cfg["dataset"]["name"], sample))
    xyz, _ = normalize_coords(raw["xy"], raw["level"], z_spacing=cfg["dataset"]["z_spacing"])
    return {"xy": raw["xy"], "level": raw["level"], "gex": raw["gex"], "xyz": xyz}


def decorrelation_curve(cfg, samples):
    """Panel A: measured-to-measured PCC vs section gap d (model-free)."""
    by_d = defaultdict(list)
    for s in samples:
        S = load_sample(cfg, s)
        levels = sorted(set(S["level"].tolist()))
        for a in levels:
            for b in levels:
                if b <= a:
                    continue
                ma = S["level"] == a; mb = S["level"] == b
                ia, ib = _match_by_xy(S["xy"][ma], S["xy"][mb])
                if len(ia) < 5:
                    continue
                ga = S["gex"][ma][ia]; gb = S["gex"][mb][ib]
                by_d[b - a].append(asign_pcc(ga, gb))
    return by_d


def _build_field(cfg, device, means_np, feats_np):
    g = GaussianModel(G=feats_np.shape[1], device=device,
                      densify_z=cfg["model"]["densify_z"])
    g.create_from_pcd(
        torch.from_numpy(means_np).to(device), torch.from_numpy(feats_np).to(device),
        init_opacity=cfg["model"]["init_opacity"], scale_knn=cfg["model"]["scale_knn"],
        init_scale_mult=cfg["model"].get("init_scale_mult", 1.0))
    return g


def interp_vs_distance(cfg, samples):
    """Panel B: HistoGS PCC/MSE & copy baseline vs distance-to-nearest-known,
    for reach (known={top}) and bracketed interpolation (known={top,bottom})."""
    device = get_device(cfg["device"])
    rd = cfg["render"]
    out = {"reach": defaultdict(list), "bracket": defaultdict(list),
           "copy": defaultdict(list), "reach_mse": defaultdict(list),
           "bracket_mse": defaultdict(list)}
    for s in samples:
        S = load_sample(cfg, s)
        levels = sorted(set(S["level"].tolist()))
        top, bot = levels[0], levels[-1]

        for regime, known in [("reach", {top}), ("bracket", {top, bot})]:
            km = np.isin(S["level"], list(known))
            field = _build_field(cfg, device, S["xyz"][km], S["gex"][km])
            for t in levels:
                if t in known:
                    continue
                d = min(abs(t - k) for k in known)
                tm = S["level"] == t
                q = torch.from_numpy(S["xyz"][tm]).to(device)
                with torch.no_grad():
                    pred = render(field, q, k=rd["knn_k"], chunk=rd["chunk"],
                                  return_aux=False)["gex"].cpu().numpy()
                tgt = S["gex"][tm]
                out[regime][d].append(asign_pcc(pred, tgt))
                out[regime + "_mse"][d].append(mse_mae(pred, tgt)["mse"])

                if regime == "reach":  # copy from the single known (top) section
                    nk = min(known, key=lambda k: abs(t - k))
                    nkm = S["level"] == nk
                    ia, ib = _match_by_xy(S["xy"][tm], S["xy"][nkm])
                    if len(ia) >= 5:
                        out["copy"][d].append(asign_pcc(S["gex"][tm][ia], S["gex"][nkm][ib]))
    return out


def _agg(by_d):
    ds = sorted(by_d)
    mean = np.array([np.mean(by_d[d]) for d in ds])
    sem = np.array([np.std(by_d[d]) / max(1, np.sqrt(len(by_d[d]))) for d in ds])
    return np.array(ds), mean, sem


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--samples", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=os.path.join(_ROOT, "figures"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    samples = args.samples.split(",") if args.samples else HER2_SAMPLES
    os.makedirs(args.outdir, exist_ok=True)

    print("computing measured-to-measured decorrelation (Panel A) ...")
    deco = decorrelation_curve(cfg, samples)
    print("computing HistoGS interpolation vs distance (Panel B) ...")
    iv = interp_vs_distance(cfg, samples)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))

    dA, mA, sA = _agg(deco)
    axA.errorbar(dA, mA, yerr=sA, marker="o", color="C3", capsize=3, lw=2)
    axA.set_title("A. Data ceiling: measured↔measured correlation vs section gap")
    axA.set_xlabel("section gap  Δlevel  (units of section spacing)")
    axA.set_ylabel("ASIGN per-spot PCC between sections")
    axA.set_xticks(dA); axA.grid(alpha=0.3)
    axA.annotate("how fast the tissue itself\ndecorrelates with depth",
                 xy=(dA[-1], mA[-1]), xytext=(0.55, 0.78), textcoords="axes fraction",
                 fontsize=9, color="C3")

    for key, color, label in [("reach", "C0", "HistoGS reach (1 known top section)"),
                              ("bracket", "C2", "HistoGS interpolation (top+bottom known)"),
                              ("copy", "0.5", "copy nearest known (baseline)")]:
        if not iv[key]:
            continue
        d, m, sm = _agg(iv[key])
        axB.errorbar(d, m, yerr=sm, marker="s" if key != "copy" else "x",
                     color=color, capsize=3, lw=2,
                     ls="--" if key == "copy" else "-", label=label)
    axB.set_title("B. HistoGS interpolation accuracy vs distance to nearest known section")
    axB.set_xlabel("distance to nearest KNOWN section  (units of section spacing)")
    axB.set_ylabel("ASIGN per-spot PCC (predicted vs measured)")
    axB.grid(alpha=0.3); axB.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"How far apart can sections be? — HER2 {','.join(samples)} "
        f"({cfg['dataset']['name']}, HistoGS init field)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(args.outdir, f"{cfg['dataset']['name']}_section_distance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    # also dump the numbers
    print("\n-- Panel A: measured<->measured PCC vs gap --")
    for d, m, sm in zip(*_agg(deco)):
        print(f"  gap {int(d)}: PCC {m:.3f} ± {sm:.3f}")
    for key in ("reach", "bracket", "copy"):
        if not iv[key]:
            continue
        print(f"-- Panel B [{key}]: PCC vs distance-to-known --")
        for d, m, sm in zip(*_agg(iv[key])):
            print(f"  d {int(d)}: PCC {m:.3f} ± {sm:.3f}")


if __name__ == "__main__":
    main()
