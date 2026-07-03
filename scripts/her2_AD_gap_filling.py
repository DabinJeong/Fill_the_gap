"""HER2 A-D (6 evenly-spaced sections): how well are MISSING sections predicted as the
known sections get sparser?

Only A,B,C,D have 6 sections (levels 1..6, evenly spaced). We hold out sections and
predict them from the remaining (known) ones with the HistoGS init field. Each predicted
section is classified by:
  - d = distance (in section-spacing units) to the nearest KNOWN section
  - INTERPOLATION = a known section exists on BOTH sides (predicted section is bracketed)
    vs EXTRAPOLATION = known sections only on one side.

Several "known-set" patterns span sparsities from every-other (1,3,5) to a single anchor.
Reported: ASIGN per-spot PCC (and MSE) of predicted vs measured, plus a copy-nearest-known
baseline, aggregated over A-D and binned by (d, interpolation/extrapolation).

Run:
  cd /nfs/team361/dj16/projects/HistoGS
  .venv/bin/python scripts/her2_AD_gap_filling.py --config configs/her2_A.yaml
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

SAMPLES_6 = ["A", "B", "C", "D"]          # the evenly-spaced 6-section HER2 tumors
KNOWN_PATTERNS = [
    {1, 3, 5},     # every-other (densest gap-filling): predict 2,4,6
    {1, 4},        # two anchors, 1/3 density
    {1, 6},        # ends only: predict the whole interior 2,3,4,5
    {2, 5},        # two interior anchors
    {1, 2},        # adjacent pair at the top -> mostly extrapolation
    {1},           # single anchor (pure reach)
]


def match_xy(xa, xb):
    lut = {tuple(p): i for i, p in enumerate(xb)}
    ia, ib = [], []
    for i, p in enumerate(xa):
        j = lut.get(tuple(p))
        if j is not None:
            ia.append(i); ib.append(j)
    return np.array(ia, int), np.array(ib, int)


def main():
    ap = ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=os.path.join(_ROOT, "figures"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg["device"]); rd = cfg["render"]
    root, name = cfg["dataset"]["root"], cfg["dataset"]["name"]

    # collect points: key (regime, d) -> list of pccs ; plus mse and copy baseline
    pcc = defaultdict(list); mse = defaultdict(list); copy = defaultdict(list)
    # also per-pattern summary
    per_pat = defaultdict(lambda: defaultdict(list))

    for s in SAMPLES_6:
        raw = load_sample_npy(sample_npy_path(root, name, s))
        xyz, _ = normalize_coords(raw["xy"], raw["level"], z_spacing=cfg["dataset"]["z_spacing"])
        xy, lv, gex = raw["xy"], raw["level"], raw["gex"]
        levels = sorted(set(lv.tolist()))

        for known in KNOWN_PATTERNS:
            known = set(known) & set(levels)
            if not known:
                continue
            km = np.isin(lv, list(known))
            field = GaussianModel(G=gex.shape[1], device=device,
                                  densify_z=cfg["model"]["densify_z"])
            field.create_from_pcd(
                torch.from_numpy(xyz[km]).to(device), torch.from_numpy(gex[km]).to(device),
                init_opacity=cfg["model"]["init_opacity"], scale_knn=cfg["model"]["scale_knn"],
                init_scale_mult=cfg["model"].get("init_scale_mult", 1.0))

            for t in levels:
                if t in known:
                    continue
                d = min(abs(t - k) for k in known)
                bracketed = any(k < t for k in known) and any(k > t for k in known)
                regime = "interp" if bracketed else "extrap"
                tm = lv == t
                q = torch.from_numpy(xyz[tm]).to(device)
                with torch.no_grad():
                    pred = render(field, q, k=rd["knn_k"], chunk=rd["chunk"],
                                  return_aux=False)["gex"].cpu().numpy()
                tgt = gex[tm]
                p = asign_pcc(pred, tgt)
                pcc[(regime, d)].append(p)
                mse[(regime, d)].append(mse_mae(pred, tgt)["mse"])
                per_pat[tuple(sorted(known))][t].append(p)
                # copy from nearest known section (matched by xy)
                nk = min(known, key=lambda k: abs(t - k))
                ia, ib = match_xy(xy[tm], xy[lv == nk])
                if len(ia) >= 5:
                    copy[(regime, d)].append(asign_pcc(gex[tm][ia], gex[lv == nk][ib]))

    # ---- figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
    for regime, color, lab in [("interp", "C2", "interpolation (known on both sides)"),
                               ("extrap", "C0", "extrapolation (known on one side)")]:
        ds = sorted({d for (r, d) in pcc if r == regime})
        m = [np.mean(pcc[(regime, d)]) for d in ds]
        se = [np.std(pcc[(regime, d)]) / max(1, np.sqrt(len(pcc[(regime, d)]))) for d in ds]
        axL.errorbar(ds, m, yerr=se, marker="o", color=color, capsize=3, lw=2, label=lab)
    # copy baseline (all regimes pooled by d)
    dcopy = sorted({d for (_, d) in copy})
    if dcopy:
        cm = [np.mean([v for (r, d2), vs in copy.items() if d2 == d for v in vs]) for d in dcopy]
        axL.plot(dcopy, cm, marker="x", ls="--", color="0.5", label="copy nearest known (baseline)")
    axL.set_title("HER2 A-D: predicting held-out sections vs gap to nearest known")
    axL.set_xlabel("distance to nearest known section (Δlevel)")
    axL.set_ylabel("ASIGN per-spot PCC (predicted vs measured)")
    axL.grid(alpha=0.3); axL.legend(fontsize=9); axL.set_xticks(range(1, 6))

    # right: per-pattern bar of mean PCC over predicted sections
    labels, vals = [], []
    for known in KNOWN_PATTERNS:
        kk = tuple(sorted(set(known)))
        allp = [v for t in per_pat[kk] for v in per_pat[kk][t]]
        if not allp:
            continue
        pred_secs = sorted(set(range(1, 7)) - set(kk))
        labels.append(f"known {{{','.join(map(str,kk))}}}\n→pred {{{','.join(map(str,pred_secs))}}}")
        vals.append(np.mean(allp))
    y = np.arange(len(labels))
    axR.barh(y, vals, color="C2", alpha=0.8)
    axR.set_yticks(y); axR.set_yticklabels(labels, fontsize=8)
    axR.invert_yaxis()
    axR.set_xlabel("mean ASIGN PCC over predicted sections")
    axR.set_title("Sparser known sets → how well the missing sections are filled")
    for yi, v in zip(y, vals):
        axR.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=8)
    axR.grid(alpha=0.3, axis="x")

    fig.suptitle("HER2 A-D (6 evenly-spaced sections): filling missing sections as sampling thins",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(args.outdir, f"{name}_AD_gap_filling.png")
    os.makedirs(args.outdir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}\n")

    print("=== PCC by regime & distance-to-nearest-known (A-D) ===")
    for regime in ("interp", "extrap"):
        for d in sorted({d for (r, d) in pcc if r == regime}):
            v = pcc[(regime, d)]
            print(f"  {regime:7s} d={d}: PCC {np.mean(v):.3f} ± {np.std(v)/max(1,np.sqrt(len(v))):.3f} (n={len(v)})")
    print("\n=== per known-set pattern (mean PCC over predicted sections) ===")
    for known in KNOWN_PATTERNS:
        kk = tuple(sorted(set(known)))
        allp = [v for t in per_pat[kk] for v in per_pat[kk][t]]
        if allp:
            print(f"  known {kk} -> pred {sorted(set(range(1,7))-set(kk))}: PCC {np.mean(allp):.3f}")


if __name__ == "__main__":
    main()
