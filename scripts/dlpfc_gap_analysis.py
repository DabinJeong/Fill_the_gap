"""DLPFC: interpolation/prediction accuracy across the documented 300 µm gap vs 10 µm.

DLPFC (Maynard 2021) sections sit at physical depths z = {0, 10, 310, 320} µm per donor:
two 10-µm-adjacent pairs separated by a 300 µm gap. Unlike HER2/ST (≤32 µm, where through-
plane decorrelation is below the noise floor), this lets us probe a genuinely large gap.

In-plane Visium pitch ≈ 100 µm/spot; we build the HistoGS field in real micrometres
(x,y = array_coord × 100 µm, z = section depth µm) so the kNN geometry is physically
faithful. Spots are matched across sections by Visium array coordinate (validated: array-
coord Jaccard 0.77-0.96 across sections, incl. across the gap).

Two views, aggregated over the 3 donors:
  A) model-free decorrelation: measured↔measured ASIGN PCC vs physical distance {10,300,310,320}.
  B) HistoGS prediction PCC: predict a section at 10 µm (its pair-mate) vs across the 300 µm
     gap (predict the far pair from the near pair), with a copy-nearest baseline.

Run:
  cd /nfs/team361/dj16/projects/HistoGS
  .venv/bin/python scripts/dlpfc_gap_analysis.py --config configs/her2_A.yaml
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
from utils.data_utils import load_sample_npy
from utils.general_utils import get_device

DLPFC_ROOT = ("/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_serial_sections/"
              "DLPFC_3D/3D_npy_information")
DONORS = ["donor1", "donor2", "donor3"]
Z_UM = {1: 0.0, 2: 10.0, 3: 310.0, 4: 320.0}      # physical depth per section
PITCH_UM = 100.0                                    # Visium in-plane spot pitch (~100 µm)


def match_xy(xa, xb):
    lut = {tuple(p): i for i, p in enumerate(xb)}
    ia, ib = [], []
    for i, p in enumerate(xa):
        j = lut.get(tuple(p))
        if j is not None:
            ia.append(i); ib.append(j)
    return np.array(ia, int), np.array(ib, int)


def load_donor(donor):
    raw = load_sample_npy(f"{DLPFC_ROOT}/{donor}.npy")
    xy, lv, gex = raw["xy"], raw["level"], raw["gex"]
    xyz_um = np.empty((len(lv), 3), np.float32)
    xyz_um[:, 0] = xy[:, 0] * PITCH_UM
    xyz_um[:, 1] = xy[:, 1] * PITCH_UM
    xyz_um[:, 2] = np.array([Z_UM[int(l)] for l in lv], np.float32)
    return xy, lv, gex, xyz_um


def build_field(cfg, device, means_um, feats):
    g = GaussianModel(G=feats.shape[1], device=device, densify_z=cfg["model"]["densify_z"])
    g.create_from_pcd(torch.from_numpy(means_um).to(device), torch.from_numpy(feats).to(device),
                      init_opacity=cfg["model"]["init_opacity"], scale_knn=cfg["model"]["scale_knn"],
                      init_scale_mult=cfg["model"].get("init_scale_mult", 1.0))
    return g


def main():
    ap = ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=os.path.join(_ROOT, "figures"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg["device"]); rd = cfg["render"]

    deco = defaultdict(list)                 # phys distance µm -> [pcc]
    pred = defaultdict(list); copy = defaultdict(list)   # "10µm"/"300µm" -> [pcc]

    for donor in DONORS:
        xy, lv, gex, xyz_um = load_donor(donor)

        # ---- A) measured<->measured decorrelation vs physical distance ----
        for a, b in [(1, 2), (3, 4), (2, 3), (1, 3), (2, 4), (1, 4)]:
            dist = abs(Z_UM[b] - Z_UM[a])
            ia, ib = match_xy(xy[lv == a], xy[lv == b])
            if len(ia) >= 5:
                deco[dist].append(asign_pcc(gex[lv == a][ia], gex[lv == b][ib]))

        # ---- B) HistoGS prediction at 10 µm vs across the 300 µm gap ----
        # 10 µm: predict a section from its adjacent pair-mate
        for known_l, tgt_l in [(1, 2), (2, 1), (3, 4), (4, 3)]:
            km = lv == known_l
            field = build_field(cfg, device, xyz_um[km], gex[km])
            tm = lv == tgt_l
            with torch.no_grad():
                pr = render(field, torch.from_numpy(xyz_um[tm]).to(device),
                            k=rd["knn_k"], chunk=rd["chunk"], return_aux=False)["gex"].cpu().numpy()
            pred["10µm"].append(asign_pcc(pr, gex[tm]))
            ia, ib = match_xy(xy[tm], xy[km])
            if len(ia) >= 5:
                copy["10µm"].append(asign_pcc(gex[tm][ia], gex[km][ib]))

        # 300 µm: predict the far pair from the near pair (nearest known across the gap)
        for known_ls, tgt_l in [((1, 2), 3), ((3, 4), 2)]:
            km = np.isin(lv, known_ls)
            field = build_field(cfg, device, xyz_um[km], gex[km])
            tm = lv == tgt_l
            with torch.no_grad():
                pr = render(field, torch.from_numpy(xyz_um[tm]).to(device),
                            k=rd["knn_k"], chunk=rd["chunk"], return_aux=False)["gex"].cpu().numpy()
            pred["300µm"].append(asign_pcc(pr, gex[tm]))
            nk = known_ls[1] if abs(known_ls[1] - tgt_l) < abs(known_ls[0] - tgt_l) else known_ls[0]
            ia, ib = match_xy(xy[tm], xy[lv == nk])
            if len(ia) >= 5:
                copy["300µm"].append(asign_pcc(gex[tm][ia], gex[lv == nk][ib]))

    # ---- figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.3))
    ds = sorted(deco)
    axA.errorbar(ds, [np.mean(deco[d]) for d in ds],
                 yerr=[np.std(deco[d]) / max(1, np.sqrt(len(deco[d]))) for d in ds],
                 marker="o", color="C3", capsize=3, lw=2)
    axA.axvspan(290, 330, color="orange", alpha=0.12)
    axA.set_title("A. DLPFC measured↔measured correlation vs physical distance")
    axA.set_xlabel("physical distance between sections (µm)")
    axA.set_ylabel("ASIGN per-spot PCC"); axA.grid(alpha=0.3)
    axA.set_xticks(ds)
    axA.text(0.62, 0.9, "300 µm gap region", transform=axA.transAxes,
             fontsize=9, color="darkorange", ha="center")

    groups = ["10µm", "300µm"]
    x = np.arange(len(groups)); w = 0.35
    pm = [np.mean(pred[g]) for g in groups]
    pe = [np.std(pred[g]) / max(1, np.sqrt(len(pred[g]))) for g in groups]
    cm = [np.mean(copy[g]) for g in groups]
    axB.bar(x - w/2, pm, w, yerr=pe, capsize=4, color="C0", label="HistoGS prediction")
    axB.bar(x + w/2, cm, w, color="0.6", label="copy nearest known")
    for xi, v in zip(x - w/2, pm): axB.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + w/2, cm): axB.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    axB.set_xticks(x); axB.set_xticklabels(["predict 10 µm\n(pair-mate)",
                                            "predict across\n300 µm gap"])
    axB.set_ylabel("ASIGN per-spot PCC (predicted vs measured)")
    axB.set_title("B. HistoGS prediction: 10 µm vs across the 300 µm gap")
    axB.grid(alpha=0.3, axis="y"); axB.legend(fontsize=9)

    fig.suptitle("DLPFC (Maynard 2021), 3 donors — interpolation across the 300 µm gap", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(args.outdir, "DLPFC_300um_gap.png")
    os.makedirs(args.outdir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {path}\n")

    print("=== A. measured↔measured PCC vs physical distance ===")
    for d in ds:
        print(f"  {int(d):>3} µm: PCC {np.mean(deco[d]):.3f} ± {np.std(deco[d])/max(1,np.sqrt(len(deco[d]))):.3f} (n={len(deco[d])})")
    print("=== B. HistoGS prediction vs copy baseline ===")
    for g in groups:
        print(f"  {g:>5}: HistoGS {np.mean(pred[g]):.3f}  | copy {np.mean(copy[g]):.3f}  (n={len(pred[g])})")


if __name__ == "__main__":
    main()
