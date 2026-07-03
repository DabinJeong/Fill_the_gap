"""Evaluate HistoGS under ASIGN's EXACT 3D protocol.

ASIGN 3D sample-wise benchmark protocol (from hrlblab/ASIGN dataloader_3d.py):
  - For each test sample, section level==1 (top section) is KNOWN.
  - Predict gene expression at ALL level>1 spots.
  - Metrics are computed on the level>1 spots only, per sample, then averaged
    over the test samples (here HER2 A..H, the same 8 samples ASIGN reports).

HistoGS is a per-sample Gaussian-field interpolator (no cross-sample training).
The winning regime (see histogs-optimization-overfits memory) is the *initialized*
field evaluated directly (optim.iterations: 0): means = known-spot xyz, features =
measured known gex, anisotropic covariance from kNN distance. We init the field from
ONLY the level==1 plane and query it at the level>1 spots. With a single known plane
HistoGS necessarily degenerates toward in-plane interpolation of the top section
(there is no through-plane signal to interpolate), so we also report a literal
copy-top-section baseline for reference.

Run:
  export PYTHONPATH=/nfs/team361/dj16/projects/HistoGS
  /software/teamtrynka/ks37/envs/torch/bin/python evals/run_eval.py \
      --config configs/her2.yaml
"""
import csv
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.metrics import asign_metrics
from renderer import render
from scene import GaussianModel, Scene
from utils.config import load_config
from utils.general_utils import get_device

# ASIGN HER2 evaluates these 8 samples (4-fold sample-wise CV, every sample is held
# out exactly once). HistoGS has no training stage, so the fold grouping is irrelevant
# to it -- each sample is imputed independently from its own level-1 plane.
HER2_SAMPLES = ["A", "B", "C", "D", "E", "F", "G", "H"]


def copy_top_section(scene):
    """Baseline: each level>1 spot copies the measured gex of the nearest (in xy)
    level==1 spot -- i.e. literally propagate the known top section downward."""
    xyz_tr, gex_tr = scene.train_points()                 # level==1
    xyz_ho, _, _ = scene.heldout_points()                 # level>1
    # nearest neighbour in the in-plane (x, y) coordinates only
    d = torch.cdist(xyz_ho[:, :2], xyz_tr[:, :2])
    nn = d.argmin(dim=1)
    return gex_tr[nn].cpu().numpy()


def eval_one_sample(cfg, sample):
    device = get_device(cfg["device"])
    d = cfg["dataset"]
    # ASIGN protocol: known = level 1, predict = levels > 1
    scene = Scene(
        sample, d["root"], d["name"],
        held_out_parity=d["held_out_parity"],
        train_levels=[1], heldout_levels=[2, 3, 4, 5, 6],
        z_spacing=d["z_spacing"], device=device,
    )
    xyz_ho, gex_ho, level_ho = scene.heldout_points()

    gaussians = GaussianModel(G=scene.G, device=device,
                              densify_z=cfg["model"]["densify_z"])
    gaussians.create_from_pcd(
        *scene.init_pcd(),
        init_opacity=cfg["model"]["init_opacity"],
        scale_knn=cfg["model"]["scale_knn"],
        init_scale_mult=cfg["model"].get("init_scale_mult", 1.0),
    )

    with torch.no_grad():
        pred = render(gaussians, xyz_ho, k=cfg["render"]["knn_k"],
                      chunk=cfg["render"]["chunk"], return_aux=False)["gex"]

    pred_np = pred.cpu().numpy()
    target_np = gex_ho.cpu().numpy()
    copy_np = copy_top_section(scene)

    gs = asign_metrics(pred_np, target_np)
    cp = asign_metrics(copy_np, target_np)
    n_known = int(scene.train_mask.sum().item())
    n_pred = int(scene.heldout_mask.sum().item())
    return {
        "sample": sample, "n_known_level1": n_known, "n_pred_level_gt1": n_pred,
        **{f"gs_{k}": v for k, v in gs.items()},
        **{f"copy_{k}": v for k, v in cp.items()},
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--samples", type=str, default=None,
                        help="comma-separated sample list (default: all HER2 A..H)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = args.samples.split(",") if args.samples else HER2_SAMPLES

    rows = [eval_one_sample(cfg, s) for s in samples]

    print("\n=== HistoGS under ASIGN protocol (per sample, metrics on level>1) ===")
    print(f"  {'sample':<7}{'#known':>7}{'#pred':>7}  "
          f"{'HistoGS PCC':>12}{'MSE':>9}{'MAE':>9}   "
          f"{'copy PCC':>9}{'MSE':>9}{'MAE':>9}")
    for r in rows:
        print(f"  {r['sample']:<7}{r['n_known_level1']:>7}{r['n_pred_level_gt1']:>7}  "
              f"{r['gs_pcc']:>12.3f}{r['gs_mse']:>9.4f}{r['gs_mae']:>9.4f}   "
              f"{r['copy_pcc']:>9.3f}{r['copy_mse']:>9.4f}{r['copy_mae']:>9.4f}")

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    print("\n=== mean over samples (ASIGN benchmark protocol) ===")
    print(f"  HistoGS  PCC {mean('gs_pcc'):.4f}  MSE {mean('gs_mse'):.4f}  MAE {mean('gs_mae'):.4f}")
    print(f"  copy-top PCC {mean('copy_pcc'):.4f}  MSE {mean('copy_mse'):.4f}  MAE {mean('copy_mae'):.4f}")

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "histogs_asign_protocol.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "protocol": "ASIGN 3D sample-wise: level1 known -> predict level>1, metrics on level>1",
        "samples": samples,
        "histogs_mean": {"pcc": mean("gs_pcc"), "mse": mean("gs_mse"), "mae": mean("gs_mae")},
        "copy_top_mean": {"pcc": mean("copy_pcc"), "mse": mean("copy_mse"), "mae": mean("copy_mae")},
        "per_sample": rows,
    }
    with open(os.path.join(out_dir, "histogs_asign_protocol.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
