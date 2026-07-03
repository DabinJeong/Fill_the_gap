"""Render held-out sections: query the trained field at held-out spot coordinates."""
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderer import render
from scene import GaussianModel, Scene
from utils.config import load_config
from utils.general_utils import get_device


def _load_model(cfg, sample, device):
    ckpt_path = os.path.join(cfg["output_dir"], str(sample), "chkpnt.pth")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    gaussians = GaussianModel(G=state["gaussians"]["G"], device=device)
    gaussians.restore(state["gaussians"])
    return gaussians


def render_one_sample(cfg, sample, save=True):
    device = get_device(cfg["device"])
    d = cfg["dataset"]
    scene = Scene(
        sample, d["root"], d["name"],
        held_out_parity=d["held_out_parity"],
        train_levels=d["train_levels"], heldout_levels=d["heldout_levels"],
        z_spacing=d["z_spacing"], device=device,
    )
    gaussians = _load_model(cfg, sample, device)

    xyz_ho, gex_ho, level_ho = scene.heldout_points()
    with torch.no_grad():
        pred = render(gaussians, xyz_ho, k=cfg["render"]["knn_k"],
                      chunk=cfg["render"]["chunk"], return_aux=False)["gex"]

    pred_np = pred.cpu().numpy()
    target_np = gex_ho.cpu().numpy()
    level_np = level_ho.cpu().numpy()

    if save:
        out_dir = os.path.join(cfg["output_dir"], str(sample))
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, "pred_heldout.npz"),
            pred=pred_np, target=target_np, level=level_np,
            xy=scene.xy[scene.heldout_mask.cpu().numpy()],
        )
        print(f"[{sample}] saved predictions -> {out_dir}/pred_heldout.npz "
              f"({pred_np.shape[0]} spots)")
    return pred_np, target_np, level_np


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sample", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = [args.sample] if args.sample else cfg["dataset"]["samples"]
    for sample in samples:
        render_one_sample(cfg, sample)


if __name__ == "__main__":
    main()
