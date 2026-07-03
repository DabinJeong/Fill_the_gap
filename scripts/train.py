"""Train a 3D Gaussian gene-expression field on the observed (train) sections.

Per sample: build the scene, initialize Gaussians from the train-section point
cloud, then iterate render -> gex loss -> backward -> densify/prune -> step.
"""
import os
import sys
from argparse import ArgumentParser

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loss import gex_loss
from renderer import render
from scene import GaussianModel, Scene
from utils.config import load_config
from utils.general_utils import get_device, seed_everything


def train_one_sample(cfg, sample):
    device = get_device(cfg["device"])
    d = cfg["dataset"]
    scene = Scene(
        sample, d["root"], d["name"],
        held_out_parity=d["held_out_parity"],
        train_levels=d["train_levels"], heldout_levels=d["heldout_levels"],
        z_spacing=d["z_spacing"], device=device,
    )
    xyz_tr, gex_tr = scene.train_points()
    xyz_ho, gex_ho, _ = scene.heldout_points()
    extent = scene.scene_extent

    gaussians = GaussianModel(G=scene.G, device=device, densify_z=cfg["model"]["densify_z"])
    gaussians.create_from_pcd(
        *scene.init_pcd(),
        init_opacity=cfg["model"]["init_opacity"],
        scale_knn=cfg["model"]["scale_knn"],
        init_scale_mult=cfg["model"].get("init_scale_mult", 1.0),
    )
    gaussians.training_setup(cfg)

    ds = cfg["densify"]
    rd = cfg["render"]
    iters = cfg["optim"]["iterations"]
    log_every = cfg.get("log_interval", 200)

    for it in range(1, iters + 1):
        out = render(gaussians, xyz_tr, k=rd["knn_k"], chunk=rd["chunk"], return_aux=True)
        loss, logs = gex_loss(out["gex"], gex_tr, gaussians, cfg)

        gaussians.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gaussians.add_densification_stats(out["vis_count"] > 0)

        with torch.no_grad():
            if ds["start"] < it < ds["end"] and it % ds["interval"] == 0:
                gaussians.densify_and_prune(
                    grad_thresh=ds["grad_thresh"], min_opacity=ds["min_opacity"],
                    scene_extent=extent, max_gaussians=ds["max_gaussians"],
                )
            if ds["opacity_reset_interval"] and it % ds["opacity_reset_interval"] == 0:
                gaussians.reset_opacity()

        gaussians.optimizer.step()

        if it % log_every == 0 or it == 1:
            with torch.no_grad():
                ho = render(gaussians, xyz_ho, k=rd["knn_k"], chunk=rd["chunk"],
                            return_aux=False)["gex"]
                ho_mse = torch.nn.functional.mse_loss(ho, gex_ho).item()
            print(f"[{sample}] it {it:5d} | mse {logs['mse']:.4f} "
                  f"| held-out mse {ho_mse:.4f} | N {gaussians.n_points}")

    out_dir = os.path.join(cfg["output_dir"], str(sample))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {"gaussians": gaussians.capture(), "cfg": dict(cfg),
         "norm_meta": scene.norm_meta, "sample": sample},
        os.path.join(out_dir, "chkpnt.pth"),
    )
    print(f"[{sample}] saved checkpoint -> {out_dir}/chkpnt.pth")
    return gaussians, scene


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sample", type=str, default=None,
                        help="override config sample list with a single sample")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    samples = [args.sample] if args.sample else cfg["dataset"]["samples"]
    for sample in samples:
        train_one_sample(cfg, sample)


if __name__ == "__main__":
    main()
