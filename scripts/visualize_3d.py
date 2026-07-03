"""Visualize the 3D object reconstructed by the Gaussian field.

The trained Gaussians define a continuous field f: R^3 -> R^250. We sample it on a
dense 3D grid spanning the tissue (at finer z resolution than the observed
sections, so the interpolated depths are filled in), keep voxels that fall inside
the tissue (where the field has support), and render the resulting volume as a 3D
point cloud coloured by the dominant molecular axis (PC1 of expression).

Outputs (under <project>/figures/):
  <tag>_3d_volume.png    - reconstructed volume from 4 viewing angles
  <tag>_3d_rotation.gif  - rotating view of the reconstructed volume
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# project-local pip target (plotly, for the interactive HTML)
sys.path.insert(0, os.path.join(_ROOT, ".pylibs"))

from renderer import render
from scene import Scene
from scripts.render import _load_model
from utils.config import load_config
from utils.general_utils import get_device


def _query_field(gaussians, pts, k, chunk=8192):
    """Return (gex (M,G), support (M,)) for query points (M,3)."""
    gex, support = [], []
    for s in range(0, pts.shape[0], chunk):
        out = render(gaussians, pts[s:s + chunk], k=k, chunk=chunk, return_aux=True)
        gex.append(out["gex"])
        support.append(out["weights"].sum(dim=1))
    return torch.cat(gex), torch.cat(support)


def reconstruct_volume(cfg, sample, n_xy=44, n_z=40, support_q=0.4, z_exag=None,
                       jitter=True, seed=0):
    device = get_device(cfg["device"])
    d = cfg["dataset"]
    scene = Scene(sample, d["root"], d["name"], held_out_parity=d["held_out_parity"],
                  train_levels=d["train_levels"], heldout_levels=d["heldout_levels"],
                  z_spacing=d["z_spacing"], device=device)
    gaussians = _load_model(cfg, sample, device)

    xyz = scene.xyz.cpu().numpy()
    lo, hi = xyz.min(0), xyz.max(0)

    # dense 3D grid spanning the tissue; z finer than observed sections
    gx = np.linspace(lo[0], hi[0], n_xy)
    gy = np.linspace(lo[1], hi[1], n_xy)
    gz = np.linspace(lo[2], hi[2], n_z)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    grid = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1).astype(np.float32)

    # Jitter points off the regular lattice so the volume looks continuous rather
    # than showing the sampling rows edge-on (the field itself is continuous).
    if jitter:
        rng = np.random.default_rng(seed)
        cell = np.array([gx[1] - gx[0] if n_xy > 1 else 0,
                         gy[1] - gy[0] if n_xy > 1 else 0,
                         gz[1] - gz[0] if n_z > 1 else 0], dtype=np.float32)
        grid = grid + rng.uniform(-0.5, 0.5, grid.shape).astype(np.float32) * cell

    pts = torch.from_numpy(grid).to(device)
    with torch.no_grad():
        gex, support = _query_field(gaussians, pts, k=cfg["render"]["knn_k"])
        # calibrate the "inside tissue" threshold against support at observed spots
        _, sup_obs = _query_field(gaussians, scene.xyz, k=cfg["render"]["knn_k"])
    gex = gex.cpu().numpy()
    support = support.cpu().numpy()
    thresh = np.quantile(sup_obs.cpu().numpy(), support_q)
    keep = support >= thresh

    grid, gex = grid[keep], gex[keep]

    # PC1 of expression (fit on observed train spots) as the colour scalar
    obs = scene.gex[scene.train_mask].cpu().numpy()
    mu = obs.mean(0)
    _, _, Vt = np.linalg.svd(obs - mu, full_matrices=False)
    pc1 = (gex - mu) @ Vt[0]

    # z exaggeration for display (serial sections form a thin slab)
    dx, dy, dz = hi - lo
    if z_exag is None:
        z_exag = max(1.0, 0.55 * max(dx, dy) / max(dz, 1e-6))
    return scene, grid, pc1, (dx, dy, dz), z_exag, gaussians


def reconstruct_split(cfg, sample, n_xy=44, n_z=40, support_q=0.4, z_exag=None,
                      seed=0):
    """Split reconstruction into MEASURED observed-section spots vs IMPUTED field.

    Returns:
        obs   = dict(xyz (No,3), pc1 (No,), level (No,))   measured train-section spots
        imp   = dict(xyz (Ni,3), pc1 (Ni,))                interpolated field samples at
                                                            depths away from observed planes
        dims, z_exag, scene
    All PC1 values share one basis (fit on observed train gex) so colours are comparable.
    """
    device = get_device(cfg["device"])
    d = cfg["dataset"]
    scene = Scene(sample, d["root"], d["name"], held_out_parity=d["held_out_parity"],
                  train_levels=d["train_levels"], heldout_levels=d["heldout_levels"],
                  z_spacing=d["z_spacing"], device=device)
    gaussians = _load_model(cfg, sample, device)

    xyz = scene.xyz.cpu().numpy()
    lo, hi = xyz.min(0), xyz.max(0)

    # shared colour basis: PC1 of expression fit on the observed (measured) train spots
    obs_gex = scene.gex[scene.train_mask].cpu().numpy()
    mu = obs_gex.mean(0)
    _, _, Vt = np.linalg.svd(obs_gex - mu, full_matrices=False)
    pc1_axis = Vt[0]

    # observed measured spots (ground truth at the known sections)
    obs_xyz = scene.xyz[scene.train_mask].cpu().numpy()
    obs_lv = scene.level[scene.train_mask].cpu().numpy()
    obs_pc1 = (obs_gex - mu) @ pc1_axis

    # observed-plane z values (to exclude from the interpolated cloud)
    obs_z = np.unique(np.round(obs_xyz[:, 2], 6))

    # dense grid; z spans the slab finer than the sections
    gx = np.linspace(lo[0], hi[0], n_xy)
    gy = np.linspace(lo[1], hi[1], n_xy)
    gz = np.linspace(lo[2], hi[2], n_z)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    grid = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    cell = np.array([gx[1] - gx[0] if n_xy > 1 else 0,
                     gy[1] - gy[0] if n_xy > 1 else 0,
                     gz[1] - gz[0] if n_z > 1 else 0], dtype=np.float32)
    grid = grid + rng.uniform(-0.5, 0.5, grid.shape).astype(np.float32) * cell

    pts = torch.from_numpy(grid).to(device)
    with torch.no_grad():
        gex, support = _query_field(gaussians, pts, k=cfg["render"]["knn_k"])
        _, sup_obs = _query_field(gaussians, scene.xyz, k=cfg["render"]["knn_k"])
    gex = gex.cpu().numpy(); support = support.cpu().numpy()
    thresh = np.quantile(sup_obs.cpu().numpy(), support_q)
    keep = support >= thresh

    # imputed = field samples inside tissue, at depths NOT on an observed plane
    dz_cell = (gz[1] - gz[0]) if n_z > 1 else 1.0
    off_plane = np.min(np.abs(grid[:, 2][:, None] - obs_z[None, :]), axis=1) > 0.5 * dz_cell
    keep_imp = keep & off_plane
    imp_xyz = grid[keep_imp]
    imp_pc1 = (gex[keep_imp] - mu) @ pc1_axis

    dx, dy, dz = hi - lo
    if z_exag is None:
        z_exag = max(1.0, 0.55 * max(dx, dy) / max(dz, 1e-6))
    obs = {"xyz": obs_xyz, "pc1": obs_pc1, "level": obs_lv}
    imp = {"xyz": imp_xyz, "pc1": imp_pc1}
    return obs, imp, (dx, dy, dz), z_exag, scene


def fig_split_volume(obs, imp, dims, z_exag, path, sample):
    """3 panels: measured observed sections | imputed field | overlay."""
    dx, dy, dz = dims
    vmin = float(np.quantile(np.concatenate([obs["pc1"], imp["pc1"]]), 0.02))
    vmax = float(np.quantile(np.concatenate([obs["pc1"], imp["pc1"]]), 0.98))

    fig = plt.figure(figsize=(18, 6))
    panels = [
        ("measured sections (observed)", "obs"),
        ("imputed / interpolated field", "imp"),
        ("overlay (measured = bold, imputed = faint)", "both"),
    ]
    for i, (title, what) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        if what in ("imp", "both"):
            ax.scatter(imp["xyz"][:, 0], imp["xyz"][:, 1], imp["xyz"][:, 2] * z_exag,
                       c=imp["pc1"], cmap="Spectral_r", vmin=vmin, vmax=vmax,
                       s=4, alpha=0.18 if what == "both" else 0.45, edgecolors="none")
        if what in ("obs", "both"):
            p = ax.scatter(obs["xyz"][:, 0], obs["xyz"][:, 1], obs["xyz"][:, 2] * z_exag,
                           c=obs["pc1"], cmap="Spectral_r", vmin=vmin, vmax=vmax,
                           s=14, alpha=0.95, edgecolors="k", linewidths=0.15)
        else:
            p = ax.collections[-1]
        ax.set_box_aspect((dx, dy, dz * z_exag))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (depth)")
        ax.view_init(elev=14, azim=-60)
        ax.set_title(title, fontsize=11)
    cb = fig.colorbar(p, ax=fig.axes, shrink=0.5, pad=0.02)
    cb.set_label("expression PC1 (shared basis)")
    n_lv = len(np.unique(obs["level"]))
    fig.suptitle(
        f"HistoGS {sample}: measured vs imputed  "
        f"({obs['xyz'].shape[0]:,} measured spots on {n_lv} sections; "
        f"{imp['xyz'].shape[0]:,} imputed field samples; z exaggerated x{z_exag:.1f})",
        fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_split_interactive(obs, imp, dims, z_exag, path, title):
    """Interactive HTML with two toggleable traces: measured vs imputed."""
    import plotly.graph_objects as go
    dx, dy, dz = dims
    zr = dz * z_exag
    m = max(dx, dy, zr)
    vmin = float(np.quantile(np.concatenate([obs["pc1"], imp["pc1"]]), 0.02))
    vmax = float(np.quantile(np.concatenate([obs["pc1"], imp["pc1"]]), 0.98))
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        name="imputed / interpolated",
        x=imp["xyz"][:, 0], y=imp["xyz"][:, 1], z=imp["xyz"][:, 2] * z_exag,
        mode="markers",
        marker=dict(size=2.0, color=imp["pc1"], colorscale="Spectral_r",
                    cmin=vmin, cmax=vmax, opacity=0.35,
                    colorbar=dict(title="PC1"), showscale=True),
        hovertemplate="imputed PC1=%{marker.color:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        name="measured (observed sections)",
        x=obs["xyz"][:, 0], y=obs["xyz"][:, 1], z=obs["xyz"][:, 2] * z_exag,
        mode="markers",
        marker=dict(size=3.6, color=obs["pc1"], colorscale="Spectral_r",
                    cmin=vmin, cmax=vmax, opacity=0.95,
                    line=dict(width=0.5, color="black"), showscale=False),
        hovertemplate="measured PC1=%{marker.color:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        legend=dict(itemsizing="constant", orientation="h", y=1.02),
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z (depth)",
                   aspectmode="manual",
                   aspectratio=dict(x=dx / m, y=dy / m, z=zr / m)),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def _scatter(ax, grid, pc1, z_exag, dims):
    dx, dy, dz = dims
    p = ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2] * z_exag,
                   c=pc1, cmap="Spectral_r", s=6, alpha=0.5,
                   edgecolors="none", depthshade=True)
    ax.set_box_aspect((dx, dy, dz * z_exag))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (depth)")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    return p


def fig_volume(grid, pc1, dims, z_exag, path, n_obs, n_pred):
    fig = plt.figure(figsize=(12, 10))
    angles = [(22, -60), (22, 30), (8, -90), (60, -60)]
    for i, (elev, azim) in enumerate(angles, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        p = _scatter(ax, grid, pc1, z_exag, dims)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"view {i}", fontsize=9)
    cb = fig.colorbar(p, ax=fig.axes, shrink=0.5, pad=0.02)
    cb.set_label("expression PC1")
    fig.suptitle(
        f"Reconstructed 3D ST volume  (z exaggerated x{z_exag:.1f})\n"
        f"{n_pred:,} field samples — observed sections + interpolated depths",
        fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_rotation(grid, pc1, dims, z_exag, path, n_frames=120, fps=10):
    """Rotating GIF. Slower rotation = more frames and/or lower fps.
    seconds per revolution = n_frames / fps; deg per frame = 360 / n_frames."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _scatter(ax, grid, pc1, z_exag, dims)
    ax.set_title("Reconstructed 3D ST volume (PC1)", fontsize=11)

    def update(f):
        ax.view_init(elev=18, azim=f * (360 / n_frames))
        return ()

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def fig_interactive(grid, pc1, dims, z_exag, path, title):
    """Standalone, mouse-rotatable 3D scatter (Plotly, self-contained HTML)."""
    import plotly.graph_objects as go

    dx, dy, dz = dims
    zr = dz * z_exag
    m = max(dx, dy, zr)
    fig = go.Figure(
        go.Scatter3d(
            x=grid[:, 0], y=grid[:, 1], z=grid[:, 2] * z_exag,
            mode="markers",
            marker=dict(size=2.4, color=pc1, colorscale="Spectral_r", opacity=0.55,
                        colorbar=dict(title="expression PC1"), showscale=True),
            hovertemplate="PC1=%{marker.color:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z (depth)",
            aspectmode="manual",
            aspectratio=dict(x=dx / m, y=dy / m, z=zr / m),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--gif", action="store_true",
                        help="also write a rotating GIF (slower)")
    parser.add_argument("--gif-frames", type=int, default=120,
                        help="frames per 360 revolution (more = smoother & slower)")
    parser.add_argument("--gif-fps", type=int, default=10,
                        help="GIF playback fps (lower = slower rotation)")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--split", action="store_true",
                        help="also write the measured-vs-imputed split figures "
                             "(<tag>_3d_split.png + <tag>_3d_split_interactive.html)")
    parser.add_argument("--no-jitter", action="store_true",
                        help="keep the regular sampling lattice (shows sampling rows)")
    parser.add_argument("--support_q", type=float, default=0.4,
                        help="tissue mask: keep grid points above this quantile of "
                             "observed-spot field support (lower = bigger blob)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample = args.sample or cfg["dataset"]["samples"][0]
    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(outdir, exist_ok=True)
    tag = f"{cfg['dataset']['name']}_{sample}"

    scene, grid, pc1, dims, z_exag, _ = reconstruct_volume(
        cfg, sample, support_q=args.support_q, jitter=not args.no_jitter)
    n_obs = int(scene.n_spots)
    title = (f"Reconstructed 3D ST volume — {tag} "
             f"({grid.shape[0]:,} field samples, z exaggerated x{z_exag:.1f})")

    if not args.no_html:
        html = os.path.join(outdir, f"{tag}_3d_interactive.html")
        fig_interactive(grid, pc1, dims, z_exag, html, title)
        print(f"wrote {tag}_3d_interactive.html  ({grid.shape[0]:,} field samples) "
              f"-- open in a browser, drag to rotate")

    if not args.no_png:
        fig_volume(grid, pc1, dims, z_exag, os.path.join(outdir, f"{tag}_3d_volume.png"),
                   n_obs, grid.shape[0])
        print(f"wrote {tag}_3d_volume.png")
    if args.gif:
        fig_rotation(grid, pc1, dims, z_exag, os.path.join(outdir, f"{tag}_3d_rotation.gif"),
                     n_frames=args.gif_frames, fps=args.gif_fps)
        rev = args.gif_frames / args.gif_fps
        print(f"wrote {tag}_3d_rotation.gif  (~{rev:.1f}s per revolution)")

    if args.split:
        obs, imp, sdims, sz_exag, _ = reconstruct_split(
            cfg, sample, support_q=args.support_q)
        spng = os.path.join(outdir, f"{tag}_3d_split.png")
        fig_split_volume(obs, imp, sdims, sz_exag, spng, sample)
        print(f"wrote {tag}_3d_split.png  "
              f"({obs['xyz'].shape[0]:,} measured + {imp['xyz'].shape[0]:,} imputed)")
        if not args.no_html:
            shtml = os.path.join(outdir, f"{tag}_3d_split_interactive.html")
            fig_split_interactive(
                obs, imp, sdims, sz_exag, shtml,
                f"{tag}: measured sections vs imputed field (toggle in legend)")
            print(f"wrote {tag}_3d_split_interactive.html")
    print(f"figures in {outdir}")


if __name__ == "__main__":
    main()
