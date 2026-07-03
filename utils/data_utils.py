"""Load ASIGN serial-section .npy samples and turn them into 3D point clouds."""
import os

import numpy as np


def _parse_section_id(img_path: str) -> str:
    """`cropped_imgs/patch_224/A4/10x11.png` -> 'A4' (HER2) or '1' (ST_Breast)."""
    parts = img_path.replace("\\", "/").split("/")
    # the section directory is the second-to-last component
    return parts[-2] if len(parts) >= 2 else img_path


def load_sample_npy(npy_path: str) -> dict:
    """Load one sample's per-spot dict array into stacked numpy arrays.

    Returns dict with:
        xy        (N, 2) int64
        level     (N,)   int64
        gex       (N, G) float32   <- target ('label')
        features  (N, F) float32   <- 'feature_1024', carried (unused in v1)
        img_path  list[str]  length N
        section_id(N,)   object    parsed from img_path
    """
    arr = np.load(npy_path, allow_pickle=True)
    xy = np.stack([np.asarray(d["position"], dtype=np.int64) for d in arr])
    level = np.array([int(d["level"]) for d in arr], dtype=np.int64)
    gex = np.stack([np.asarray(d["label"], dtype=np.float32) for d in arr])
    features = np.stack([np.asarray(d["feature_1024"], dtype=np.float32) for d in arr])
    img_path = [str(d["img_path"]) for d in arr]
    section_id = np.array([_parse_section_id(p) for p in img_path], dtype=object)
    return {
        "xy": xy,
        "level": level,
        "gex": gex,
        "features": features,
        "img_path": img_path,
        "section_id": section_id,
    }


def normalize_coords(xy: np.ndarray, level: np.ndarray, z_spacing: float, cube: float = 1.0):
    """Map (x, y, level) into a roughly cubic box of side `cube`.

    In-plane x, y share one isotropic scale; z = (level - level_min) * z_spacing is
    converted with the SAME scale, so `z_spacing` directly tunes through-plane vs
    in-plane anisotropy. Returns (xyz_norm (N,3) float32, norm_meta dict).
    """
    xy = xy.astype(np.float64)
    xy_min = xy.min(axis=0)
    xy_shift = xy - xy_min
    xy_range = xy_shift.max(axis=0)
    scale = cube / max(xy_range.max(), 1e-8)

    level_min = int(level.min())
    z = (level.astype(np.float64) - level_min) * float(z_spacing)

    xyz = np.empty((xy.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = (xy_shift[:, 0] * scale).astype(np.float32)
    xyz[:, 1] = (xy_shift[:, 1] * scale).astype(np.float32)
    xyz[:, 2] = (z * scale).astype(np.float32)

    norm_meta = {
        "xy_min": xy_min.tolist(),
        "level_min": level_min,
        "scale": float(scale),
        "z_spacing": float(z_spacing),
    }
    return xyz, norm_meta


def denormalize_coords(xyz: np.ndarray, norm_meta: dict):
    """Invert normalize_coords -> (xy float, level float)."""
    scale = norm_meta["scale"]
    xy = xyz[:, :2] / scale + np.asarray(norm_meta["xy_min"])
    level = xyz[:, 2] / (scale * norm_meta["z_spacing"]) + norm_meta["level_min"]
    return xy, level


def parity_split(level: np.ndarray, held_out_parity: str = "even",
                 train_levels=None, heldout_levels=None):
    """Return (train_mask, heldout_mask) boolean arrays over spots.

    Explicit train_levels / heldout_levels override the parity rule (needed for
    ST_Breast: train {1,3}, hold {2}). Otherwise 'even' holds out even levels.
    """
    if train_levels is not None or heldout_levels is not None:
        train_set = set(train_levels or [])
        heldout_set = set(heldout_levels or [])
        if not train_set:
            train_set = set(np.unique(level).tolist()) - heldout_set
        if not heldout_set:
            heldout_set = set(np.unique(level).tolist()) - train_set
    else:
        uniq = np.unique(level)
        even = set(int(l) for l in uniq if l % 2 == 0)
        odd = set(int(l) for l in uniq if l % 2 == 1)
        if held_out_parity == "even":
            heldout_set, train_set = even, odd
        elif held_out_parity == "odd":
            heldout_set, train_set = odd, even
        else:
            raise ValueError(f"held_out_parity must be 'even' or 'odd', got {held_out_parity}")

    train_mask = np.isin(level, list(train_set))
    heldout_mask = np.isin(level, list(heldout_set))
    return train_mask, heldout_mask


def register_coords(xy: np.ndarray, level: np.ndarray):
    """No-op hook for future per-section rigid alignment. Returns xy unchanged."""
    return xy


def sample_npy_path(root: str, name: str, sample: str) -> str:
    return os.path.join(root, name, "3D_npy_information", f"{sample}.npy")
