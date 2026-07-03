"""Config loading: YAML + dotted attribute access + defaults merge."""
import copy

import yaml

DEFAULTS = {
    "dataset": {
        "root": "/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_serial_sections",
        "name": "HER2_3D",
        "samples": ["A"],
        "held_out_parity": "even",
        "train_levels": None,
        "heldout_levels": None,
        "z_spacing": 1.0,
        "G": 250,
    },
    "model": {"init_opacity": 0.1, "scale_knn": 3, "init_scale_mult": 1.0, "densify_z": False},
    "render": {"knn_k": 16, "chunk": 8192},
    "optim": {
        "iterations": 7000,
        "lr_xyz": 1.0e-3,
        "lr_feature": 2.5e-3,
        "lr_opacity": 5.0e-2,
        "lr_scaling": 5.0e-3,
        "lr_rotation": 1.0e-3,
    },
    "densify": {
        "start": 500,
        "end": 5000,
        "interval": 100,
        "grad_thresh": 2.0e-4,
        "min_opacity": 0.005,
        "opacity_reset_interval": 3000,
        "max_gaussians": 200000,
    },
    "loss": {"w_cos": 0.0, "w_opacity": 0.0, "w_scale": 0.0},
    "output_dir": "/nfs/team361/dj16/projects/HistoGS/runs/default",
    "device": "cuda",
    "seed": 0,
    "log_interval": 200,
}


class Config(dict):
    """dict with attribute access; nested dicts are wrapped recursively."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(val) if isinstance(val, dict) else val

    def __setattr__(self, key, value):
        self[key] = value


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Config:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return Config(_deep_merge(DEFAULTS, user))
