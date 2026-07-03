"""STScene: one serial-section sample as a normalized 3D gene-expression point cloud."""
import numpy as np
import torch

from utils.data_utils import (
    load_sample_npy,
    normalize_coords,
    parity_split,
    sample_npy_path,
)
from utils.general_utils import get_device


class STScene:
    """Holds one sample's spots (xyz, gex, features) with a train/held-out split.

    The observed (train) sections form the SfM-analog point cloud used to
    initialize the Gaussians; the held-out sections are the interleaved sections
    whose gene expression we predict by querying the learned field.
    """

    def __init__(self, sample, root, name, held_out_parity="even",
                 train_levels=None, heldout_levels=None, z_spacing=1.0,
                 cube=1.0, device="cpu"):
        self.sample = sample
        self.name = name
        self.device = get_device(device) if isinstance(device, str) else device

        raw = load_sample_npy(sample_npy_path(root, name, sample))
        self.G = raw["gex"].shape[1]

        xyz_np, self.norm_meta = normalize_coords(
            raw["xy"], raw["level"], z_spacing=z_spacing, cube=cube
        )
        train_mask, heldout_mask = parity_split(
            raw["level"], held_out_parity, train_levels, heldout_levels
        )

        dev = self.device
        self.xyz = torch.from_numpy(xyz_np).to(dev)                       # (N,3) f32
        self.gex = torch.from_numpy(raw["gex"]).to(dev)                   # (N,G) f32
        self.level = torch.from_numpy(raw["level"]).to(dev)              # (N,)  i64
        self.features = torch.from_numpy(raw["features"]).to(dev)        # (N,F) f32 (unused)
        self.xy = raw["xy"]                                              # (N,2) int (raw grid)
        self.section_id = raw["section_id"]
        self.train_mask = torch.from_numpy(train_mask).to(dev)
        self.heldout_mask = torch.from_numpy(heldout_mask).to(dev)

    @property
    def n_spots(self):
        return self.xyz.shape[0]

    @property
    def scene_extent(self):
        """Largest in-plane span of the train cloud (used to scale densification)."""
        pts = self.xyz[self.train_mask]
        if pts.numel() == 0:
            return 1.0
        return float((pts.max(dim=0).values - pts.min(dim=0).values).max().item())

    def train_points(self):
        m = self.train_mask
        return self.xyz[m], self.gex[m]

    def heldout_points(self):
        m = self.heldout_mask
        return self.xyz[m], self.gex[m], self.level[m]

    def init_pcd(self):
        """SfM-analog cloud: train-section means + their measured gex as features."""
        return self.train_points()
