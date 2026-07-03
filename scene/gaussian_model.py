"""GaussianModel: a 3D Gaussian field whose color attribute is a gene-expression vector.

Mirrors the graphdeco-inria base-repo GaussianModel API, but spherical-harmonic color
is replaced by a raw G-dimensional gene-expression feature (targets are signed
log-normalized values, so the feature carries NO activation).
"""
import numpy as np
import torch
import torch.nn.functional as F

from utils.general_utils import build_covariance, inverse_sigmoid


class GaussianModel:
    def __init__(self, G=250, device="cpu", densify_z=False):
        self.G = G
        self.device = torch.device(device) if isinstance(device, str) else device
        self.densify_z = densify_z
        self.percent_dense = 0.01  # size threshold (fraction of scene extent) for clone vs split

        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._features = torch.empty(0)

        self.optimizer = None
        self.xyz_grad_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii = torch.empty(0)

    # ---- activations -------------------------------------------------------
    @property
    def scaling(self):
        return torch.exp(self._scaling)

    @property
    def rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def features(self):
        return self._features

    @property
    def xyz(self):
        return self._xyz

    @property
    def n_points(self):
        return self._xyz.shape[0]

    def build_covariance(self):
        """Return (Sigma, Sigma_inv), each (N, 3, 3)."""
        return build_covariance(self.scaling, self.rotation)

    # ---- initialization ----------------------------------------------------
    def create_from_pcd(self, means, feats, init_opacity=0.1, scale_knn=3,
                        init_scale_mult=1.0):
        means = means.to(self.device).float()
        feats = feats.to(self.device).float()
        N = means.shape[0]

        # per-point scale from mean distance to `scale_knn` nearest neighbours,
        # optionally inflated by `init_scale_mult` so adjacent z-planes overlap.
        with torch.no_grad():
            d2 = torch.cdist(means, means)                       # (N,N)
            d2.fill_diagonal_(float("inf"))
            k = min(scale_knn, max(N - 1, 1))
            knn_d = d2.topk(k, largest=False).values             # (N,k)
            mean_d = knn_d.clamp_min(1e-8).mean(dim=1) * float(init_scale_mult)  # (N,)
        log_scale = torch.log(mean_d).clamp_min(np.log(1e-6))
        scaling = log_scale[:, None].repeat(1, 3)                # (N,3)

        rotation = torch.zeros(N, 4, device=self.device)
        rotation[:, 0] = 1.0                                     # identity quaternion
        opacity = inverse_sigmoid(
            torch.full((N, 1), float(init_opacity), device=self.device)
        )

        self._xyz = torch.nn.Parameter(means.contiguous().requires_grad_(True))
        self._scaling = torch.nn.Parameter(scaling.contiguous().requires_grad_(True))
        self._rotation = torch.nn.Parameter(rotation.contiguous().requires_grad_(True))
        self._opacity = torch.nn.Parameter(opacity.contiguous().requires_grad_(True))
        self._features = torch.nn.Parameter(feats.contiguous().requires_grad_(True))
        self._reset_stats()

    def _reset_stats(self):
        N = self.n_points
        self.xyz_grad_accum = torch.zeros(N, 1, device=self.device)
        self.denom = torch.zeros(N, 1, device=self.device)
        self.max_radii = torch.zeros(N, device=self.device)

    # ---- optimizer ---------------------------------------------------------
    def training_setup(self, cfg):
        o = cfg["optim"] if "optim" in cfg else cfg
        params = [
            {"params": [self._xyz], "lr": o["lr_xyz"], "name": "xyz"},
            {"params": [self._features], "lr": o["lr_feature"], "name": "features"},
            {"params": [self._opacity], "lr": o["lr_opacity"], "name": "opacity"},
            {"params": [self._scaling], "lr": o["lr_scaling"], "name": "scaling"},
            {"params": [self._rotation], "lr": o["lr_rotation"], "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

    # ---- optimizer tensor surgery (keeps Adam state aligned) ---------------
    def _replace_tensor_in_optimizer(self, tensor, name):
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            stored = self.optimizer.state.get(group["params"][0], None)
            new_p = torch.nn.Parameter(tensor.contiguous().requires_grad_(True))
            if stored is not None:
                stored["exp_avg"] = torch.zeros_like(tensor)
                stored["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                self.optimizer.state[new_p] = stored
            group["params"][0] = new_p
            return new_p

    def _cat_to_optimizer(self, tensors_dict):
        out = {}
        for group in self.optimizer.param_groups:
            ext = tensors_dict[group["name"]]
            stored = self.optimizer.state.get(group["params"][0], None)
            new_p = torch.nn.Parameter(
                torch.cat([group["params"][0], ext], dim=0).requires_grad_(True))
            if stored is not None:
                stored["exp_avg"] = torch.cat(
                    [stored["exp_avg"], torch.zeros_like(ext)], dim=0)
                stored["exp_avg_sq"] = torch.cat(
                    [stored["exp_avg_sq"], torch.zeros_like(ext)], dim=0)
                del self.optimizer.state[group["params"][0]]
                self.optimizer.state[new_p] = stored
            group["params"][0] = new_p
            out[group["name"]] = new_p
        return out

    def _prune_optimizer(self, mask):
        out = {}
        for group in self.optimizer.param_groups:
            stored = self.optimizer.state.get(group["params"][0], None)
            new_p = torch.nn.Parameter(group["params"][0][mask].requires_grad_(True))
            if stored is not None:
                stored["exp_avg"] = stored["exp_avg"][mask]
                stored["exp_avg_sq"] = stored["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                self.optimizer.state[new_p] = stored
            group["params"][0] = new_p
            out[group["name"]] = new_p
        return out

    def _assign_from_groups(self, groups):
        self._xyz = groups["xyz"]
        self._features = groups["features"]
        self._opacity = groups["opacity"]
        self._scaling = groups["scaling"]
        self._rotation = groups["rotation"]

    # ---- densification stats ----------------------------------------------
    def add_densification_stats(self, visible_mask=None):
        """Accumulate per-point xyz gradient norm (call after backward, pre-step)."""
        if self._xyz.grad is None:
            return
        grad_norm = torch.norm(self._xyz.grad, dim=-1, keepdim=True)   # (N,1)
        mask = (torch.ones(self.n_points, dtype=torch.bool, device=self.device)
                if visible_mask is None else visible_mask)
        self.xyz_grad_accum[mask] += grad_norm[mask]
        self.denom[mask] += 1

    # ---- adaptive density control -----------------------------------------
    def _densify_and_clone(self, grads, grad_thresh, scene_extent):
        sel = grads.squeeze(-1) >= grad_thresh
        sel &= self.scaling.max(dim=1).values <= self.percent_dense * scene_extent
        if sel.sum() == 0:
            return
        new = {
            "xyz": self._xyz[sel],
            "features": self._features[sel],
            "opacity": self._opacity[sel],
            "scaling": self._scaling[sel],
            "rotation": self._rotation[sel],
        }
        groups = self._cat_to_optimizer(new)
        self._assign_from_groups(groups)
        self._grow_stats(int(sel.sum()))

    def _densify_and_split(self, grads, grad_thresh, scene_extent, N=2):
        n0 = self.n_points
        padded = torch.zeros(n0, device=self.device)
        padded[: grads.shape[0]] = grads.squeeze(-1)
        sel = padded >= grad_thresh
        sel &= self.scaling.max(dim=1).values > self.percent_dense * scene_extent
        if sel.sum() == 0:
            return

        stds = self.scaling[sel].repeat(N, 1)                    # (Ns*N,3)
        if not self.densify_z:
            stds = stds.clone()
            stds[:, 2] = 0.0                                     # no through-plane growth
        noise = torch.randn_like(stds) * stds
        new_xyz = self._xyz[sel].repeat(N, 1) + noise
        new_scaling = torch.log(self.scaling[sel].repeat(N, 1) / (0.8 * N))
        new = {
            "xyz": new_xyz,
            "features": self._features[sel].repeat(N, 1),
            "opacity": self._opacity[sel].repeat(N, 1),
            "scaling": new_scaling,
            "rotation": self._rotation[sel].repeat(N, 1),
        }
        groups = self._cat_to_optimizer(new)
        self._assign_from_groups(groups)
        self._grow_stats(int(sel.sum()) * N)

        # prune the original split points
        prune = torch.cat([
            sel,
            torch.zeros(int(sel.sum()) * N, dtype=torch.bool, device=self.device),
        ])
        self._prune_points(prune)

    def _grow_stats(self, n_new):
        self.xyz_grad_accum = torch.cat(
            [self.xyz_grad_accum, torch.zeros(n_new, 1, device=self.device)])
        self.denom = torch.cat([self.denom, torch.zeros(n_new, 1, device=self.device)])
        self.max_radii = torch.cat([self.max_radii, torch.zeros(n_new, device=self.device)])

    def _prune_points(self, prune_mask):
        keep = ~prune_mask
        groups = self._prune_optimizer(keep)
        self._assign_from_groups(groups)
        self.xyz_grad_accum = self.xyz_grad_accum[keep]
        self.denom = self.denom[keep]
        self.max_radii = self.max_radii[keep]

    def densify_and_prune(self, grad_thresh, min_opacity, scene_extent,
                          max_gaussians=None, max_scale_factor=None):
        grads = self.xyz_grad_accum / self.denom.clamp_min(1)
        grads[grads.isnan()] = 0.0

        if max_gaussians is None or self.n_points < max_gaussians:
            self._densify_and_clone(grads, grad_thresh, scene_extent)
            self._densify_and_split(grads, grad_thresh, scene_extent)

        prune = (self.opacity < min_opacity).squeeze(-1)
        if max_scale_factor is not None:
            big = self.scaling.max(dim=1).values > max_scale_factor * scene_extent
            prune = prune | big
        if prune.sum() > 0:
            self._prune_points(prune)

        self._reset_accum()

    def _reset_accum(self):
        self.xyz_grad_accum = torch.zeros(self.n_points, 1, device=self.device)
        self.denom = torch.zeros(self.n_points, 1, device=self.device)

    def reset_opacity(self):
        new_op = inverse_sigmoid(
            torch.minimum(self.opacity, torch.full_like(self.opacity, 0.1)))
        self._opacity = self._replace_tensor_in_optimizer(new_op, "opacity")

    # ---- checkpoint --------------------------------------------------------
    def capture(self):
        return {
            "G": self.G,
            "densify_z": self.densify_z,
            "_xyz": self._xyz.detach().cpu(),
            "_scaling": self._scaling.detach().cpu(),
            "_rotation": self._rotation.detach().cpu(),
            "_opacity": self._opacity.detach().cpu(),
            "_features": self._features.detach().cpu(),
        }

    def restore(self, state):
        self.G = state["G"]
        self.densify_z = state.get("densify_z", False)
        self._xyz = torch.nn.Parameter(state["_xyz"].to(self.device).requires_grad_(True))
        self._scaling = torch.nn.Parameter(state["_scaling"].to(self.device).requires_grad_(True))
        self._rotation = torch.nn.Parameter(state["_rotation"].to(self.device).requires_grad_(True))
        self._opacity = torch.nn.Parameter(state["_opacity"].to(self.device).requires_grad_(True))
        self._features = torch.nn.Parameter(state["_features"].to(self.device).requires_grad_(True))
        self._reset_stats()
