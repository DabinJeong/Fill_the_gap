"""General-purpose helpers: quaternion math, activations, seeding, device."""
import random

import numpy as np
import torch


def get_device(spec: str = "cuda") -> torch.device:
    """Resolve a device spec, falling back to CPU when CUDA is unavailable."""
    if spec.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(spec)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """logit: inverse of sigmoid. Expects values in (0, 1)."""
    return torch.log(x / (1.0 - x))


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4) quaternions [w, x, y, z] to (..., 3, 3) rotation matrices.

    Quaternions are normalized internally, so raw (unnormalized) parameters are
    accepted.
    """
    q = torch.nn.functional.normalize(quat, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(quat.shape[:-1] + (3, 3))


def build_scaling_rotation(scaling: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Return M = R @ S where S = diag(scaling), so that covariance = M @ M^T.

    scaling: (..., 3) positive scales.
    rotation: (..., 4) quaternions.
    returns: (..., 3, 3).
    """
    R = quat_to_rotmat(rotation)               # (..., 3, 3)
    S = torch.diag_embed(scaling)              # (..., 3, 3)
    return R @ S


def build_covariance(scaling: torch.Tensor, rotation: torch.Tensor):
    """Return (Sigma, Sigma_inv) of shape (..., 3, 3) from scale + rotation.

    Sigma = M M^T with M = R S. Inverse computed in closed form via M^{-1}.
    """
    M = build_scaling_rotation(scaling, rotation)      # (..., 3, 3)
    Sigma = M @ M.transpose(-1, -2)
    # Sigma^{-1} = M^{-T} M^{-1}; M = R S so M^{-1} = S^{-1} R^T (cheap & stable).
    R = quat_to_rotmat(rotation)
    S_inv = torch.diag_embed(1.0 / scaling)
    M_inv = S_inv @ R.transpose(-1, -2)                # (R S)^{-1} = S^{-1} R^{-1}
    Sigma_inv = M_inv.transpose(-1, -2) @ M_inv
    return Sigma, Sigma_inv
