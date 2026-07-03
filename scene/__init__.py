from .dataset import STScene
from .gaussian_model import GaussianModel

# `scripts/train.py` imports `Scene` from the base-repo layout.
Scene = STScene

__all__ = ["Scene", "STScene", "GaussianModel"]
