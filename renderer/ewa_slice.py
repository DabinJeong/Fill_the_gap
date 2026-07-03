"""ALTERNATIVE rendering operator (deferred): EWA orthographic slice splatting.

The proposal cites EWA volume splatting [Zwicker et al. 2001]. Instead of querying
the field at discrete points, R(f, z0) would project every 3D Gaussian onto the
plane z = z0 and rasterize an N=G-channel feature image:

  - Condition / marginalize each 3D Gaussian onto the slice plane to obtain a 2D
    splat (mean, 2x2 covariance, weight) at z0.
  - Alpha-composite the G-channel features over a pixel/spot grid.

This needs a custom N-channel differentiable rasterizer (e.g. nerfstudio `gsplat`
with arbitrary feature dimension, or a modified `diff-gaussian-rasterization` whose
SH color is replaced by a G-dim feature). It is NOT implemented in v1; the
point-query operator in `point_query.py` is the primary R. This file is a
placeholder documenting the intended interface.
"""


def render_slice(*args, **kwargs):
    raise NotImplementedError(
        "EWA slice splatting is deferred to v2; use renderer.point_query.render."
    )
