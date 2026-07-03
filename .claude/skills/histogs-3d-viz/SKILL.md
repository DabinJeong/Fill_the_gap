---
name: histogs-3d-viz
description: >
  Visualize the 3D object reconstructed by the HistoGS Gaussian field — renders the
  trained continuous gene-expression field as a 3D volume and writes an interactive
  HTML (mouse-rotatable), a static multi-angle PNG, and a slowly rotating GIF. Use
  whenever the user asks to visualize HistoGS results, the reconstructed 3D volume /
  ST atlas, a trained Gaussian field, or asks for a 3D / interactive / rotating figure
  of a serial-section sample.
---

# HistoGS 3D reconstruction visualization

Renders the trained Gaussian field (a continuous `f: R³ → R²⁵⁰`) as a 3D volume by
sampling it on a dense grid (observed sections + interpolated depths), masking to
tissue support, and colouring by expression PC1. One script produces all three
formats.

## Prerequisites
- A trained checkpoint must exist: `runs/<run>/<sample>/chkpnt.pth`. If missing, train
  first: `.venv/bin/python scripts/train.py --config <config>`.
- Python env: the project uv env `/nfs/team361/dj16/projects/HistoGS/.venv` (has plotly,
  matplotlib, torch_geometric — see [[histogs-torch-env]]). Rebuild with `uv sync` if missing.

## Run (from the project root `/nfs/team361/dj16/projects/HistoGS`)
```bash
export PYTHONPATH=/nfs/team361/dj16/projects/HistoGS
PY=/nfs/team361/dj16/projects/HistoGS/.venv/bin/python

# default: interactive HTML + multi-angle PNG
$PY scripts/visualize_3d.py --config configs/her2_A.yaml

# measured-vs-imputed split figures (observed sections vs interpolated field)
$PY scripts/visualize_3d.py --config configs/her2_A.yaml --split

# all three formats incl. a slow rotating GIF (~22.5 s per revolution)
$PY scripts/visualize_3d.py --config configs/her2_A.yaml \
    --gif --gif-frames 180 --gif-fps 8
```
Long-running (the GIF renders 100+ 3D frames) — launch with `run_in_background: true`
and watch the task output for the `per revolution` / `figures in` completion line.

## Key flags
- `--sample <id>`   : sample to render (default = first in the config). E.g. `B`, or an
  ST_Breast stack id; set the matching `dataset.name`/levels in the config.
- `--split`         : also write measured-vs-imputed figures — `<tag>_3d_split.png`
  (3 panels: measured observed sections | imputed/interpolated field | overlay) and
  `<tag>_3d_split_interactive.html` (two legend-toggleable traces). Measured spots are
  the ground-truth gex on the known sections; the imputed cloud is the field sampled at
  depths off the observed planes. Both share one PC1 colour basis.
- `--gif`           : also write the rotating GIF (off by default; it is the slow step).
- `--gif-frames N`  : frames per 360° revolution — **higher = smoother AND slower**.
- `--gif-fps F`     : playback fps — **lower = slower rotation**. seconds/revolution = frames/fps.
- `--no-html`, `--no-png` : skip those outputs.
- `--support_q Q`   : tissue mask; keep grid points above the Q-quantile of observed-spot
  field support (lower Q = larger/looser blob). Default 0.4.
- `--outdir DIR`    : output directory (default `<project>/figures/`).

## Outputs (in `figures/`, named `<dataset>_<sample>_3d_*`)
- `*_3d_interactive.html` — self-contained (plotly.js embedded, works offline); open in a
  browser, **drag to rotate, scroll to zoom, right-drag to pan**.
- `*_3d_volume.png` — static reconstructed volume from 4 viewing angles.
- `*_3d_split.png` / `*_3d_split_interactive.html` — measured vs imputed (only with `--split`).
- `*_3d_rotation.gif` — slowly rotating volume (only with `--gif`).

## Related: section-distance analysis (`scripts/section_distance_analysis.py`)
Answers "how far apart can serial sections be and still support 3D interpolation?".
Aggregates over HER2 A..H and writes `figures/<dataset>_section_distance.png` with two
panels: **A** the model-free data ceiling (measured↔measured ASIGN-PCC vs section gap —
how fast the tissue decorrelates with depth) and **B** HistoGS interpolation/reach PCC vs
distance to the nearest known section, plus a copy-nearest baseline.
```bash
.venv/bin/python scripts/section_distance_analysis.py --config configs/her2_A.yaml
```
For HER2 the tissue decorrelates slowly with depth (PCC ~0.61→0.56 over 5 sections), so
HistoGS interpolation degrades gracefully (~0.74→0.68) and stays well above copy — i.e.
sections can be several spacings apart and still be imputable within this dataset's span.

Note: z (depth) is exaggerated for display because serial sections form a thin slab; the
exaggeration factor is printed in the title.

## Tips
- To make the GIF slower, increase `--gif-frames` and/or decrease `--gif-fps`.
- If the HTML "won't open" (e.g. running on the farm), produce the GIF/PNG instead, or
  copy the HTML locally: `scp farm:.../figures/<file>.html .`
- Implementation: `scripts/visualize_3d.py` (`reconstruct_volume` samples the field;
  `fig_interactive`/`fig_volume`/`fig_rotation` render each format).
