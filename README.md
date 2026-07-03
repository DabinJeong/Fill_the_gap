# Fill the gap: 3D Gaussian Splatting for serial-section spatial-transcriptome reconstruction

Predict the gene-expression (gex) profile of **interleaved held-out serial ST sections**
by modeling spatial gene expression as a continuous 3D field `f: R³ → R^G` represented by
3D Gaussians (after _3D reconstruction of spatial transcriptome_, D. Jeong, Jan 2026;
base framework: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)).

Observed sections are samples of `f` at discrete depths `z_i` (`X^(i)_j ≈ f(x,y,z_i)`); a
section at any depth is produced by a rendering operator `X̂(x,y;z)=R(f,z)`, so interpolation
of the missing sections emerges by querying the learned field at unseen depths. **H&E is not
used in this version** (deferred per the proposal).

## Representation & rendering

- **Gaussians** (`scene/gaussian_model.py`): mean `μ∈R³`, covariance `Σ=RSSᵀRᵀ` (scale +
  rotation quaternion), opacity, and a **G-dim gene-expression feature** replacing SH color
  (raw, no activation — targets are signed log-normalized values).
- **Rendering operator R** (`renderer/point_query.py`, primary): point-query field evaluation,
  `f(p)=Σ_i w_i g_i / Σ_i w_i`, `w_i = α_i·exp(-½(p-μ_i)ᵀΣ_i⁻¹(p-μ_i))` over the k nearest
  Gaussians. Pure PyTorch, differentiable, no CUDA. (`renderer/ewa_slice.py` documents the
  deferred EWA splatting alternative.)
- Gaussians are initialized from the observed-section point cloud: means = train-spot xyz,
  features = measured train gex (`scene/dataset.py: STScene.init_pcd`).

## Data

ASIGN serial-section datasets at
`/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_serial_sections`:

- `HER2_3D/` — patients A–H, 6 sections each (levels 1–6), ~2000 spots, G=250.
- `ST_Breast_3D/` — 16 stacks, 3 sections each (levels 1–3), ~1700 spots, G=250.

Each `3D_npy_information/<sample>.npy` is an object array of per-spot dicts
(`position`→[x,y], `level`→z, `label`→(250,) gex target, `feature_1024`→histology feature
(carried, unused)).

**Held-out scheme:** HER2 train {1,3,5} → predict {2,4,6}; ST_Breast train {1,3} → predict {2}.

## Usage

```bash
PY=/software/teamtrynka/ks37/envs/torch/bin/python
export PYTHONPATH=/nfs/team361/dj16/projects/HistoGS
$PY scripts/train.py   --config configs/her2_A.yaml   # fit field on observed sections
$PY scripts/render.py  --config configs/her2_A.yaml   # predict held-out sections
$PY evals/run_eval.py  --config configs/her2_A.yaml   # metrics vs measured + baselines
```

Config knobs: `dataset.{z_spacing,train_levels,heldout_levels}`, `render.knn_k`,
`model.{init_scale_mult,init_opacity}`, `optim.lr_*`, `densify.*`, `loss.w_*`.

## Key finding (HER2 patient A, {1,3,5}→{2,4,6})

The field initialized with measured gex as features is already a strong smooth 3D
interpolator and **beats the naive baselines**; aggressively optimizing the per-Gaussian
features _overfits the observed planes and hurts held-out prediction_. Best regime: keep
features = measured gex, use many neighbours (k≈32) and mild scale inflation.

| method               | per-gene r (median) | spot r (mean) | MSE       |
| -------------------- | ------------------- | ------------- | --------- |
| nearest-section copy | 0.126               | 0.684         | 0.511     |
| linear interpolation | 0.140               | 0.709         | 0.463     |
| **HistoGS field**    | **0.216**           | **0.804**     | **0.302** |

## Layout

```
utils/   general_utils (quat/cov/seed), config (yaml), data_utils (npy load, normalize, split)
scene/   dataset.py (STScene), gaussian_model.py (GaussianModel)
renderer/point_query.py (R), ewa_slice.py (deferred)
loss/    loss.py (MSE + optional cosine/opacity/scale regularizers)
scripts/ train.py, render.py
evals/   metrics.py, baselines.py (copy / linear-interp), run_eval.py
configs/ her2_A.yaml
```

## Status / next

- v1 pipeline complete and verified end-to-end on HER2 patient A.
- TODO: sweep all HER2 patients (+ symmetric split) and ST_Breast stacks; tune
  `z_spacing`/`knn_k`/`init_scale_mult`; compare against ASIGN / HoloTea / UniST / VorTex;
  optional per-section rigid **registration** (`utils/data_utils.register_coords`, currently
  a no-op) — the biggest modeling assumption; future H&E conditioning ("splat on H&E").

```

```
