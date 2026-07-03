"""Naive baselines the learned field must beat (no Gaussians, just copy/interpolate).

These operate directly on the scene's raw grid coordinates (x, y, level), matching
held-out spots to train spots at the SAME (x, y) grid location.
"""
import numpy as np


def _scene_arrays(scene):
    xy = np.asarray(scene.xy)                       # (N,2) raw grid
    level = scene.level.cpu().numpy()
    gex = scene.gex.cpu().numpy()
    train = scene.train_mask.cpu().numpy()
    heldout = scene.heldout_mask.cpu().numpy()
    return xy, level, gex, train, heldout


def _index_by_xy_per_level(xy, level, mask):
    """Map level -> {(x,y): row index} for the masked spots."""
    idx = {}
    for i in np.where(mask)[0]:
        lv = int(level[i])
        idx.setdefault(lv, {})[(int(xy[i, 0]), int(xy[i, 1]))] = i
    return idx


def _nearest_spatial(xy_target, candidates_xy, candidates_idx):
    d = np.sum((candidates_xy - xy_target) ** 2, axis=1)
    return candidates_idx[int(np.argmin(d))]


def nearest_section_copy(scene):
    """For each held-out spot, copy gex of the same (x,y) spot in the nearest
    (in z) train section; fall back to nearest spatial neighbour there."""
    xy, level, gex, train, heldout = _scene_arrays(scene)
    train_levels = sorted(set(int(l) for l in level[train]))
    by_lv = _index_by_xy_per_level(xy, level, train)
    train_xy = {lv: np.array([k for k in by_lv[lv]]) for lv in by_lv}
    train_ix = {lv: np.array([by_lv[lv][k] for k in by_lv[lv]]) for lv in by_lv}

    ho_rows = np.where(heldout)[0]
    pred = np.zeros((len(ho_rows), gex.shape[1]), dtype=np.float32)
    for r, i in enumerate(ho_rows):
        lv = int(level[i])
        key = (int(xy[i, 0]), int(xy[i, 1]))
        for tl in sorted(train_levels, key=lambda x: abs(x - lv)):
            if key in by_lv[tl]:
                pred[r] = gex[by_lv[tl][key]]
                break
        else:
            tl = min(train_levels, key=lambda x: abs(x - lv))
            j = _nearest_spatial(xy[i], train_xy[tl], train_ix[tl])
            pred[r] = gex[j]
    return pred


def linear_interp_baseline(scene):
    """Average the same-(x,y) spot from the two bracketing train sections.
    Falls back to nearest_section_copy where a bracket is missing."""
    xy, level, gex, train, heldout = _scene_arrays(scene)
    train_levels = sorted(set(int(l) for l in level[train]))
    by_lv = _index_by_xy_per_level(xy, level, train)
    copy = nearest_section_copy(scene)

    ho_rows = np.where(heldout)[0]
    pred = copy.copy()
    for r, i in enumerate(ho_rows):
        lv = int(level[i])
        key = (int(xy[i, 0]), int(xy[i, 1]))
        lower = [tl for tl in train_levels if tl < lv]
        upper = [tl for tl in train_levels if tl > lv]
        if not lower or not upper:
            continue
        lo, hi = max(lower), min(upper)
        if key in by_lv[lo] and key in by_lv[hi]:
            w = (hi - lv) / (hi - lo)
            pred[r] = w * gex[by_lv[lo][key]] + (1 - w) * gex[by_lv[hi][key]]
    return pred
