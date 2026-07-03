"""Preprocess raw DLPFC Visium (Maynard 2021) into the HistoGS/ASIGN npy format.

Each of 3 donors has 4 sections; section IDs in order map to physical depth
z = {0, 10, 310, 320} µm (two 10-µm-adjacent pairs, 300 µm between pairs).

Input (downloaded to <scratch>/dlpfc): <id>.h5 (10x filtered matrix) + <id>_pos.txt
(tissue_positions_list: barcode,in_tissue,array_row,array_col,pxl_row,pxl_col).

Output: <out>/DLPFC_3D/3D_npy_information/<donor>.npy — list of per-spot dicts with keys
matching load_sample_npy: img_path, label (250 HVG log-norm gex), position [x,y] =
[array_col, array_row], level (1..4), feature_1024 (zeros, unused by HistoGS).
A common 250-HVG gene set is selected once on the pooled 12 sections so all donors and
sections share the same genes (comparable PCC).
"""
import os
import numpy as np
import h5py
from scipy.sparse import csc_matrix

SCRATCH = "/tmp/claude-22843/-nfs-team361-dj16-projects-HistoGS/9c451182-f61c-461b-b2a9-0f2fa7c02702/scratchpad/dlpfc"
OUT = "/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_serial_sections/DLPFC_3D/3D_npy_information"
DONORS = {
    "donor1": ["151507", "151508", "151509", "151510"],
    "donor2": ["151669", "151670", "151671", "151672"],
    "donor3": ["151673", "151674", "151675", "151676"],
}
Z_UM = [0.0, 10.0, 310.0, 320.0]   # physical depth of section 1..4 within a donor
N_HVG = 250


def read_section(sid):
    """Return (spots x genes dense-ish counts as float32 ndarray, gene_ids, xy (N,2))."""
    with h5py.File(os.path.join(SCRATCH, f"{sid}.h5"), "r") as f:
        m = f["matrix"]
        data = m["data"][:]; indices = m["indices"][:]; indptr = m["indptr"][:]
        n_genes, n_spots = m["shape"][:]
        gene_ids = np.array([x.decode() for x in m["features"]["id"][:]])
        barcodes = np.array([x.decode() for x in m["barcodes"][:]])
    # 10x h5 is CSC over spots (columns), genes are rows
    X = csc_matrix((data, indices, indptr), shape=(n_genes, n_spots)).T.toarray().astype(np.float32)

    pos = {}
    with open(os.path.join(SCRATCH, f"{sid}_pos.txt")) as fh:
        for line in fh:
            p = line.strip().split(",")
            # barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
            pos[p[0]] = (int(p[1]), int(p[2]), int(p[3]))
    xy = np.array([[pos[b][2], pos[b][1]] for b in barcodes], dtype=np.int64)  # [array_col, array_row]
    in_tissue = np.array([pos[b][0] for b in barcodes], dtype=bool)
    return X[in_tissue], gene_ids, xy[in_tissue]


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---- load all sections, normalize, pool for HVG selection ----
    sections = {}  # sid -> (lognorm (N,G), xy)
    ref_genes = None
    pooled_sum = None; pooled_sumsq = None; pooled_n = 0
    for sid in [s for ids in DONORS.values() for s in ids]:
        X, gids, xy = read_section(sid)
        if ref_genes is None:
            ref_genes = gids
        assert np.array_equal(gids, ref_genes), f"gene order mismatch in {sid}"
        # CPM(1e4) + log1p per spot
        tot = X.sum(1, keepdims=True); tot[tot == 0] = 1
        ln = np.log1p(X / tot * 1e4).astype(np.float32)
        sections[sid] = (ln, xy)
        s = ln.sum(0); ss = (ln ** 2).sum(0)
        pooled_sum = s if pooled_sum is None else pooled_sum + s
        pooled_sumsq = ss if pooled_sumsq is None else pooled_sumsq + ss
        pooled_n += ln.shape[0]
        print(f"  {sid}: {ln.shape[0]} spots, {X.shape[1]} genes")

    var = pooled_sumsq / pooled_n - (pooled_sum / pooled_n) ** 2
    hvg = np.argsort(var)[::-1][:N_HVG]
    hvg = np.sort(hvg)
    print(f"selected {len(hvg)} HVGs (pooled variance); genes e.g. {ref_genes[hvg[:5]]}")

    # ---- write one npy per donor (4 sections, level 1..4) ----
    for donor, ids in DONORS.items():
        items = []
        for lvl, sid in enumerate(ids, 1):
            ln, xy = sections[sid]
            g = ln[:, hvg]
            for i in range(g.shape[0]):
                items.append({
                    "img_path": f"dlpfc/{donor}/{sid}/{xy[i,0]}x{xy[i,1]}.png",
                    "label": g[i].astype(np.float32),
                    "position": [int(xy[i, 0]), int(xy[i, 1])],
                    "level": lvl,
                    "feature_1024": np.zeros(1024, dtype=np.float32),
                })
        out = os.path.join(OUT, f"{donor}.npy")
        np.save(out, np.array(items, dtype=object))
        print(f"wrote {out}  ({len(items)} spots, 4 sections z={Z_UM}µm)")


if __name__ == "__main__":
    main()
