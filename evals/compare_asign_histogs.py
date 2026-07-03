"""Assemble the final ASIGN-vs-HistoGS comparison table (HER2, identical protocol).

Reads:
  - asign_repro/weight/ours/asign_her2_summary.json   (ASIGN reproduced, 4-fold CV)
  - runs/her2_A/histogs_asign_protocol.json           (HistoGS, same protocol)
Prints a markdown table and the ASIGN paper reference numbers.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASIGN_JSON = os.path.join(os.path.dirname(ROOT), "HistoGS_dependencies", "ASIGN",
                          "weight", "ours", "asign_her2_summary.json")
HISTOGS_JSON = os.path.join(ROOT, "runs", "her2_A", "histogs_asign_protocol.json")

# ASIGN paper, Table 1, 3D Sample-wise, HER2 (from the histogs-asign-benchmark memory).
PAPER = {"pcc": 0.693, "mse": 0.363, "mae": 0.473}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    asign = load(ASIGN_JSON)
    histogs = load(HISTOGS_JSON)

    rows = []
    rows.append(("ASIGN (paper Table 1, 3D sample-wise)", PAPER["pcc"], PAPER["mse"], PAPER["mae"]))
    if asign:
        rows.append(("ASIGN (reproduced, this run)",
                     asign["mean_pcc"], asign["mean_mse"], asign["mean_mae"]))
    else:
        rows.append(("ASIGN (reproduced, this run)", None, None, None))
    if histogs:
        h = histogs["histogs_mean"]
        c = histogs["copy_top_mean"]
        rows.append(("HistoGS (init field, ASIGN protocol)", h["pcc"], h["mse"], h["mae"]))
        rows.append(("copy-top-section baseline", c["pcc"], c["mse"], c["mae"]))

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "

    print("\n## HER2 3D imputation -- level-1 known, predict level>1 (metrics on level>1)\n")
    print("| Method | PCC up | MSE down | MAE down |")
    print("|---|---|---|---|")
    for name, pcc, mse, mae in rows:
        print(f"| {name} | {fmt(pcc)} | {fmt(mse)} | {fmt(mae)} |")
    print()
    if asign:
        print(f"ASIGN reproduced over n={asign['n_test_samples']} test-sample evaluations "
              f"(4-fold sample-wise CV: {{A,E}},{{B,F}},{{C,G}},{{D,H}}).")
    if histogs:
        print(f"HistoGS evaluated on samples {histogs['samples']} "
              f"(per-sample init-field interpolation, optim.iterations=0).")


if __name__ == "__main__":
    main()
