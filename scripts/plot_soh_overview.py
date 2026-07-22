"""绘制 Stanford 数据集 SOH 轨迹总览图（按协议族分面）。

用法:
    python scripts/plot_soh_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    table = pd.read_parquet(ROOT / "data/processed/stanford_soh_table.parquet")
    regular = table[~table["is_diagnostic"]]
    families = ["CC", "Periodic", "Synthetic", "Drive"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True, constrained_layout=True)
    for ax, fam in zip(axes, families):
        sub = regular[regular["protocol_family"] == fam]
        for _cell_id, g in sub.groupby("cell_id"):
            ax.plot(g["cycle_index"], g["soh"], lw=0.6, alpha=0.45, color="steelblue")
        mean_curve = sub.groupby("cycle_index")["soh"].mean()
        ax.plot(mean_curve.index, mean_curve.values, lw=2.2, color="crimson", label="family mean")
        n_cells = sub["cell_id"].nunique()
        ax.set_title(f"{fam} (n={n_cells})", fontsize=12)
        ax.set_xlabel("Cycle")
        ax.set_ylim(0.4, 1.1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")
    axes[0].set_ylabel("SOH (discharge capacity / initial)")

    out = ROOT / "results/figures/stanford_soh_trajectories.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.exit(main())
