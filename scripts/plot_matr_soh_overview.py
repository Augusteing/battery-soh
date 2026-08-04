"""MATR 各电池 SOH 轨迹总览图（按协议上色）。

用法:
    python scripts/plot_matr_soh_overview.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/matr_soh_table.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "results/figures/matr_soh_trajectories.png")
    args = parser.parse_args()

    table = pd.read_parquet(args.input)
    policies = table["policy"].dropna().unique()
    cmap = plt.get_cmap("tab20")
    color = {p: cmap(i % 20) for i, p in enumerate(sorted(policies))}

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for cell_id, g in table.groupby("cell_id"):
        ax.plot(g["cycle_index"], g["soh"], lw=0.8, alpha=0.6, color=color[g["policy"].iloc[0]])
    ax.axhline(0.8, color="k", ls="--", lw=1, label="EOL (80%)")
    ax.set_xlabel("cycle")
    ax.set_ylabel("SOH")
    ax.set_ylim(0.5, 1.05)
    ax.set_title(f"MATR 20170512: {table['cell_id'].nunique()} LFP cells")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=7, ncol=3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
