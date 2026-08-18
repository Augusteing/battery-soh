"""M2 数据构成可视化：matr_windows.parquet（审计/报告用）。

四联图展示窗口表的构成:
  (a) 各 batch 的窗口数（标注电池数与协议数）
  (b) 各 batch 内部老化阶段占比（100% 堆叠柱）
  (c) 每只电池的窗口数分布（按 batch 分色）
  (d) 窗口当前 SOH s(k) 分布（标注四档阶段边界）

设计说明（软件工程）
---------------------
- 单一职责：本脚本只做"读取 + 绘图"，不做任何数据处理；
- 数据来源：窗口表 + 标签表（窗口表本身不含 SOH 数值，只有 stage 分类，
  绘制 s(k) 分布需按 (cell_id, k) 关联 labels 的 soh_q2）；
- 可复现：固定 matplotlib 样式与随机性（本图无随机成分）。

用法:
    python scripts/plot_windows_composition.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# 中文字体（Windows）；axes.unicode_minus=False 避免负号显示为方块
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

STAGE_ORDER = ("s1_healthy", "s2_mild", "s3_aged", "s4_heavy")
STAGE_COLORS = {"s1_healthy": "#4caf50", "s2_mild": "#ffb300",
                "s3_aged": "#ef5350", "s4_heavy": "#7e57c2"}
STAGE_CN = {"s1_healthy": "s1 健康(>0.95)", "s2_mild": "s2 轻度(0.90-0.95)",
            "s3_aged": "s3 老化(0.85-0.90)", "s4_heavy": "s4 重度(<0.85)"}


def load(windows_path: Path, labels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取窗口表，并按 (cell_id, k) 关联出每个窗口的当前 SOH s(k)。"""
    win = pd.read_parquet(windows_path)
    labels = pd.read_parquet(labels_path)[["cell_id", "cycle_index", "soh_q2"]]
    labels = labels.rename(columns={"cycle_index": "k", "soh_q2": "s_k"})
    win = win.merge(labels, on=["cell_id", "k"], how="left")
    missing = win["s_k"].isna().sum()
    if missing:
        raise ValueError(f"{missing:,} 个窗口找不到对应的 SOH 标签，请检查标签表")
    return win, labels


def panel_batch(ax, win: pd.DataFrame) -> None:
    """(a) 各 batch 窗口数 + 电池数/协议数标注。"""
    order = sorted(win["batch"].unique())
    counts = win.groupby("batch").size().reindex(order)
    cells = win.groupby("batch")["cell_id"].nunique().reindex(order)
    policies = win.groupby("batch")["policy"].nunique().reindex(order)

    bars = ax.bar(range(len(order)), counts.values, color="#42a5f5")
    ax.set_xticks(range(len(order)), order, rotation=15)
    ax.set_ylabel("窗口数")
    ax.set_title("(a) 各 batch 窗口构成")
    for i, (b, c, p) in enumerate(zip(bars, cells.values, policies.values)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 800,
                f"{int(b.get_height()):,}\n{c} 电池 / {p} 协议",
                ha="center", va="bottom", fontsize=9)


def panel_stage(ax, win: pd.DataFrame) -> None:
    """(b) 各 batch 内老化阶段占比（100% 堆叠柱）。"""
    order = sorted(win["batch"].unique())
    pivot = (win.assign(n=1)
                .pivot_table(index="batch", columns="stage", values="n",
                             aggfunc="sum", fill_value=0)
                .reindex(index=order, columns=STAGE_ORDER, fill_value=0))
    frac = pivot.div(pivot.sum(axis=1), axis=0)
    bottom = np.zeros(len(order))
    for stage in STAGE_ORDER:
        if stage not in frac.columns:
            continue
        ax.bar(range(len(order)), frac[stage].values, bottom=bottom,
               color=STAGE_COLORS[stage], label=STAGE_CN[stage])
        bottom += frac[stage].values
    ax.set_xticks(range(len(order)), order, rotation=15)
    ax.set_ylabel("窗口占比")
    ax.set_title("(b) 各 batch 老化阶段占比")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower center", ncol=3, fontsize=8, frameon=False)


def panel_per_cell(ax, win: pd.DataFrame) -> None:
    """(c) 每只电池的窗口数分布（按 batch 分色直方图）。"""
    per = win.groupby(["cell_id", "batch"]).size().reset_index(name="n")
    for batch, g in per.groupby("batch"):
        ax.hist(g["n"], bins=40, alpha=0.55, label=batch,
                histtype="stepfilled")
    ax.set_xlabel("每只电池的窗口数")
    ax.set_ylabel("电池数")
    ax.set_title("(c) 每电池窗口数分布")
    ax.legend(fontsize=8)


def panel_soh(ax, win: pd.DataFrame) -> None:
    """(d) 窗口当前 SOH s(k) 分布 + 四档阶段边界。"""
    bounds = [0.85, 0.90, 0.95]
    for lo, hi, stage in ((0.85, 0.90, "s3_aged"),
                          (0.90, 0.95, "s2_mild"),
                          (0.95, win["s_k"].max() + 0.01, "s1_healthy")):
        ax.axvspan(lo, hi, color=STAGE_COLORS[stage], alpha=0.18)
    ax.hist(win["s_k"], bins=80, color="#78909c", edgecolor="white")
    for b in bounds:
        ax.axvline(b, color="#37474f", ls="--", lw=0.8)
        ax.text(b, ax.get_ylim()[1] * 0.98, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xlabel("当前 SOH s(k)")
    ax.set_ylabel("窗口数")
    ax.set_title("(d) 窗口当前 SOH 分布")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=Path,
                        default=ROOT / "data/processed/matr_windows.parquet")
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "data/processed/matr_soh_labels.parquet")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/figures/windows_composition.png")
    args = parser.parse_args()

    win, _ = load(args.windows, args.labels)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_batch(axes[0, 0], win)
    panel_stage(axes[0, 1], win)
    panel_per_cell(axes[1, 0], win)
    panel_soh(axes[1, 1], win)

    # 总体统计角注
    dist = win["stage"].value_counts(normalize=True)
    txt = (f"总计 {len(win):,} 窗口 / {win['cell_id'].nunique()} 电池 / "
           f"{win['policy'].nunique()} 协议\n"
           + "  ".join(f"{s}={dist.get(s, 0):.1%}" for s in STAGE_ORDER))
    fig.text(0.5, 0.955, txt, ha="center", fontsize=10,
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#bdbdbd"))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"已保存 -> {args.out}")
    print(txt)


if __name__ == "__main__":
    main()
