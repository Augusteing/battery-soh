"""窗口表构建模块（M2 · 第四步）。

论文口径（World Model, arXiv:2603.10527）:
  "Sliding windows of W = 30 consecutive input cycles with future target
   horizon H = 80 cycles are constructed within each cell. No cross-cell
   windows are created."

窗口定义（k 为窗口末尾循环）:
  - 输入: 循环 k-29 .. k（W = 30 个连续循环）
  - 输出: 当前 SOH s(k) + 未来轨迹 s(k+1 .. k+80)（H = 80）
  - 合法性: 要求 k + H <= 该电池总循环数（未来地平线完整）
            且 k 之前至少有 W-1 个循环（输入窗口完整）

设计说明（软件工程）
---------------------
- 索引式窗口：本模块只产出"窗口索引表"（每行 = 一个 (cell, k)），
  不物化任何曲线数据。全部窗口若物化（30x3x1000）约 36GB，
  而索引表只有几十万行，训练时按索引读取即可；
- 单一职责：只负责"哪些 (cell, k) 是合法窗口"；
  标签/曲线的取用由后续数据集加载器完成；
- 坏循环清洗：触碰坏循环（data_quality.BAD_CYCLES）的窗口直接丢弃，
  不插值、不整删电池；窗口仍保持"连续 30 循环"语义；
- 防御性检查：W/H/stride 参数合法性、无窗口时显式报错。

用法:
    python "src/world_model/DataLoader/windows.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import data_quality  # 同目录模块：论文数据排除规则

ROOT = Path(__file__).resolve().parents[3]

# 论文默认参数
DEFAULT_W = 30   # 输入窗口长度（循环数）
DEFAULT_H = 80   # 未来预测地平线（循环数）

# 四档老化阶段（论文 Imbalance handling 口径）
AGING_STAGES = (
    (0.95, float("inf"), "s1_healthy"),
    (0.90, 0.95, "s2_mild"),
    (0.85, 0.90, "s3_aged"),
    (-float("inf"), 0.85, "s4_heavy"),
)


def assign_stage(soh: float) -> str:
    """按当前 SOH 划分老化阶段（用于逆频率采样与分桶评估）。"""
    for lo, hi, name in AGING_STAGES:
        if lo < soh <= hi:
            return name
    raise ValueError(f"SOH {soh} 无法划分老化阶段")


def build_window_table(labels: pd.DataFrame,
                       W: int = DEFAULT_W,
                       H: int = DEFAULT_H,
                       stride: int = 1) -> pd.DataFrame:
    """在每只电池内部生成合法窗口的索引表。

    参数
    ----
    labels : 标签表（至少含 cell_id, cycle_index, batch, policy）
    W      : 输入窗口长度（循环数）
    H      : 未来预测地平线（循环数）
    stride : 窗口滑动步长（1 = 每个循环都产生一个窗口）

    返回
    ----
    pd.DataFrame: 列 [cell_id, pos, start, k, batch, policy]
      - pos   : 窗口末尾循环在该电池内的 0-based 位置（用于索引曲线数组）
      - start : 窗口起始循环的 cycle_index
      - k     : 窗口末尾循环的 cycle_index
    注意: 窗口绝不跨电池（组内滑动）。
    """
    if W < 1 or H < 0 or stride < 1:
        raise ValueError(f"非法参数: W={W}, H={H}, stride={stride}")
    if "soh_q2" not in labels.columns:
        raise ValueError("labels 缺少 soh_q2 列，请先运行 labels.py 生成标签")

    rows: list[dict] = []
    for cell_id, g in labels.groupby("cell_id", sort=False):
        g = g.sort_values("cycle_index").reset_index(drop=True)
        n = len(g)
        # 坏循环位置（0-based 行索引）；若标签表没有 is_bad_cycle 列则视为无
        if "is_bad_cycle" in g.columns:
            bad_positions = set(g.index[g["is_bad_cycle"]].tolist())
        else:
            bad_positions = set()

        # 防御：该电池循环数不足以容纳 输入窗口 + 未来地平线 -> 无窗口
        if n < W + H:
            continue

        # 窗口末尾的 0-based 位置: 需要 pos >= W-1 且 pos + H <= n-1
        positions = range(W - 1, n - H, stride)
        meta = {"batch": g["batch"].iloc[0], "policy": g["policy"].iloc[0]}
        for p in positions:
            # 坏循环清洗：输入 (p-W+1..p)、当前标签 p、未来 (p+1..p+H)
            # 任一位置为坏循环则整窗丢弃（保持窗口连续，不插值）
            if any(p - W + 1 <= q <= p + H for q in bad_positions):
                continue
            rows.append({
                "cell_id": cell_id,
                "pos": p,
                "start": g.loc[p - W + 1, "cycle_index"],
                "k": g.loc[p, "cycle_index"],
                "stage": assign_stage(g.loc[p, "soh_q2"]),
                **meta,
            })

    table = pd.DataFrame(rows, columns=["cell_id", "pos", "start", "k", "batch", "policy", "stage"])
    if table.empty:
        raise ValueError("没有任何合法窗口：请检查 W/H 是否超过电池最短寿命")
    return table


def count_bad_windows(table: pd.DataFrame,
                      labels: pd.DataFrame,
                      W: int = DEFAULT_W,
                      H: int = DEFAULT_H) -> int:
    """统计 table 中"触碰坏循环"的窗口数（用于审计与报告）。"""
    if "is_bad_cycle" not in labels.columns:
        return 0
    bad = labels[labels["is_bad_cycle"]]
    total = 0
    for cell_id, g in bad.groupby("cell_id"):
        cycles = set(g["cycle_index"].tolist())
        sub = table[table["cell_id"] == cell_id]
        for start, k in sub[["start", "k"]].to_numpy():
            s, kk = int(start), int(k)
            if any(s <= q <= kk + H for q in cycles):
                total += 1
    return total


def window_crossing_rate(table: pd.DataFrame,
                         labels: pd.DataFrame,
                         H: int = DEFAULT_H,
                         threshold: float = 0.95) -> float:
    """统计"未来 H 循环内 SOH 跌破 threshold"的窗口比例。

    用途：对照论文 "10.9% of windows cross SOH 0.95"（sanity check）。
    定义：窗口的预测地平线 s(k+1..k+H) 内最小值 < threshold 即视为跨过。
    """
    total = 0
    crossing = 0
    for cell_id, g in labels.groupby("cell_id", sort=False):
        g = g.sort_values("cycle_index").reset_index(drop=True)
        s = g["soh_q2"].to_numpy()
        if len(s) < H + 1:
            continue
        # 以每个位置为起点的 H 长未来窗口的最小值
        future_min = sliding_window_view(s, H).min(axis=1)  # future_min[i] = min(s[i..i+H-1])
        cell_table = table[table["cell_id"] == cell_id]
        for p in cell_table["pos"].to_numpy():
            total += 1
            if future_min[p + 1] < threshold:   # 未来从 p+1 开始
                crossing += 1
    return crossing / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/processed/matr_soh_labels.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/matr_windows.parquet")
    parser.add_argument("--W", type=int, default=DEFAULT_W)
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--keep-all", action="store_true", help="禁用论文排除规则（消融用）")
    args = parser.parse_args()

    labels = pd.read_parquet(args.labels)
    print(f"标签表: {len(labels):,} 行, {labels['cell_id'].nunique()} 只电池")

    if not args.keep_all:
        labels = data_quality.apply_exclusions(labels)
        labels = data_quality.mark_bad_cycles(labels)

    table = build_window_table(labels, W=args.W, H=args.H, stride=args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)

    rate = window_crossing_rate(table, labels, H=args.H)
    n_bad = count_bad_windows(table, labels, W=args.W, H=args.H)

    print(f"窗口表: {len(table):,} 个窗口 -> {args.out}")
    print(f"  参与电池: {table['cell_id'].nunique()} / {labels['cell_id'].nunique()}")
    if n_bad:
        print(f"  警告: 仍有 {n_bad:,} 个窗口触碰坏循环（应已全部清除）")
    else:
        print("  坏循环窗口: 0（已全部清除）")
    print(f"  每电池窗口数: min={table.groupby('cell_id').size().min()}, "
          f"max={table.groupby('cell_id').size().max()}")
    print(f"  未来 {args.H} 循环内 SOH 跌破 0.95 的窗口比例: {rate:.1%}  "
          f"(论文报告 10.9%)")
    dist = table["stage"].value_counts(normalize=True).sort_index()
    print("窗口老化阶段分布（按当前 SOH s(k)）:")
    for stage, frac in dist.items():
        print(f"  {stage}: {frac:.1%}")
    print(table.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
