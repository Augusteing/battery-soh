"""逐循环 SOH 标签模块（M2 · 第三步）。

论文口径（World Model, arXiv:2603.10527）:
  "SOH is defined as the ratio of current discharge capacity to the reference
   capacity at cycle 2, following the normalisation convention of [1]."

即: SOH(k) = Q_discharge(k) / Q_ref,  Q_ref = Q_discharge(cycle 2)

设计说明（软件工程）
---------------------
- 单一职责：本模块只产出"逐循环 SOH 标签"，不构建窗口、不切样本。
  样本标签的组装（当前 s(k) + 未来 s(k+1..k+80)）由 windows 模块负责；
- 兼容并保留现有口径：原表的 soh 列（前 10 循环中位数口径）保留不动，
  新增 soh_q2 列（论文口径），便于后续两套口径对比；
- 纯数据处理：输入 parquet -> 输出 parquet，无其他副作用；
- 防御性检查：参考容量缺失 / 非正 / NaN 都会显式报错（fail fast）。

用法:
    python "src/world_model/DataLoader/labels.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# 论文参考循环（按 cycle_index 计数，1 起）
Q2_CYCLE = 2


def add_soh_q2(table: pd.DataFrame) -> pd.DataFrame:
    """按电池计算 Q(2) 口径的 SOH，返回新增 soh_q2 列的表（副本）。

    参考容量选取规则:
      1) 优先取 cycle_index == Q2_CYCLE 的放电容量；
      2) 若该循环缺失，则退而取该电池第 2 个有效循环；
      3) 若有效循环不足 2 个，直接报错（该电池无法定义 SOH）。
    """
    table = table.copy()
    pieces: list[pd.Series] = []

    for cell_id, g in table.groupby("cell_id", sort=False):
        g = g.sort_values("cycle_index")

        # --- 确定参考容量 Q_ref ---
        exact = g[g["cycle_index"] == Q2_CYCLE]
        if len(exact) == 1:
            q_ref = float(exact["discharge_capacity"].iloc[0])
        elif len(g) >= 2:
            q_ref = float(g["discharge_capacity"].iloc[1])
        else:
            raise ValueError(f"{cell_id}: 有效循环不足 2 个，无法计算 Q(2) 参考容量")

        if not np.isfinite(q_ref) or q_ref <= 0:
            raise ValueError(f"{cell_id}: Q_ref = {q_ref} 非法（非正或非有限值）")

        pieces.append(pd.Series(g["discharge_capacity"] / q_ref,
                                index=g.index, name="soh_q2"))

    table["soh_q2"] = pd.concat(pieces).sort_index()

    # --- 防御：全表不应出现 NaN ---
    n_bad = int(table["soh_q2"].isna().sum())
    if n_bad:
        raise ValueError(f"soh_q2 存在 {n_bad} 个 NaN，请检查输入数据")

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/matr_soh_table.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/matr_soh_labels.parquet")
    args = parser.parse_args()

    table = pd.read_parquet(args.input)
    print(f"输入: {len(table):,} 行, {table['cell_id'].nunique()} 只电池")

    labeled = add_soh_q2(table)
    labeled.to_parquet(args.out, index=False)

    # 摘要输出
    print(f"输出: {args.out}")
    print(f"soh_q2 分布: min={labeled['soh_q2'].min():.3f} "
          f"median={labeled['soh_q2'].median():.3f} max={labeled['soh_q2'].max():.3f}")
    # 展示一只电池两种口径的差异（供人工检查）
    cell = sorted(labeled["cell_id"].unique())[0]
    sub = labeled[labeled["cell_id"] == cell].sort_values("cycle_index").head(5)
    print(f"\n示例 {cell} 前 5 个循环 (soh=前10中位数口径, soh_q2=Q(2)口径):")
    print(sub[["cycle_index", "discharge_capacity", "soh", "soh_q2"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()