"""数据划分模块（M2 · 第六步）。

方案口径（docs/b_plan_design.md §1.3）:
  - 划分单元 = 电池：同一电池产生的全部窗口必须属于同一集合（防泄漏，
    否则同电池样本的高度相关性会导致指标虚高）；
  - 主线（split_by_cell）: 按电池随机划分 70/15/15，且在每个 batch
    内部等比例划分，检验"未见过的新电池"的泛化能力；
  - 扩展（split_by_policy）: 以充放电协议为划分单元，测试协议在训练中
    完全不可见，检验"未见工况"的适应能力（对应赛题变工况要求）。

设计说明（软件工程）
---------------------
- 单一职责：只生成"电池 -> 集合"的映射表，不移动/复制任何窗口或曲线数据；
- 确定性：seed 固定即可完全复现；
- 防泄漏：映射表按 cell_id 合并到窗口表后，同一电池不可能跨集合；
- 平衡：按电池划分时在 batch 内部分层；按协议划分时按电池数做容量均衡。

用法:
    python "src/world_model/DataLoader/splits.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_SEED = 42


def _check_ratios(ratios: tuple[float, float, float]) -> None:
    """校验三段比例：长度 3、全为正、和为 1。"""
    if len(ratios) != 3 or any(r <= 0 for r in ratios):
        raise ValueError(f"非法比例: {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"比例之和必须为 1: {ratios}")


def _split_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """按比例把 n 分成三段整数（舍入误差并入 test）。"""
    _check_ratios(ratios)
    n1 = int(round(n * ratios[0]))
    n2 = int(round(n * ratios[1]))
    n3 = n - n1 - n2
    if n1 < 0 or n2 < 0 or n3 < 0:
        raise ValueError(f"n={n} 太小，无法按 {ratios} 划分")
    return n1, n2, n3


def split_by_cell(cells: pd.DataFrame,
                  ratios: tuple[float, float, float] = DEFAULT_RATIOS,
                  seed: int = DEFAULT_SEED) -> dict[str, str]:
    """按电池随机划分（主线），返回 {cell_id: split}。

    划分单元 = 电池；在每个 batch 内部单独洗牌并切 70/15/15，
    保证各集合的批次构成与全量一致（分层抽样）。
    """
    rng = np.random.default_rng(seed)
    assign: dict[str, str] = {}
    for batch, g in cells.groupby("batch", sort=True):
        ids = g["cell_id"].to_numpy()
        rng.shuffle(ids)
        n1, n2, _ = _split_sizes(len(ids), ratios)
        bounds = (ids[:n1], ids[n1:n1 + n2], ids[n1 + n2:])
        for split, chunk in zip(SPLIT_NAMES, bounds):
            for cid in chunk:
                assign[str(cid)] = split
    return assign


def split_by_policy(cells: pd.DataFrame,
                    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
                    seed: int = DEFAULT_SEED) -> dict[str, str]:
    """按协议划分（扩展），返回 {cell_id: split}。

    划分单元 = policy；同一协议的全部电池必须同属一个集合（即使跨批次，
    例如 4.8C(80%)-4.8C 同时出现在两个 batch）。

    均衡策略：洗牌后按"相对目标占比缺口最大优先"贪心分配，保证
    train/val/test 的电池数尽量接近 70/15/15。
    """
    rng = np.random.default_rng(seed)
    total = len(cells)
    n1, n2, n3 = _split_sizes(total, ratios)
    targets = dict(zip(SPLIT_NAMES, (n1, n2, n3)))

    groups = cells.groupby("policy", sort=True)["cell_id"].agg(list)
    order = list(groups.index)
    rng.shuffle(order)

    counts = {s: 0 for s in SPLIT_NAMES}
    assign: dict[str, str] = {}
    for policy in order:
        ids = [str(c) for c in groups[policy]]
        # 选"距离自身目标缺口最大"的集合，避免某个集合被撑爆
        split = min(SPLIT_NAMES,
                    key=lambda s: (counts[s] / max(targets[s], 1), SPLIT_NAMES.index(s)))
        counts[split] += len(ids)
        for cid in ids:
            assign[cid] = split
    return assign


def build_splits(windows: pd.DataFrame,
                 ratios: tuple[float, float, float] = DEFAULT_RATIOS,
                 seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """生成 {cell_id, batch, policy, split_by_cell, split_by_policy} 映射表。"""
    cells = (windows.groupby("cell_id", as_index=False)
                    .agg(batch=("batch", "first"), policy=("policy", "first")))
    by_cell = split_by_cell(cells, ratios=ratios, seed=seed)
    by_policy = split_by_policy(cells, ratios=ratios, seed=seed)
    cells["split_by_cell"] = cells["cell_id"].map(by_cell)
    cells["split_by_policy"] = cells["cell_id"].map(by_policy)

    missing = cells[cells["split_by_cell"].isna() | cells["split_by_policy"].isna()]
    if len(missing):
        raise ValueError(f"有电池未被划分: {missing['cell_id'].tolist()}")
    return cells


def _summarize(cells: pd.DataFrame, windows: pd.DataFrame, col: str) -> None:
    """打印某个划分方案的统计：电池数、窗口数、窗口老化阶段分布。"""
    print(f"\n[{col}]")
    per = cells.groupby(col)["cell_id"].agg(["count", "nunique"])
    print("  电池数: " + ", ".join(f"{s}={int(per.loc[s, 'count'])}"
                                   for s in SPLIT_NAMES if s in per.index))
    joined = windows.merge(cells[["cell_id", col]], on="cell_id", how="left")
    wc = joined.groupby(col).size()
    print("  窗口数: " + ", ".join(f"{s}={int(wc.get(s, 0))}" for s in SPLIT_NAMES))
    dist = joined.groupby([col, "stage"]).size().unstack(fill_value=0)
    dist = dist.div(dist.sum(axis=1), axis=0) * 100
    print("  老化阶段分布(%):")
    for s in SPLIT_NAMES:
        if s in dist.index:
            row = "  ".join(f"{st}={dist.loc[s, st]:.1f}" for st in dist.columns)
            print(f"    {s}: {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=Path, default=ROOT / "data/processed/matr_windows.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/splits.parquet")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    windows = pd.read_parquet(args.windows)
    print(f"窗口表: {len(windows):,} 个窗口, {windows['cell_id'].nunique()} 只电池")

    cells = build_splits(windows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(args.out, index=False)
    print(f"划分映射表: {len(cells)} 只电池 -> {args.out}")

    _summarize(cells, windows, "split_by_cell")
    _summarize(cells, windows, "split_by_policy")

    # 防泄漏自检：任一电池只能出现在一个集合
    for col in ("split_by_cell", "split_by_policy"):
        dup = cells.groupby("cell_id")[col].nunique()
        assert (dup == 1).all(), f"{col} 存在跨集合电池"
    print("\n防泄漏自检通过：同一电池不跨集合。")


if __name__ == "__main__":
    main()
