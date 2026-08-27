"""temperature_soh 数据集划分模块。

本模块只负责把“电池”分到 train / val / test，不读取曲线、不生成片段。

所有策略共同遵守的电池级原则：

    - 必须在电池级别划分；
    - 同一只电池的所有片段只能出现在同一个 split 中；
    - 绝不能出现同一只电池的部分片段既在训练集、又在测试集。

三种策略（--strategy）：

1. paper（默认）：纯随机 99 train / 24 test，与 partial_soh 完全一致，
   用于复现对照。它验证的是“同工况分布内的跨电池泛化”；
2. ratio：随机 train/val/test = 70/15/15，开发期通用；
3. policy：按充电协议留出（held-out policies）。train/val/test 的
   充电协议集合互不重叠，测试集只包含训练时完全没见过的协议，
   用于验证“工况泛化能力”（新充电协议下的 SOH 估计）。

运行：

```powershell
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/DataLoader/splits.py
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/DataLoader/splits.py --strategy policy
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = ROOT / "data" / "processed" / "temperature_soh" / "soh_labels.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "temperature_soh" / "splits.parquet"


def paper_count_split(
    cells: pd.DataFrame,
    train_count: int = 99,
    test_count: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """纯随机 99 train / 24 test，与 partial_soh（Scientific Reports 2026）一致。

    输入 cells 至少包含 cell_id；电池总数必须等于 train_count + test_count
    （123-cell 口径），否则显式报错，防止混入未过滤的数据。

    注意：本策略只分 train/test，没有 val。若训练需要早停，
    请在训练器里从 train 中再留出小比例验证（或用 --strategy ratio）。
    """
    n_cells = len(cells)
    expected = train_count + test_count
    if n_cells != expected:
        raise ValueError(
            f"当前电池数 {n_cells} 不等于论文口径 {expected} "
            f"(train={train_count}, test={test_count})。"
            "请先用 quality.apply_paper_123 过滤到 123-cell 口径。"
        )

    rng = np.random.default_rng(seed)
    shuffled = cells["cell_id"].to_numpy(dtype=str)
    rng.shuffle(shuffled)

    split = ["train"] * train_count + ["test"] * test_count
    return pd.DataFrame({"cell_id": shuffled, "split": split})


def ratio_split(
    cells: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """随机按比例划分 train/val/test（不按工况分层，开发期通用）。"""
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios 必须是三个和为 1 的数，得到 {ratios}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios 不能为负数: {ratios}")

    n = len(cells)
    counts = [int(round(n * r)) for r in ratios]
    counts[-1] = n - sum(counts[:-1])
    if any(c < 1 for c in counts):
        raise ValueError(f"电池数 {n} 太少，无法按 {ratios} 划分")

    rng = np.random.default_rng(seed)
    shuffled = cells["cell_id"].to_numpy(dtype=str)
    rng.shuffle(shuffled)

    split = (
        ["train"] * counts[0]
        + ["val"] * counts[1]
        + ["test"] * counts[2]
    )
    return pd.DataFrame({"cell_id": shuffled, "split": split})


def policy_split(
    cells: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.80, 0.10, 0.10),
    seed: int = 42,
) -> pd.DataFrame:
    """按充电协议留出划分 train/val/test（工况泛化验证）。

    输入 cells 至少包含 cell_id 与 policy 两列。

    核心约束：同一个 policy 的所有电池只能进入同一个 split。
    因此 train/val/test 的协议集合互不重叠，测试集协议在训练时
    完全不可见——这正是“工况泛化”的含义。

    分配算法（贪心，目标：各组电池数接近 ratios）：
      1. 按 policy 聚合并统计每个协议的电池数；
      2. 协议按电池数降序排列（同数量时顺序由 seed 打乱决定，
         让 seed 对结果有可复现的影响）；
      3. 依次把每个协议放入“当前电池数 / 目标比例”最小的组，
         即让大协议分散、小协议填空，最终各组占比贴近目标。

    校验：协议数至少 3（否则无法三组互斥），每组至少 1 个协议、
    至少 1 只电池。
    """
    required = {"cell_id", "policy"}
    missing = required - set(cells.columns)
    if missing:
        raise KeyError(f"cells 缺少字段: {sorted(missing)}（policy_split 需要 policy 列）")

    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios 必须是三个和为 1 的数，得到 {ratios}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios 不能为负数: {ratios}")

    # 按协议聚合电池数。
    counts = cells.groupby("policy")["cell_id"].nunique().sort_values(ascending=False)
    n_policies = len(counts)
    if n_policies < 3:
        raise ValueError(
            f"只有 {n_policies} 种协议，无法分成三个互斥组（至少需要 3 种）"
        )

    # 打乱后按电池数降序：同数量协议的先后顺序受 seed 控制。
    rng = np.random.default_rng(seed)
    ordered = counts.index.to_numpy(dtype=str)
    rng.shuffle(ordered)
    ordered = sorted(ordered, key=lambda p: -counts[p])

    # 贪心分配：每组累计电池数 / 目标比例 最小者获得下一个协议。
    group_names = ("train", "val", "test")
    group_policies: dict[str, list[str]] = {g: [] for g in group_names}
    group_sizes: dict[str, int] = {g: 0 for g in group_names}
    target_ratio = dict(zip(group_names, ratios))

    for policy in ordered:
        chosen = min(
            group_names,
            key=lambda g: group_sizes[g] / max(target_ratio[g], 1e-9),
        )
        group_policies[chosen].append(policy)
        group_sizes[chosen] += int(counts[policy])

    for g in group_names:
        if not group_policies[g]:
            raise ValueError(f"组 {g} 没有分到任何协议，请调整 ratios 或增加协议数")

    # 生成 cell_id -> split 映射表。
    policy_to_group: dict[str, str] = {}
    for g, policies in group_policies.items():
        for p in policies:
            policy_to_group[p] = g

    out = cells[["cell_id", "policy"]].copy()
    out["split"] = out["policy"].map(policy_to_group)
    return out[["cell_id", "split"]].drop_duplicates().reset_index(drop=True)


def _print_summary(split_table: pd.DataFrame, cells: pd.DataFrame) -> None:
    """打印各 split 的电池数与协议数（policy 策略时才有协议信息）。"""
    merged = split_table.merge(
        cells[["cell_id", "policy"]].drop_duplicates(), on="cell_id", how="left"
    )
    print()
    print("电池数:", len(split_table))
    for split_name, group in merged.groupby("split", sort=False):
        n_cells = group["cell_id"].nunique()
        n_policies = group["policy"].nunique() if group["policy"].notna().any() else 0
        print(
            f"  {split_name}: 电池 {n_cells} 只"
            + (f", 协议 {n_policies} 种" if n_policies else "")
        )


def main() -> None:
    """从标签表生成电池级划分，保存 parquet。"""
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strategy", choices=("paper", "ratio", "policy"), default="paper"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = pd.read_parquet(args.labels)
    cells = labels[["cell_id", "policy"]].drop_duplicates().reset_index(drop=True)

    if args.strategy == "paper":
        split_table = paper_count_split(cells, seed=args.seed)
    elif args.strategy == "ratio":
        split_table = ratio_split(cells, seed=args.seed)
    else:
        split_table = policy_split(cells, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    split_table.to_parquet(args.out, index=False)
    print(f"策略: {args.strategy}, seed: {args.seed}")
    _print_summary(split_table, cells)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
