"""partial_soh 数据集划分模块。

本模块只负责把“电池”分到 train / val / test。

最重要的原则：

    - 必须在电池级别划分；
    - 同一只电池的所有片段只能出现在同一个 split 中；
    - 绝不能出现同一只电池的部分片段既在训练集、又在测试集。

它不生成片段，不读取曲线，只处理 cell_id。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = ROOT / "data" / "processed" / "partial_soh_labels.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "partial_splits.parquet"


def paper_count_split(
    cells: pd.DataFrame,
    train_count: int = 99,
    test_count: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """按论文的 99 train / 24 test 电池数划分。

    输入 cells 至少包含 cell_id。
    如果电池总数不等于 train_count + test_count，会显式报错，
    避免误把不符合 123-cell 口径的数据混进来。

    注意：Scientific Reports 2026 的 123 只 = Severson 124 - 1 只
    异常短寿命电池 b2c1。因此这里期望的电池数是 123，而不是 124。
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
    """按比例划分 train/val/test。

    这是通用划分方式，适合在还没确定论文 124-cell 筛选口径时做开发。
    """
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


def main() -> None:
    """从标签表生成电池级划分。

    默认使用论文的 99/24 口径（需先过滤到 123 只电池）；
    也可用 --strategy ratio 使用 70/15/15 的通用开发口径。
    """
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strategy", choices=("ratio", "paper"), default="paper")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = pd.read_parquet(args.labels)
    cells = labels[["cell_id"]].drop_duplicates().reset_index(drop=True)

    if args.strategy == "paper":
        split_table = paper_count_split(cells, seed=args.seed)
    else:
        split_table = ratio_split(cells, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    split_table.to_parquet(args.out, index=False)
    print(split_table["split"].value_counts().to_string())
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
