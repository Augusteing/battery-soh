"""partial_soh 数据构建统一入口。

本模块只负责“编排”其他模块：

    labels.py   -> 生成 SOH 标签；
    splits.py   -> 生成电池级 train/val/test；
    mat_io.py   -> 读取原始曲线；
    charge.py   -> 提取充电阶段；
    segments.py -> 生成部分充电片段索引。

最终产物是一个轻量的索引表，而不是把每条 V/I/T 曲线都写进 parquet。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DL_DIR = Path(__file__).resolve().parent
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from mat_io import discover_batch_files  # noqa: E402
from quality import mark_bad_cycles  # noqa: E402
from segments import build_segment_index_for_cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = ROOT / "data" / "processed" / "partial_soh_labels.parquet"
DEFAULT_SPLITS = ROOT / "data" / "processed" / "partial_splits.parquet"
DEFAULT_OUT = ROOT / "data" / "processed" / "partial_segments_index.parquet"


def cell_index_from_id(cell_id: str) -> int:
    """从 cell_id（例如 2017-06-30_c000）解析批次内编号。"""
    suffix = cell_id.rsplit("_", 1)[1]
    if not suffix.startswith("c"):
        raise ValueError(f"cell_id 格式错误: {cell_id}")
    return int(suffix[1:])


def batch_from_id(cell_id: str) -> str:
    """从 cell_id 中提取批次名（例如 2017-06-30）。"""
    return cell_id.rsplit("_", 1)[0]


def build_index_table(
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    mat_dir: Path,
    max_cells: int | None = None,
    max_cycles_per_cell: int | None = None,
) -> pd.DataFrame:
    """为划分好的电池生成部分充电片段索引表。

    当前版本不做曲线物化，只生成索引。
    """
    files = discover_batch_files(mat_dir)
    cell_ids = splits["cell_id"].tolist()
    if max_cells is not None:
        cell_ids = cell_ids[:max_cells]

    frames: list[pd.DataFrame] = []
    for cell_id in cell_ids:
        batch = batch_from_id(cell_id)
        if batch not in files:
            raise FileNotFoundError(f"找不到批次文件: {batch}")

        cell_labels = labels[labels["cell_id"] == cell_id].sort_values("cycle_index")
        cycles = [int(c) for c in cell_labels["cycle_index"].tolist()]
        if max_cycles_per_cell is not None:
            cycles = cycles[:max_cycles_per_cell]
        if not cycles:
            continue

        cell_index = cell_index_from_id(cell_id)
        seg = build_segment_index_for_cell(
            files[batch],
            cell_id=cell_id,
            cell_index=cell_index,
            cycles=cycles,
        )
        seg = seg.merge(
            cell_labels[["cell_id", "cycle_index", "soh_nominal", "soh_q2", "policy"]],
            on=["cell_id", "cycle_index"],
            how="left",
        )
        frames.append(seg)

    if not frames:
        raise ValueError("没有生成任何片段，请检查 max_cells / max_cycles 设置")

    table = pd.concat(frames, ignore_index=True)
    table = table.merge(splits, on="cell_id", how="left")
    return table


def main() -> None:
    """开发阶段默认小规模运行；加 --full 生成全量片段索引表。"""
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--mat-dir", type=Path, default=ROOT / "data" / "external" / "matr")
    parser.add_argument("--max-cells", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--full",
        action="store_true",
        help="忽略 max-cells / max-cycles，处理全部 123 只电池的所有循环",
    )
    args = parser.parse_args()

    labels = pd.read_parquet(args.labels)
    labels = mark_bad_cycles(labels)
    labels = labels[labels["is_valid_label"] & ~labels["is_bad_cycle"]].copy()
    splits = pd.read_parquet(args.splits)

    max_cells = None if args.full else args.max_cells
    max_cycles = None if args.full else args.max_cycles
    table = build_index_table(
        labels,
        splits,
        mat_dir=args.mat_dir,
        max_cells=max_cells,
        max_cycles_per_cell=max_cycles,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(table.head().to_string(index=False))
    print(f"rows: {len(table)}, cells: {table['cell_id'].nunique()}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
