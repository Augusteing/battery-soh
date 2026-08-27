"""temperature_soh 统一片段索引表构建入口。

本模块负责“编排”其他模块，把三张中间表合并成一份训练用的索引表：

    mat_io.py   -> 读取 Severson .mat（统一循环结构）；
    segments.py -> 生成 4 通道部分充电片段索引（每循环最多 51 个窗口）；
    labels.py   -> SOH 标签（Qc / 1.1，原论文口径）；
    splits.py   -> 电池级 train/val/test 划分。

最终产物是**轻量索引表**（parquet），不把 V/I/T/Q 曲线写进文件：
训练时按 (cell_id, cycle_index, start_ah, end_ah) 惰性读取并插值，
避免把约 8 GB 的原始数据复制一份。

合并键：
  - 片段索引 + 标签：cell_id + cycle_index；
  - 标签 + 划分：cell_id。

输出列：

  cell_id, cycle_index, split,
  start_ah, end_ah, pred_start_ah, pred_end_ah,
  is_valid_soh, is_valid_pretrain,
  n_charge_points, q_min_ah, q_max_ah,
  temperature_c, charge_capacity_ah, soh, policy,
  is_bad_cycle, is_valid_label

运行：

```powershell
# 开发期小规模（默认 3 只电池 × 3 个循环）
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/DataLoader/build_index.py

# 全量构建（123 只电池全部有效循环，约 500 万行，需数分钟）
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/DataLoader/build_index.py --full
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DL_DIR = Path(__file__).resolve().parent
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from mat_io import discover_batch_files, load_unified_cell  # noqa: E402
from segments import build_segment_index_for_cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = ROOT / "data" / "processed" / "temperature_soh" / "soh_labels.parquet"
DEFAULT_SPLITS = ROOT / "data" / "processed" / "temperature_soh" / "splits.parquet"
DEFAULT_OUT = ROOT / "data" / "processed" / "temperature_soh" / "segment_index.parquet"
DEFAULT_MAT_DIR = ROOT / "data" / "external" / "matr"

# Severson 全部在 30°C 恒温箱，片段索引的温度标签统一为 30°C。
MATR_TEMPERATURE_C = 30.0


def batch_from_id(cell_id: str) -> str:
    """从 cell_id（例如 2017-06-30_c000）提取批次名。"""
    return cell_id.rsplit("_", 1)[0]


def cell_index_from_id(cell_id: str) -> int:
    """从 cell_id（例如 2017-06-30_c000）解析批次内编号。"""
    suffix = cell_id.rsplit("_", 1)[1]
    if not suffix.startswith("c"):
        raise ValueError(f"cell_id 格式错误: {cell_id}")
    return int(suffix[1:])


def build_index_table(
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    mat_dir: Path | None = None,
    max_cells: int | None = None,
    max_cycles_per_cell: int | None = None,
) -> pd.DataFrame:
    """为划分好的电池生成统一片段索引表。

    参数
    ----
    labels : 已过滤到“有效标签”的标签表（main 里完成过滤）。
    splits : cell_id -> split 的划分表。
    max_cells / max_cycles_per_cell : 开发期限制规模；None 表示全量。

    每只电池的处理流程：
      1. 用 load_unified_cell 把该电池的全部循环读成统一结构；
      2. 对该电池的有效循环生成片段索引；
      3. 合并该电池的标签（soh、policy 等）；
      4. 最后统一合并划分（split）。
    """
    mat_dir = Path(mat_dir) if mat_dir is not None else DEFAULT_MAT_DIR
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
        cell = load_unified_cell(files[batch], cell_index=cell_index, batch_name=batch)

        seg = build_segment_index_for_cell(
            cell,
            cycles=cycles,
            temperature_c=MATR_TEMPERATURE_C,
        )
        seg = seg.merge(
            cell_labels[
                [
                    "cell_id",
                    "cycle_index",
                    "charge_capacity_ah",
                    "soh",
                    "policy",
                    "is_bad_cycle",
                    "is_valid_label",
                ]
            ],
            on=["cell_id", "cycle_index"],
            how="left",
        )
        frames.append(seg)
        print(f"[build_index] {cell_id}: {len(seg)} 片段", flush=True)

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
    parser.add_argument("--mat-dir", type=Path, default=DEFAULT_MAT_DIR)
    parser.add_argument("--max-cells", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--full",
        action="store_true",
        help="忽略 max-cells / max-cycles，处理全部电池的所有有效循环",
    )
    args = parser.parse_args()

    labels = pd.read_parquet(args.labels)
    # 只保留有效标签、非坏循环的行（训练口径）。
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
    print("\n表头预览:")
    print(table.head().to_string(index=False))
    print(f"\nrows: {len(table)}, cells: {table['cell_id'].nunique()}")
    if "split" in table.columns:
        print(table["split"].value_counts().to_string())
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
