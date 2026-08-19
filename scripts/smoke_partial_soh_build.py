"""partial_soh 中等规模冒烟测试。

目的（全量构建前的体检）：

  1. 拿少量电池、全循环，跑一遍片段索引构建流程；
  2. 统计耗时，据此估算 123 只电池的全量构建时间；
  3. 统计合法片段比例（is_valid_soh / is_valid_pretrain）；
  4. 检查老化循环的充电容量上限 q_max 是否仍覆盖
     50% + 20% + 7% 的窗口（0.847 Ah）。

用法:
    & "E:\\conda\\envs\\battery-soh\\python.exe" "scripts/smoke_partial_soh_build.py"
    & "E:\\conda\\envs\\battery-soh\\python.exe" "scripts/smoke_partial_soh_build.py" --cells 6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DL_DIR = ROOT / "src" / "partial_soh" / "DataLoader"
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from mat_io import discover_batch_files  # noqa: E402
from segments import (  # noqa: E402
    build_segment_index_for_cell,
    OBSERVED_CAPACITY_PCT,
    PREDICTION_CAPACITY_PCT,
    START_MAX_PCT,
    NOMINAL_CAPACITY_AH,
)


def pick_cells(labels: pd.DataFrame, splits: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """从 123 只电池中挑 n 只，保证跨批次、跨 train/test 的均衡抽样。"""
    meta = (
        labels[["cell_id", "batch"]]
        .drop_duplicates()
        .merge(splits, on="cell_id", how="left")
    )
    rng = np.random.default_rng(seed)

    # 按 (batch, split) 分组后，每组按比例抽，保证不集中在某个批次或某个 split。
    picked: list[pd.DataFrame] = []
    groups = list(meta.groupby(["batch", "split"], sort=False))
    per_group = max(1, n // len(groups))
    for (batch, split), g in groups:
        sample = g.sample(min(per_group, len(g)), random_state=rng.integers(0, 2**31))
        picked.append(sample)

    result = pd.concat(picked, ignore_index=True)
    # 如果分组抽样后不足 n，再随机补足；如果超过 n，截断。
    if len(result) < n:
        rest = meta[~meta["cell_id"].isin(result["cell_id"])]
        extra = rest.sample(n - len(result), random_state=rng.integers(0, 2**31))
        result = pd.concat([result, extra], ignore_index=True)
    return result.head(n).reset_index(drop=True)


def run_cell(
    cell_id: str,
    batch: str,
    cell_index: int,
    cycles: list[int],
    mat_path: Path,
) -> tuple[float, pd.DataFrame]:
    """构建一只电池的片段索引，返回 (耗时秒, 索引表)。"""
    t0 = time.perf_counter()
    table = build_segment_index_for_cell(
        mat_path,
        cell_id=cell_id,
        cell_index=cell_index,
        cycles=cycles,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, table


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "processed" / "partial_soh_labels.parquet")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "processed" / "partial_splits.parquet")
    parser.add_argument("--mat-dir", type=Path, default=ROOT / "data" / "external" / "matr")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "_smoke_partial_index.parquet")
    args = parser.parse_args()

    labels = pd.read_parquet(args.labels)
    splits = pd.read_parquet(args.splits)
    files = discover_batch_files(args.mat_dir)

    selected = pick_cells(labels, splits, args.cells, args.seed)
    print(f"抽样 {len(selected)} 只电池：")
    print(selected.sort_values(["batch", "cell_id"]).to_string(index=False))

    max_needed_ah = (START_MAX_PCT + OBSERVED_CAPACITY_PCT + PREDICTION_CAPACITY_PCT) * NOMINAL_CAPACITY_AH
    print(f"\n需要覆盖的容量窗口上界: {max_needed_ah:.4f} Ah "
          f"(= {START_MAX_PCT*100:.0f}% + {OBSERVED_CAPACITY_PCT*100:.0f}% + {PREDICTION_CAPACITY_PCT*100:.0f}%)")

    frames: list[pd.DataFrame] = []
    timings: list[dict] = []
    for _, row in selected.iterrows():
        cell_id = row["cell_id"]
        batch = row["batch"]
        cell_index = int(cell_id.rsplit("_", 1)[1][1:])
        if batch not in files:
            raise FileNotFoundError(f"找不到批次文件: {batch}")

        cell_labels = labels[labels["cell_id"] == cell_id].sort_values("cycle_index")
        cycles = [int(c) for c in cell_labels["cycle_index"].tolist()]

        t0 = time.perf_counter()
        seg = build_segment_index_for_cell(
            files[batch],
            cell_id=cell_id,
            cell_index=cell_index,
            cycles=cycles,
        )
        elapsed = time.perf_counter() - t0
        seg = seg.merge(
            cell_labels[["cell_id", "cycle_index", "soh_nominal"]],
            on=["cell_id", "cycle_index"],
            how="left",
        )
        frames.append(seg)

        n_cycles = len(cycles)
        n_valid_soh = int(seg["is_valid_soh"].sum())
        n_valid_pre = int(seg["is_valid_pretrain"].sum())
        q_max_min = float(seg["q_max_ah"].min())
        q_max_max = float(seg["q_max_ah"].max())
        q_max_below = int((seg["q_max_ah"] < max_needed_ah).sum())

        timings.append(
            {
                "cell_id": cell_id,
                "batch": batch,
                "split": row["split"],
                "n_cycles": n_cycles,
                "sec": elapsed,
                "sec_per_cycle": elapsed / n_cycles,
                "valid_soh": n_valid_soh,
                "valid_pretrain": n_valid_pre,
                "q_max_min": q_max_min,
                "q_max_max": q_max_max,
                "q_max_below_window": q_max_below,
            }
        )
        print(
            f"{cell_id:20s} cycles={n_cycles:5d} "
            f"time={elapsed:6.2f}s ({elapsed/n_cycles*1000:5.1f} ms/cycle) "
            f"valid_soh={n_valid_soh} valid_pre={n_valid_pre} "
            f"q_max=[{q_max_min:.3f},{q_max_max:.3f}] below={q_max_below}"
        )

    index = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(args.out, index=False)

    timings_df = pd.DataFrame(timings)
    total_sec = float(timings_df["sec"].sum())
    total_cycles = int(timings_df["n_cycles"].sum())
    avg_ms = total_sec / total_cycles * 1000

    print("\n==== 汇总 ====")
    print(f"抽样电池数: {len(timings_df)}, 循环总数: {total_cycles}, "
          f"片段索引行数: {len(index)}")
    print(f"平均耗时: {avg_ms:.1f} ms/cycle")

    # 全量估算：123 只电池共约 96815 个有效循环（标签行数）。
    n_all_cycles = len(labels)
    est_full_sec = n_all_cycles * avg_ms / 1000
    print(f"\n全量 123 只（约 {n_all_cycles:,} 个循环）估算构建时间: "
          f"{est_full_sec/60:.1f} 分钟")

    print("\n合法片段占比（本次抽样）:")
    print(f"  is_valid_soh      : {int(index['is_valid_soh'].sum())} / {len(index)}")
    print(f"  is_valid_pretrain : {int(index['is_valid_pretrain'].sum())} / {len(index)}")

    print("\nq_max_ah 分布:")
    print(index.groupby("cell_id")["q_max_ah"].agg(["min", "max", "count"]).round(4).to_string())
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
