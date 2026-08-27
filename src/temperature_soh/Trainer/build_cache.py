"""把 4 通道片段索引物化为磁盘缓存（memmap），加速全量训练。

问题
----
493 万片段的全量训练中，每个 batch 都要在 CPU 上把几千个片段插值到
101 点容量网格。实测 CPU 插值成为瓶颈（GPU 大量时间在等数据），
惰性加载跑不完 50+50 epoch。

关键观察
--------
片段是静态数据：同一个片段每次插值结果完全一样。因此可以一次性把
全部片段插值好写入磁盘，训练时直接按行切片读取，CPU 几乎不干活。

产物（每个 split 一份，4 通道）
-----------------------------
cache_dir/
  X_<split>.npy                  float32 (N, 101, 4)  输入 [I, V, Q, T']
  X_future_<split>.npy           float32 (N, 36, 4)   未来 7% 预测窗 [I, V, Q, T']
  y_<split>.npy                  float32 (N,)         SOH 标签
  is_valid_pretrain_<split>.npy  bool (N,)            预训练任务可用性
  group_ids_<split>.npy          int64 (N,)           同循环分组编号
  meta.json                      形状 / 样本数等元信息

与 partial_soh 版 build_cache 的区别：
  - 通道数 3 -> 4，增加归一化温度 T' = (T - 25) / 10；
  - 使用统一单位（I 安培、t 秒）后再提取充电段，与 dataset.py 一致；
  - SOH 标签列名是 soh（temperature_soh 口径）。

用法
----
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/build_cache.py --split train
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/build_cache.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# 把 temperature_soh/DataLoader 放进 sys.path，复用 MAT 读取与插值模块。
ROOT = Path(__file__).resolve().parents[3]
DL_DIR = ROOT / "src" / "temperature_soh" / "DataLoader"
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from mat_io import (  # noqa: E402
    convert_cycle_to_unified,
    discover_batch_files,
    read_raw_cycle_from_file,
)
from segments import NOMINAL_CAPACITY_AH, interpolate_segment  # noqa: E402

# 温度归一化常数（与 Trainer/dataset.py 一致）。
TEMP_CENTER_C = 25.0
TEMP_SCALE_C = 10.0

# 未来 7% 预测窗的等距点数：7% * 1.1 Ah / (0.0022 Ah/步) = 35 -> 36 点。
PRED_WINDOW_POINTS = 36


def _cell_index_from_id(cell_id: str) -> int:
    """从 cell_id（如 2017-06-30_c000）解析批次内 0-based 下标。"""
    suffix = cell_id.rsplit("_", 1)[1]
    if not suffix.startswith("c"):
        raise ValueError(f"cell_id 格式错误: {cell_id}")
    return int(suffix[1:])


def _batch_from_id(cell_id: str) -> str:
    """从 cell_id 中提取批次名（如 2017-06-30）。"""
    return cell_id.rsplit("_", 1)[0]


def _read_charge_curve(
    handle: h5py.File, cell_id: str, cycle_index: int
) -> dict[str, np.ndarray]:
    """读取一个循环的充电段曲线（统一单位），返回 dict[t/V/I/Qc/T]。

    与 Trainer/dataset.py 的 _read_charge_curve 逻辑保持一致：
      1. 读原始 .mat（I 是 C-rate、t 是分钟）；
      2. 统一单位：I ×1.1 -> 安培，t ×60 -> 秒；
      3. 只保留充电段（I > 0）。
    """
    batch = _batch_from_id(cell_id)
    cell_index = _cell_index_from_id(cell_id)
    raw = read_raw_cycle_from_file(handle, cell_index, int(cycle_index))
    cycle = convert_cycle_to_unified(raw, cycle_number=int(cycle_index))

    charge = {
        "t": np.asarray(cycle["time_in_s"], dtype=float),
        "V": np.asarray(cycle["voltage_in_V"], dtype=float),
        "I": np.asarray(cycle["current_in_A"], dtype=float),
        "Qc": np.asarray(cycle["charge_capacity_in_Ah"], dtype=float),
        "T": np.asarray(cycle["temperature_in_C"], dtype=float),
    }
    mask = charge["I"] > 0.0
    for key in charge:
        charge[key] = charge[key][mask]
    if charge["V"].size < 2:
        raise ValueError(f"{cell_id} cycle {cycle_index} 充电段点数不足")
    return charge


def _stack_4channels(seg: dict[str, np.ndarray]) -> np.ndarray:
    """把插值结果按 [I, V, Q, T'] 顺序堆叠成 (n_points, 4)。"""
    return np.stack(
        [
            seg["I"],
            seg["V"],
            seg["capacity"],
            (seg["T"] - TEMP_CENTER_C) / TEMP_SCALE_C,
        ],
        axis=1,
    ).astype(np.float32)


def build_split(
    index_path: Path,
    mat_dir: Path,
    cache_dir: Path,
    split: str,
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
) -> None:
    """为一个划分（train / test）构建 4 通道磁盘缓存。"""
    t0 = time.perf_counter()
    index = pd.read_parquet(index_path)
    index = index[(index["split"] == split) & (index["is_valid_soh"])].copy()
    index.reset_index(drop=True, inplace=True)
    n = len(index)
    if n == 0:
        raise ValueError(f"split={split} 没有有效样本")

    # 同循环分组编号、预训练可用性、SOH 标签。
    group_ids = (
        index.groupby(["cell_id", "cycle_index"], sort=False)
        .ngroup()
        .to_numpy(dtype=np.int64)
    )
    is_valid_pretrain = index["is_valid_pretrain"].to_numpy(dtype=bool)
    y = index["soh"].to_numpy(dtype=np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    # 预分配输入张量：按索引行顺序写入。
    x_path = cache_dir / f"X_{split}.npy"
    x = np.memmap(str(x_path), dtype=np.float32, mode="w+", shape=(n, 101, 4))

    # 未来 7% 预测窗：只有 is_valid_pretrain 的行有效，其余填 0。
    x_future_path = cache_dir / f"X_future_{split}.npy"
    x_future = np.memmap(
        str(x_future_path), dtype=np.float32, mode="w+",
        shape=(n, PRED_WINDOW_POINTS, 4),
    )
    x_future[:] = 0.0

    files = discover_batch_files(mat_dir)
    handles = {batch: h5py.File(str(path), "r") for batch, path in files.items()}

    print(f"[build_cache] split={split}, 有效片段数={n:,}，开始构建 ...", flush=True)
    pos = 0
    n_groups = 0
    for (cell_id, cycle_index), rows in index.groupby(
        ["cell_id", "cycle_index"], sort=False
    ):
        batch = _batch_from_id(cell_id)
        if batch not in handles:
            raise FileNotFoundError(f"找不到批次文件: {batch}")

        # 一个循环只读一次 MAT、提取一次充电段；该循环所有片段共用。
        charge = _read_charge_curve(handles[batch], cell_id, int(cycle_index))

        for row in rows.itertuples(index=False):
            seg = interpolate_segment(
                charge,
                start_ah=float(row.start_ah),
                end_ah=float(row.end_ah),
                nominal_capacity=nominal_capacity,
            )
            x[pos] = _stack_4channels(seg)
            if row.is_valid_pretrain:
                seg_future = interpolate_segment(
                    charge,
                    start_ah=float(row.pred_start_ah),
                    end_ah=float(row.pred_end_ah),
                    nominal_capacity=nominal_capacity,
                )
                x_future[pos] = _stack_4channels(seg_future)
            pos += 1

        n_groups += 1
        if n_groups % 5000 == 0:
            print(
                f"[build_cache] 进度: 已处理 {pos:,}/{n:,} 片段 "
                f"({pos / n * 100:.1f}%)",
                flush=True,
            )

    for handle in handles.values():
        try:
            handle.close()
        except Exception:
            pass

    x.flush()   # 确保 memmap 写回磁盘
    x_future.flush()
    np.save(cache_dir / f"y_{split}.npy", y)
    np.save(cache_dir / f"is_valid_pretrain_{split}.npy", is_valid_pretrain)
    np.save(cache_dir / f"group_ids_{split}.npy", group_ids)

    meta_path = cache_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta[f"shape_{split}"] = [int(n), 101, 4]
    meta[f"shape_future_{split}"] = [int(n), PRED_WINDOW_POINTS, 4]
    meta[f"n_{split}"] = int(n)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(
        f"[build_cache] split={split} 完成：{n:,} 片段，"
        f"耗时 {elapsed / 60:.1f} 分钟，写入 {x_path}"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "data" / "processed" / "temperature_soh" / "segment_index.parquet",
    )
    parser.add_argument(
        "--mat-dir",
        type=Path,
        default=ROOT / "data" / "external" / "matr",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "temperature_soh" / "cache",
    )
    parser.add_argument("--split", choices=("train", "test"), required=True)
    args = parser.parse_args()

    build_split(args.index, args.mat_dir, args.cache_dir, args.split)


if __name__ == "__main__":
    main()
