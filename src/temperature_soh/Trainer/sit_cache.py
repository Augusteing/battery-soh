"""SIT 片段缓存构建：把 xlsx 片段一次性物化为 memmap，加速 few-shot 迭代。

背景
----
SIT 数据是逐个 xlsx 存储的（20 只电池 × 约 700 循环 × 2 sheet），
每次评估/微调都要重新读 xlsx + 插值，单只电池约 3~4 分钟，
17 只电池评估要 1 小时。片段是静态数据，可以一次性缓存。

产物（data/processed/sit_cache/）：
  X.npy            float32 (N, 101, 3)  归一化输入 [I(C-rate), V, Q(SOC)]
  y.npy            float32 (N,)         SOH = Qc / Qc_max
  cell_ids.npy     str (N,)             每行所属电池
  temp_features.npy float32 (N, 12)     温度曲线形状特征（温度模块用）
  cycle_ids.npy    int64 (N,)           每行所属循环号（物理约束用）
  meta.json        每电池样本数、总样本数

用法：
```powershell
# 构建全部 20 只（约 70 分钟）
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/sit_cache.py

# 只构建指定电池（快速实验，约 3~4 分钟/只）
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/sit_cache.py --cells 001-1,001-2,002-1
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TR_DIR = Path(__file__).resolve().parent
DL_DIR = ROOT / "src" / "temperature_soh" / "DataLoader"
for d in (TR_DIR, DL_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from finetune_sit import build_cell_samples_full  # noqa: E402
from sit_io import DEFAULT_SIT_DIR, discover_sit_cells  # noqa: E402

DEFAULT_CACHE_DIR = ROOT / "data" / "processed" / "sit_cache"


def build_cache(
    cell_ids: list[str],
    data_dir: Path,
    cache_dir: Path,
    rebuild: bool = False,
) -> None:
    """为指定电池构建片段缓存（追加模式：已存在的电池跳过）。

    rebuild=True 时忽略已有缓存，从空数组开始重建全部（用于缓存格式升级）。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 已有缓存信息
    meta_path = cache_dir / "meta.json"
    existing: dict = {}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
    done = set(existing.get("cells", []))
    if rebuild:
        done = set()
    todo = [c for c in cell_ids if c not in done]
    if not todo:
        print("所有指定电池已在缓存中，跳过")
        return

    # 追加写入：读已有数组 + 新数组拼接后重写。
    x_path = cache_dir / "X.npy"
    y_path = cache_dir / "y.npy"
    cell_path = cache_dir / "cell_ids.npy"
    temp_path = cache_dir / "temp_features.npy"
    cyc_path = cache_dir / "cycle_ids.npy"
    if rebuild:
        x_all = np.zeros((0, 101, 3), np.float32)
        y_all = np.zeros((0,), np.float32)
        cell_all = np.zeros((0,), dtype=object)
        temp_all = np.zeros((0, 12), np.float32)
        cyc_all = np.zeros((0,), np.int64)
        # 直接覆盖旧文件，避免旧格式残留。
        np.save(x_path, x_all)
        np.save(y_path, y_all)
        np.save(cell_path, cell_all)
        np.save(temp_path, temp_all)
        np.save(cyc_path, cyc_all)
        meta_path.write_text(
            json.dumps({"cells": [], "n": 0}, ensure_ascii=False), encoding="utf-8"
        )
        print("[sit_cache] 重建模式：清空旧缓存", flush=True)
    else:
        x_all = np.load(x_path) if x_path.exists() else np.zeros((0, 101, 3), np.float32)
        y_all = np.load(y_path) if y_path.exists() else np.zeros((0,), np.float32)
        cell_all = (
            np.load(cell_path, allow_pickle=True) if cell_path.exists()
            else np.zeros((0,), dtype=object)
        )
        temp_all = (
            np.load(temp_path) if temp_path.exists()
            else np.zeros((0, 12), np.float32)
        )
        cyc_all = (
            np.load(cyc_path) if cyc_path.exists() else np.zeros((0,), np.int64)
        )

    t0 = time.perf_counter()
    for cell_id in todo:
        try:
            x, y, temp, cyc = build_cell_samples_full(cell_id, data_dir)
            x_all = np.concatenate([x_all, x], axis=0)
            y_all = np.concatenate([y_all, y], axis=0)
            temp_all = np.concatenate([temp_all, temp], axis=0)
            cyc_all = np.concatenate([cyc_all, cyc], axis=0)
            cell_all = np.concatenate(
                [cell_all, np.full(len(x), cell_id, dtype=object)], axis=0
            )
            done.add(cell_id)
            # 每构建完一只立即落盘：中途失败只丢当前电池，不丢已完成的。
            np.save(x_path, x_all)
            np.save(y_path, y_all)
            np.save(cell_path, cell_all)
            np.save(temp_path, temp_all)
            np.save(cyc_path, cyc_all)
            meta = {
                "cells": sorted(done),
                "n": int(len(y_all)),
                "shape_x": [int(x_all.shape[0]), 101, 3],
                "shape_temp": [int(temp_all.shape[0]), 12],
            }
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"[sit_cache] {cell_id}: {len(x):,} 片段, "
                f"累计 {len(x_all):,} ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        except Exception as exc:
            print(f"[sit_cache] {cell_id} 构建失败（已保存之前完成的电池）: {exc}",
                  flush=True)
            continue

    print(f"[sit_cache] 完成: {len(x_all):,} 片段 -> {x_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SIT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cells", default=None,
                        help="只构建指定电池（逗号分隔）；默认全部 20 只")
    parser.add_argument("--rebuild", action="store_true",
                        help="忽略已有缓存，全量重建（缓存格式升级时用）")
    args = parser.parse_args()

    all_cells = discover_sit_cells(args.data_dir)["cell_id"].tolist()
    cells = (
        [c.strip() for c in args.cells.split(",") if c.strip()]
        if args.cells else all_cells
    )
    missing = [c for c in cells if c not in all_cells]
    if missing:
        raise ValueError(f"未知电池: {missing}")
    print(f"构建 {len(cells)} 只电池的缓存 ...")
    build_cache(cells, args.data_dir, args.cache_dir, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
