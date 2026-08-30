"""缓存第 4 通道转换：T'（温度）-> tanh(dV/dQ)。

导师方案：把平滑 dV/dQ（S-G deriv=1）经 tanh 压制到 [-1,1]，作为
第 4 个物理通道（I, V, Q, dV/dQ）送入一阶段 LSTM，并在 Severson
123 只电池上全量重训。本脚本只做"缓存转换"，不碰模型与训练：

  - Severson（--target severson）：
      现有 cache/X_{split}.npy 是原始 memmap (N,101,4)，
      第 4 列是归一化温度 T'=(T-25)/10。转换后第 4 列 = tanh(dV/dQ)，
      其余文件（y / is_valid_pretrain / group_ids / temp_scalars /
      X_future / meta.json）原样复制到新目录 cache_dvdq/。
      X_future 只作为电压监督目标（取第 1 列 V），不需要改。

  - SIT（--target sit）：
      现有 sit_cache/X.npy 是 np.save 格式 (N,101,3) [I,V,Q]。
      转换后追加第 4 列 tanh(dV/dQ)，新目录 sit_cache_dvdq/。

dV/dQ 计算口径（与 DataLoader/dvdq_features.py 一致）：
  等容量网格步长 0.002 SOC，savgol_filter(deriv=1, delta=0.002)，
  即"拟合局部多项式的同时输出解析导数"，先平滑再差分会引入两步误差。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]

SOC_STEP = 0.2 / 100.0          # 0.002 SOC / 网格点
SG_WINDOW = 21                  # S-G 窗口 ≈ 0.04 SOC
SG_POLYORDER = 3
CHUNK = 250_000                 # 分块行数，控制峰值内存


def dvdq_channel(v: np.ndarray) -> np.ndarray:
    """从 (..., 101) 电压曲线计算 tanh(dV/dSOC)，返回同形状 float32。"""
    dvdq = savgol_filter(
        v, window_length=SG_WINDOW, polyorder=SG_POLYORDER,
        deriv=1, delta=SOC_STEP, axis=-1,
    )
    return np.tanh(dvdq).astype(np.float32)


def convert_severson(src: Path, dst: Path) -> None:
    """Severson 缓存：第 4 列 T' -> tanh(dV/dQ)，其余文件复制。"""
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    for split in ("train", "test"):
        n = int(meta[f"n_{split}"])
        shape = (n, 101, 4)
        src_x = np.memmap(str(src / f"X_{split}.npy"), dtype=np.float32,
                          mode="r", shape=shape)
        dst_x = np.memmap(str(dst / f"X_{split}.npy"), dtype=np.float32,
                          mode="w+", shape=shape)
        t0 = time.perf_counter()
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            chunk = np.asarray(src_x[start:stop]).copy()
            chunk[:, :, 3] = dvdq_channel(chunk[:, :, 1])  # V 通道 -> dV/dQ
            dst_x[start:stop] = chunk
        dst_x.flush()
        print(f"[severson] {split}: {n:,} 片段，"
              f"通道 4 -> tanh(dV/dQ) 完成 ({time.perf_counter()-t0:.0f}s)")
        # 其余文件原样复制（X_future 只作监督目标，无需改）。
        for name in (
            f"y_{split}.npy",
            f"is_valid_pretrain_{split}.npy",
            f"group_ids_{split}.npy",
            f"temp_scalars_{split}.npy",
            f"X_future_{split}.npy",
        ):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
    (dst / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[severson] 新缓存写入: {dst}")


def convert_sit(src: Path, dst: Path) -> None:
    """SIT 缓存：3 通道 [I,V,Q] -> 4 通道 [I,V,Q,dV/dQ]。"""
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    n = int(meta["n"])
    src_x = np.load(src / "X.npy", mmap_mode="r")  # (n, 101, 3)
    # np.lib.format.open_memmap 写入标准 .npy 头，兼容后续 np.load(mmap)。
    dst_x = np.lib.format.open_memmap(
        str(dst / "X.npy"), mode="w+", dtype=np.float32, shape=(n, 101, 4)
    )
    t0 = time.perf_counter()
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        chunk = np.asarray(src_x[start:stop]).copy()
        dvdq = dvdq_channel(chunk[:, :, 1])  # V 通道 -> dV/dQ
        dst_x[start:stop] = np.concatenate([chunk, dvdq[:, :, None]], axis=-1)
    dst_x.flush()
    print(f"[sit] {n:,} 片段 -> 4 通道完成 ({time.perf_counter()-t0:.0f}s)")
    for name in ("y.npy", "cell_ids.npy", "temp_features.npy", "cycle_ids.npy"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    meta["shape_x"] = [n, 101, 4]
    (dst / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[sit] 新缓存写入: {dst}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("severson", "sit"), required=True)
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--dst", type=Path, default=None)
    args = parser.parse_args()

    if args.target == "severson":
        convert_severson(
            args.src or ROOT / "data" / "processed" / "temperature_soh" / "cache",
            args.dst or ROOT / "data" / "processed" / "temperature_soh" / "cache_dvdq",
        )
    else:
        convert_sit(
            args.src or ROOT / "data" / "processed" / "sit_cache",
            args.dst or ROOT / "data" / "processed" / "sit_cache_dvdq",
        )


if __name__ == "__main__":
    main()
