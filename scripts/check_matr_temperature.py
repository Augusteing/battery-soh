"""检查 MATR（Severson）数据的温度波动情况。

目的：回答“只加入温度曲线”是否可行——温度曲线里到底有没有信号。
统计三个批次各抽几只电池，每只电池看 3 个循环（早期 / 中期 / 末期）：
  - 单循环内 T 的 min / max / mean（波动幅度）；
  - 充电段（I>0）与放电段（I<0）的 T 差异；
  - 老化后（末期循环）温度是否整体抬升（内阻增大的焦耳热）。

用法：
    & "E:\conda\envs\battery-soh\python.exe" scripts/check_matr_temperature.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# 复用 temperature_soh 的读取函数
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "temperature_soh" / "DataLoader"))

from mat_io import discover_batch_files, read_raw_cycle_from_file  # noqa: E402


def summarize_temperature(
    f: h5py.File,
    cell_index: int,
    cycle_index: int,
) -> dict[str, float] | None:
    """统计一个循环的温度：整体、充电段、放电段。"""
    try:
        raw = read_raw_cycle_from_file(f, cell_index, cycle_index)
    except (IndexError, KeyError, ValueError):
        return None

    T = raw["T"]
    I = raw["I"]
    n = len(T)
    if n == 0:
        return None

    charge = I > 0.01
    discharge = I < -0.01

    def stats(mask: np.ndarray) -> tuple[float, float, float, int]:
        sub = T[mask]
        if sub.size == 0:
            return (float("nan"), float("nan"), float("nan"), 0)
        return (
            float(sub.min()),
            float(sub.max()),
            float(sub.mean()),
            int(sub.size),
        )

    tmin, tmax, tmean, _ = stats(np.ones(n, dtype=bool))
    cmin, cmax, cmean, cn = stats(charge)
    dmin, dmax, dmean, dn = stats(discharge)

    return {
        "cycle": int(cycle_index),
        "n": n,
        "t_min": tmin,
        "t_max": tmax,
        "t_mean": tmean,
        "t_span": tmax - tmin,
        "charge_n": cn,
        "charge_mean": cmean,
        "charge_max": cmax,
        "charge_min": cmin,
        "discharge_n": dn,
        "discharge_mean": dmean,
        "discharge_max": dmax,
    }


def main() -> None:
    batch_files = discover_batch_files()
    print(f"发现批次: {list(batch_files)}")
    print()

    # 每个批次抽 4 只电池：选中间偏前的 4 只，覆盖不同协议
    for batch_name, mat_path in batch_files.items():
        print(f"===== 批次 {batch_name} =====")
        with h5py.File(str(mat_path), "r") as f:
            batch = f["batch"]
            n_cells = int(batch["cycles"].shape[0])
            cell_indices = [0, n_cells // 3, (2 * n_cells) // 3, n_cells - 1]

            for ci in cell_indices:
                # 查该电池循环数
                cell_ref = batch["cycles"][ci, 0]
                cell = f[cell_ref]
                n_cycles = int(np.asarray(cell["V"]).shape[0])
                cycle_samples = [2, max(3, n_cycles // 2), n_cycles]

                print(f"  电池 {ci}（共 {n_cycles} 循环）:")
                for cy in cycle_samples:
                    s = summarize_temperature(f, ci, cy)
                    if s is None:
                        continue
                    print(
                        f"    cycle {s['cycle']:>5d}: "
                        f"T ∈ [{s['t_min']:.1f}, {s['t_max']:.1f}] "
                        f"均值 {s['t_mean']:.1f}°C，跨度 {s['t_span']:.1f}°C | "
                        f"充电段 {s['charge_min']:.1f}→{s['charge_max']:.1f}°C "
                        f"(跨度 {s['charge_max'] - s['charge_min']:.1f}°C，"
                        f"均值 {s['charge_mean']:.1f}°C) "
                        f"放电段均值 {s['discharge_mean']:.1f}°C"
                    )
        print()


if __name__ == "__main__":
    main()
