"""对比 Severson 与 SIT 的 SOH 轨迹（相对自身 cycle 2 口径）。

目的：
  1. 检验用户观察：SIT 电池的 SOH 走向是否与 Severson 不同；
  2. 对比两种口径：
       - 绝对口径：Qc / 标称（Severson 1.1 / SIT 50）；
       - 相对口径：Qc_k / Qc_2（各自 cycle 2 为基准，起始 ≈ 1.0）。

输出：每只电池的起始 SOH、终值 SOH、衰减到 0.95/0.90/0.85 的循环数、
      以及相对口径下的典型轨迹对比（打印统计，不画图）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def severson_relative_trajectory() -> pd.DataFrame:
    """从 temperature_soh 的标签表读取 Severson 每循环充电容量，算相对口径。"""
    labels = pd.read_parquet(
        ROOT / "data" / "processed" / "temperature_soh" / "soh_labels.parquet"
    )
    labels = labels[labels["is_valid_label"] & ~labels["is_bad_cycle"]].copy()
    rows = []
    for cell_id, g in labels.groupby("cell_id"):
        g = g.sort_values("cycle_index")
        qc = g["charge_capacity_ah"].to_numpy(float)
        qc2 = qc[1] if len(qc) > 1 else qc[0]  # cycle 2 基准
        rows.append(
            {
                "cell_id": cell_id,
                "source": "Severson",
                "soh_abs_start": qc[0] / 1.1,
                "soh_rel_start": qc[0] / qc2,
                "soh_abs_end": qc[-1] / 1.1,
                "soh_rel_end": qc[-1] / qc2,
                "n_cycles": len(g),
                "cyc_to_095": _first_below(g["cycle_index"].to_numpy(), qc / qc2, 0.95),
                "cyc_to_090": _first_below(g["cycle_index"].to_numpy(), qc / qc2, 0.90),
                "cyc_to_085": _first_below(g["cycle_index"].to_numpy(), qc / qc2, 0.85),
            }
        )
    return pd.DataFrame(rows)


def sit_trajectory() -> pd.DataFrame:
    """从 SIT Cycle_Summary 读取每循环充/放电容量，算相对口径。"""
    sit_dir = ROOT / "data" / "external" / "SIT" / "Data" / "Cycle_Summary"
    rows = []
    for f in sorted(os.listdir(sit_dir)):
        if not f.endswith(".csv"):
            continue
        d = pd.read_csv(sit_dir / f)
        ch = d[d["Type"] == "charge"].sort_values("Cycle").copy()
        dis = d[d["Type"] == "discharge"].sort_values("Cycle").copy()
        qc = ch["Capacity_Ah"].to_numpy(float)
        qd = dis["Capacity_Ah"].to_numpy(float)
        qc2 = qc[1] if len(qc) > 1 else qc[0]
        rows.append(
            {
                "cell_id": f.replace(".csv", ""),
                "source": "SIT",
                "soh_abs_start": qc[0] / 50.0,
                "soh_rel_start": qc[0] / qc2,
                "soh_abs_end": qc[-1] / 50.0,
                "soh_rel_end": qc[-1] / qc2,
                "n_cycles": len(ch),
                "cyc_to_095": _first_below(ch["Cycle"].to_numpy(), qc / qc2, 0.95),
                "cyc_to_090": _first_below(ch["Cycle"].to_numpy(), qc / qc2, 0.90),
                "cyc_to_085": _first_below(ch["Cycle"].to_numpy(), qc / qc2, 0.85),
            }
        )
    return pd.DataFrame(rows)


def _first_below(cycles: np.ndarray, soh: np.ndarray, threshold: float):
    """返回 SOH 首次低于 threshold 的循环号；未达到则 NaN。"""
    idx = np.flatnonzero(soh < threshold)
    return int(cycles[idx[0]]) if len(idx) else float("nan")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sev = severson_relative_trajectory()
    sit = sit_trajectory()
    df = pd.concat([sev, sit], ignore_index=True)

    for src in ("Severson", "SIT"):
        g = df[df["source"] == src]
        print(f"===== {src}（{len(g)} 只电池）=====")
        print(f"  绝对口径起始 SOH: {g['soh_abs_start'].median():.3f} "
              f"({g['soh_abs_start'].min():.3f}~{g['soh_abs_start'].max():.3f})")
        print(f"  相对口径起始 SOH: {g['soh_rel_start'].median():.3f} "
              f"({g['soh_rel_start'].min():.3f}~{g['soh_rel_start'].max():.3f})")
        print(f"  相对口径终值 SOH: {g['soh_rel_end'].median():.3f} "
              f"({g['soh_rel_end'].min():.3f}~{g['soh_rel_end'].max():.3f})")
        print(f"  循环数中位: {g['n_cycles'].median():.0f}")
        for col, label in [
            ("cyc_to_095", "首次跌破 0.95"),
            ("cyc_to_090", "首次跌破 0.90"),
            ("cyc_to_085", "首次跌破 0.85"),
        ]:
            v = g[col].dropna()
            if len(v):
                print(f"  {label}: 中位 {v.median():.0f} 循环 (达到比例 {len(v)/len(g)*100:.0f}%)")
            else:
                print(f"  {label}: 无电池达到")
        print()


if __name__ == "__main__":
    main()
