"""检查三个数据集充电片段的采样密度与插值质量。

回答一个问题：HUST / SNL / MATR 的充电曲线是否足够密集，
支撑“20% 容量窗口、每 1% 插 5 点（共 101 点）”的插值？

判断标准：
  1. 充电段原始点数：越多越好（至少明显多于 101）；
  2. Qc 覆盖范围：要覆盖 0.55 Ah 起点 + 0.22 Ah 窗口（即约 0.77 Ah）；
  3. Qc 单调性：充电容量必须严格递增，否则 np.interp 会失真；
  4. 有效片段比例：is_valid_soh / is_valid_pretrain 各占多少。

运行：
```powershell
& "E:\conda\envs\battery-soh\python.exe" "scripts/check_segment_density.py"
```
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 让脚本能 import temperature_soh 的模块。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "temperature_soh" / "DataLoader"))

from mat_io import (  # noqa: E402
    discover_batch_files,
    load_raw_cycle,
    load_unified_cycle,
)
from registry import DATASETS, list_cell_files, load_cell  # noqa: E402
from segments import (  # noqa: E402
    build_segment_index_for_cycle,
    extract_charge_curve,
)

# 每个数据集抽查的循环：早期（2）、中期（1/4、1/2、3/4）和晚期。
PROBE_CYCLES = (2, 100, 500, 1000, 1500, 2000, 2500, 3000)


def _probe_pkl(spec, max_cells: int) -> pd.DataFrame:
    """抽查 pkl 数据集（SNL / HUST）的若干电池与循环。"""
    rows: list[dict] = []
    cells = list_cell_files(spec)[:max_cells]
    for path in cells:
        cell = load_cell(spec, path)
        by_cycle = {int(c["cycle_number"]): c for c in cell["cycle_data"]}
        n_cycles = len(cell["cycle_data"])
        for cyc in PROBE_CYCLES:
            if cyc not in by_cycle:
                continue
            cycle = by_cycle[cyc]
            charge = extract_charge_curve(cycle)
            q = np.asarray(charge["Qc"], dtype=float)
            dq = np.diff(q)
            table = build_segment_index_for_cycle(
                cycle,
                cell_id=cell["cell_id"],
                cycle_index=cyc,
                temperature_c=30.0,
            )
            rows.append(
                {
                    "dataset": spec.name,
                    "cell_id": cell["cell_id"],
                    "cycle_index": cyc,
                    "cycle_ratio": cyc / n_cycles,
                    "n_charge_points": int(len(charge["V"])),
                    "q_min_ah": float(q.min()),
                    "q_max_ah": float(q.max()),
                    "q_nonmonotonic_frac": float(np.mean(dq <= 0)),
                    "valid_soh": int(table["is_valid_soh"].sum()),
                    "valid_pretrain": int(table["is_valid_pretrain"].sum()),
                    "total": len(table),
                }
            )
    return pd.DataFrame(rows)


def _probe_matr(max_cells: int) -> pd.DataFrame:
    """抽查 MATR 的若干电池与循环（与 pkl 相同的统计口径）。"""
    rows: list[dict] = []
    batches = discover_batch_files()
    for batch_name, path in list(batches.items())[:1]:
        for cell_idx in range(max_cells):
            # 先探测该电池总循环数（cycle 2 的 n_cycles 字段）。
            probe = load_raw_cycle(path, cell_index=cell_idx, cycle_index=2)
            n_cycles = int(probe["n_cycles"])
            for cyc in PROBE_CYCLES:
                if cyc > n_cycles:
                    continue
                cycle = load_unified_cycle(path, cell_index=cell_idx, cycle_index=cyc)
                charge = extract_charge_curve(cycle)
                q = np.asarray(charge["Qc"], dtype=float)
                dq = np.diff(q)
                table = build_segment_index_for_cycle(
                    cycle,
                    cell_id=f"{batch_name}_c{cell_idx:03d}",
                    cycle_index=cyc,
                    temperature_c=30.0,
                )
                rows.append(
                    {
                        "dataset": "MATR",
                        "cell_id": f"{batch_name}_c{cell_idx:03d}",
                        "cycle_index": cyc,
                        "cycle_ratio": cyc / n_cycles,
                        "n_charge_points": int(len(charge["V"])),
                        "q_min_ah": float(q.min()),
                        "q_max_ah": float(q.max()),
                        "q_nonmonotonic_frac": float(np.mean(dq <= 0)),
                        "valid_soh": int(table["is_valid_soh"].sum()),
                        "valid_pretrain": int(table["is_valid_pretrain"].sum()),
                        "total": len(table),
                    }
                )
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame, name: str) -> None:
    """打印一个数据集的汇总。"""
    if df.empty:
        print(f"[{name}] 无样本")
        return
    print(f"\n===== {name}（{len(df)} 个抽样循环）=====")
    print(f"充电段点数: min={df['n_charge_points'].min()}, "
          f"median={df['n_charge_points'].median():.0f}, "
          f"max={df['n_charge_points'].max()}")
    print(f"Qc 范围: min={df['q_min_ah'].min():.4f}, "
          f"max={df['q_max_ah'].max():.4f} Ah")
    print(f"Qc 非单调比例（应≈0）: "
          f"max={df['q_nonmonotonic_frac'].max():.4f}")
    print(f"有效 soh 片段: "
          f"{df['valid_soh'].sum()}/{df['total'].sum()}"
          f"（{(df['valid_soh'].sum()/df['total'].sum()*100):.1f}%）")
    print(f"有效 pretrain 片段: "
          f"{df['valid_pretrain'].sum()}/{df['total'].sum()}"
          f"（{(df['valid_pretrain'].sum()/df['total'].sum()*100):.1f}%）")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    snl = _probe_pkl(DATASETS["SNL"], max_cells=18)
    hust = _probe_pkl(DATASETS["HUST"], max_cells=10)
    matr = _probe_matr(max_cells=8)

    _summary(snl, "SNL")
    _summary(hust, "HUST")
    _summary(matr, "MATR")

    # 插值质量：以窗口内原始点数估计。
    # 窗口 0.22Ah 占 Qc 跨度（约 1.0Ah）的 22%，窗口内原始点数 ≈ 总数 × 22%。
    print("\n===== 插值密度估计（20% 窗口内原始点数 vs 插值 101 点）=====")
    for name, df in (("SNL", snl), ("HUST", hust), ("MATR", matr)):
        if df.empty:
            continue
        span = (df["q_max_ah"] - df["q_min_ah"]).median()
        in_window = df["n_charge_points"] * (0.22 / span)
        print(f"{name}: 窗口内原始点数约 "
              f"{in_window.median():.0f}（中位）/{in_window.min():.0f}（最少）"
              f" -> 插值 101 点"
              f"{'，充足' if in_window.median() > 200 else '，偏少！'}")


if __name__ == "__main__":
    main()
