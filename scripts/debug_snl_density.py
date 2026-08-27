"""诊断 SNL 充电段点数过少的根因。

背景：check_segment_density.py 发现 SNL 充电段中位数只有 57 点，
不足以支撑 101 点插值。本脚本逐循环统计：
  - 循环总点数、I>0 / I<0 / I=0 的点数；
  - 充电段点数与容量跨度；
  - 充电段电流分布（判断是否被降采样/分段）。

运行：
```powershell
& "E:\conda\envs\battery-soh\python.exe" "scripts/debug_snl_density.py"
```
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "temperature_soh" / "DataLoader"))

from registry import DATASETS, list_cell_files, load_cell  # noqa: E402
from segments import extract_charge_curve  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    # 抽查 3 只 SNL：15C（1 只）、25C（1 只）、35C（1 只）
    files = list_cell_files(DATASETS["SNL"])
    probes = [files[0], files[2], files[12]]
    for path in probes:
        cell = load_cell(DATASETS["SNL"], path)
        by_cycle = {int(c["cycle_number"]): c for c in cell["cycle_data"]}
        print(f"\n===== {cell['cell_id']} =====")
        for cyc in (2, 100, 500, 1000, 2000, 3000):
            if cyc not in by_cycle:
                continue
            c = by_cycle[cyc]
            i = np.asarray(c["current_in_A"], dtype=float)
            qc = np.asarray(c["charge_capacity_in_Ah"], dtype=float)
            n_pos = int((i > 0).sum())
            n_neg = int((i < 0).sum())
            n_zero = int((i == 0).sum())
            ch = extract_charge_curve(c)
            i_pos = np.asarray(ch["I"], dtype=float)
            uniq = np.unique(np.round(i_pos, 3))
            print(
                f"cycle {cyc:4d}: total={len(i):5d} "
                f"I>0={n_pos:5d} I<0={n_neg:5d} I=0={n_zero:5d} "
                f"| charge_pts={len(ch['V']):5d} "
                f"Qc={ch['Qc'].min():.3f}~{ch['Qc'].max():.3f} "
                f"| I档位={uniq[:6]}"
            )


if __name__ == "__main__":
    main()
