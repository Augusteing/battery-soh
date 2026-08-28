"""对比 Severson 与 SIT 在 Qc/Qc_max 口径下的分阶段退化循环数。

目的：检验"最大容量基准"下，两个数据集退化到各 SOH 阶段
（0.98/0.95/0.90/0.85/0.80/0.75）所用的循环数是否接近。
若 SIT 明显更快，说明除温度外还有协议/衰减速率差异。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.98, 0.95, 0.90, 0.85, 0.80, 0.75]


def _first_below(cycles: np.ndarray, soh: np.ndarray, t: float):
    idx = np.flatnonzero(soh < t)
    return int(cycles[idx[0]]) if len(idx) else float("nan")


def severson_stages() -> pd.DataFrame:
    labels = pd.read_parquet(
        ROOT / "data" / "processed" / "temperature_soh" / "soh_labels.parquet"
    )
    labels = labels[labels["is_valid_label"] & ~labels["is_bad_cycle"]].copy()
    rows = []
    for cell_id, g in labels.groupby("cell_id"):
        g = g.sort_values("cycle_index")
        qc = g["charge_capacity_ah"].to_numpy(float)
        qc_max = qc.max()
        soh = qc / qc_max
        cyc = g["cycle_index"].to_numpy(int)
        rows.append(
            {
                "cell_id": cell_id,
                "source": "Severson",
                "n": len(g),
                **{f"t{t:.2f}": _first_below(cyc, soh, t) for t in THRESHOLDS},
            }
        )
    return pd.DataFrame(rows)


def sit_stages() -> pd.DataFrame:
    sit_dir = ROOT / "data" / "external" / "SIT" / "Data" / "Cycle_Summary"
    temp_map = {
        "001-1": "ambient", "001-2": "ambient", "001-3": "ambient", "001-4": "ambient",
        "001-5": "ambient", "001-6": "ambient", "001-7": "ambient", "001-8": "ambient",
        "101-1": "ambient", "101-3": "ambient",
        "002-1": "chamber40", "002-2": "chamber40", "002-3": "chamber40", "002-4": "chamber40",
        "002-5": "chamber40", "002-7": "chamber40",
        "003-1": "chamber40", "003-3": "chamber40", "003-5": "chamber40", "003-7": "chamber40",
    }
    rows = []
    for f in sorted(os.listdir(sit_dir)):
        if not f.endswith(".csv"):
            continue
        cell = f.replace(".csv", "")
        d = pd.read_csv(sit_dir / f)
        ch = d[d["Type"] == "charge"].sort_values("Cycle").copy()
        qc = ch["Capacity_Ah"].to_numpy(float)
        qc_max = qc.max()
        soh = qc / qc_max
        cyc = ch["Cycle"].to_numpy(int)
        rows.append(
            {
                "cell_id": cell,
                "source": "SIT",
                "temp_group": temp_map.get(cell, "?"),
                "n": len(ch),
                **{f"t{t:.2f}": _first_below(cyc, soh, t) for t in THRESHOLDS},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sev = severson_stages()
    sit = sit_stages()

    print("== 各阶段首次达到的循环数（中位数）==")
    header = "数据集/组".ljust(14) + "".join(f"<{t:.2f}".rjust(8) for t in THRESHOLDS) + "  n"
    print(header)
    print("-" * len(header))
    for label, g in [
        ("Severson", sev),
        ("SIT 全部", sit),
        ("SIT 环境温", sit[sit["temp_group"] == "ambient"]),
        ("SIT 40°C", sit[sit["temp_group"] == "chamber40"]),
    ]:
        cells = []
        for t in THRESHOLDS:
            k = f"t{t:.2f}"
            col = g[k]
            if col.notna().any():
                cells.append(f"{int(col.median()):>8}")
            else:
                cells.append(f"{'-':>8}")
        vals = "".join(cells)
        print(label.ljust(14) + vals + f"  {len(g)}")

    print("\n== 达到比例（各阶段有电池跌破的占比）==")
    for label, g in [("Severson", sev), ("SIT 全部", sit)]:
        ratio = "".join(
            f"{g[f't{t:.2f}'].notna().mean()*100:.0f}%".rjust(8) for t in THRESHOLDS
        )
        print(label.ljust(14) + ratio)


if __name__ == "__main__":
    main()
