"""诊断 SOH 标签表中的异常值（临时调试脚本）。

找出每个数据集的 SOH 异常高/低样本，打印 cell/cycle/容量，
用于决定是否需要额外的数据清洗规则。
"""

from __future__ import annotations

import sys

import pandas as pd

LABELS = r"C:\Users\PLUTO\Desktop\battery-soh\data\processed\temperature_soh\soh_labels.parquet"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    t = pd.read_parquet(LABELS)
    print("总行数:", len(t))

    for ds in ("MATR", "SNL", "HUST"):
        g = t[t["dataset"] == ds]
        print()
        print("====", ds, "====")
        print(
            "soh < 0.5 数量:", int((g["soh"] < 0.5).sum()),
            " soh > 1.3 数量:", int((g["soh"] > 1.3).sum()),
        )

        high = g[g["soh"] > 1.3].sort_values("soh", ascending=False).head(10)
        low = g[g["soh"] < 0.5].sort_values("soh").head(10)

        for label, df in (("HIGH", high), ("LOW", low)):
            if df.empty:
                continue
            print(f"  -- {label} --")
            for _, r in df.iterrows():
                print(
                    f"    cell={r['cell_id']} cycle={r['cycle_index']} "
                    f"Qc={r['charge_capacity_ah']:.4f} "
                    f"soh={r['soh']:.3f} bad_cycle={r['is_bad_cycle']}"
                )


if __name__ == "__main__":
    main()
