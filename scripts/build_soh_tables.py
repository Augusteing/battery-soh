"""汇总已下载数据，生成统一的 SOH 表到 data/processed/。

用法:
    python scripts/build_soh_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from battery_soh.data.stanford_dynamic import build_soh_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    summary_dir = ROOT / "data/external/stanford_dynamic/Publishing_data"
    out = ROOT / "data/processed/stanford_soh_table.parquet"
    table = build_soh_table(summary_dir, out)
    n_cells = table["cell_id"].nunique()
    print(f"stanford: {n_cells} 只电池, {len(table)} 行 -> {out}")
    regular = table[~table["is_diagnostic"]]
    diag = table[table["is_diagnostic"]]
    print(f"常规循环 {len(regular)} 行, 诊断循环 {len(diag)} 行")
    print("\n各电池最终常规循环 SOH 分布:")
    print(regular.groupby("cell_id")["soh"].last().describe())
    print("\n按协议族的循环数与最终 SOH:")
    fam = regular.groupby("protocol_family").agg(
        cells=("cell_id", "nunique"),
        cycles_median=("cycle_index", "median"),
        final_soh_mean=("soh", lambda s: regular.loc[s.index].groupby("cell_id")["soh"].last().mean()),
    )
    print(fam)


if __name__ == "__main__":
    main()
