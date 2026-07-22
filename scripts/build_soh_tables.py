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
    print(table.groupby("cell_id")["soh"].last().describe())


if __name__ == "__main__":
    main()
