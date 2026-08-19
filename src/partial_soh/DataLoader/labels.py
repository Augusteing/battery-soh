"""partial_soh SOH 标签模块。

本模块只负责从统一 SOH 表中生成两种标签：

1. `soh_nominal = Q_charge / 1.1`
   这是 Scientific Reports 2026 论文使用的口径。论文原文：
   "SOH is defined as the ratio between the current chargeable capacity
   and the nominal capacity"，即“可充电容量 / 额定容量”。
   对应统一 SOH 表中的 charge_capacity（QCharge）。

2. `soh_q2 = Q_discharge / Q_discharge(cycle 2)`
   这是之前 World Model 复现使用的口径，保留用于和旧结果比较。

它不读取 MAT，不切片段，也不做训练/测试划分。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DL_DIR = Path(__file__).resolve().parent
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from quality import apply_exclusions  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data" / "processed" / "matr_soh_table.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "partial_soh_labels.parquet"

NOMINAL_CAPACITY_AH = 1.1
REFERENCE_CYCLE = 2


def build_labels(
    soh_table: pd.DataFrame,
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
    reference_cycle: int = REFERENCE_CYCLE,
    exclude_cycle_one: bool = True,
) -> pd.DataFrame:
    """把统一 SOH 表转换为 partial_soh 需要的标签表。

    输入至少需要这些列：
      cell_id, cycle_index, discharge_capacity, charge_capacity, ir, batch, policy, cycle_life

    输出新增：
      soh_nominal : 可充电容量 / 额定容量（论文口径）；
      soh_q2      : 放电容量 / 每只电池 reference_cycle 的放电容量。
    """
    required = {
        "cell_id",
        "cycle_index",
        "discharge_capacity",
        "charge_capacity",
        "ir",
        "batch",
        "policy",
        "cycle_life",
    }
    missing = required - set(soh_table.columns)
    if missing:
        raise KeyError(f"SOH 表缺少字段: {sorted(missing)}")

    table = soh_table.copy()
    if exclude_cycle_one:
        table = table[table["cycle_index"] != 1].copy()

    table = table.sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)

    # 论文口径：Q_charge / 1.1 Ah（current chargeable capacity / nominal capacity）。
    table["soh_nominal"] = table["charge_capacity"] / nominal_capacity

    # 旧复现口径：Qd / Q(2)。
    refs: dict[str, float] = {}
    for cell_id, group in table.groupby("cell_id", sort=False):
        ref = group.loc[group["cycle_index"] == reference_cycle, "discharge_capacity"]
        if ref.empty:
            refs[cell_id] = np.nan
        else:
            refs[cell_id] = float(ref.iloc[0])

    table["soh_q2"] = table.apply(
        lambda row: row["discharge_capacity"] / refs[row["cell_id"]]
        if refs[row["cell_id"]] and np.isfinite(refs[row["cell_id"]])
        else np.nan,
        axis=1,
    )

    # 标签有效性：两种口径都必须有限且为正。
    table["is_valid_label"] = (
        table["soh_nominal"].notna()
        & table["soh_q2"].notna()
        & np.isfinite(table["soh_nominal"])
        & np.isfinite(table["soh_q2"])
        & (table["soh_nominal"] > 0)
        & (table["soh_q2"] > 0)
    )

    keep = [
        "cell_id",
        "cycle_index",
        "discharge_capacity",
        "charge_capacity",
        "ir",
        "batch",
        "policy",
        "cycle_life",
        "soh_nominal",
        "soh_q2",
        "is_valid_label",
    ]
    return table[keep].reset_index(drop=True)


def main() -> None:
    """从统一 SOH 表生成 partial_soh 标签表。"""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    soh_table = pd.read_parquet(args.input)
    soh_table = apply_exclusions(soh_table)
    labels = build_labels(soh_table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.out, index=False)

    n_cells = labels["cell_id"].nunique()
    n_rows = len(labels)
    print(f"cells: {n_cells}, rows: {n_rows}")
    print("soh_nominal range: "
          f"{labels['soh_nominal'].min():.4f} ~ {labels['soh_nominal'].max():.4f}")
    print("soh_q2 range: "
          f"{labels['soh_q2'].min():.4f} ~ {labels['soh_q2'].max():.4f}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
