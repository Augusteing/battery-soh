"""从 MATR SOH 表构建逐循环基线特征矩阵。

只使用当前时刻及之前可观测的信息（rolling 窗口、差分均不向未来看）。
不直接使用容量水平值作为特征（避免与 SOH 目标高度重合的平凡特征）。

用法:
    python scripts/build_matr_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def build_features(table: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """按电池分组计算特征，返回与 table 行一一对应的特征表。"""
    groups: list[pd.DataFrame] = []
    for _cell_id, g in table.sort_values("cycle_index").groupby("cell_id", sort=False):
        g = g.copy()
        g["cumulative_charge"] = g["charge_capacity"].cumsum()
        g["temp_amp"] = g["tmax"] - g["tmin"]
        g[f"ir_mean{window}"] = g["ir"].rolling(window, min_periods=3).mean()
        g[f"ir_std{window}"] = g["ir"].rolling(window, min_periods=3).std()
        g[f"tavg_mean{window}"] = g["tavg"].rolling(window, min_periods=3).mean()
        g[f"tavg_std{window}"] = g["tavg"].rolling(window, min_periods=3).std()
        g[f"ir_deriv{window}"] = g["ir"] - g["ir"].shift(window)
        g[f"capacity_deriv{window}"] = g["discharge_capacity"] - g["discharge_capacity"].shift(window)
        g["chargetime_ratio"] = g["chargetime"] / g["chargetime"].rolling(window, min_periods=3).mean()
        groups.append(g)
    # 丢弃滚动窗口未填满的早期循环（NaN）
    return pd.concat(groups, ignore_index=True).dropna().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/matr_soh_table.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/matr_features.parquet")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    table = pd.read_parquet(args.input)
    features = build_features(table, window=args.window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.out, index=False)
    print(f"rows: {len(features)}  cells: {features['cell_id'].nunique()}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())