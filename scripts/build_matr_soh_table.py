"""MATR 批次 .mat -> 统一 SOH 标签表（从零实现）。

用法:
    python scripts/build_matr_soh_table.py --mat <path1> [--mat <path2> ...]
    python scripts/build_matr_soh_table.py  # 缺省扫描 data/external/matr/ 下全部批次
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# SOH 基准：取前 N 个正容量循环的中位数，抑制初期波动
REF_CYCLES = 10

SUMMARY_FIELDS = ("cycle", "QDischarge", "QCharge", "IR", "Tmax", "Tavg", "Tmin", "chargetime")


def _deref(f: h5py.File, value) -> np.ndarray:
    """MATLAB v7.3 中 struct/cell 数组元素为对象引用，解引用成数组。"""
    ref = np.asarray(value).item()
    return np.asarray(f[ref][()])


def _decode_policy(f: h5py.File, value) -> str:
    """policy_readable 在 .mat 中存为 UTF-16LE 字符串。"""
    raw = _deref(f, value).tobytes()
    if b"\x00" in raw:
        return raw.decode("utf-16-le", errors="ignore").strip("\x00").strip()
    return raw.decode("latin1", errors="ignore").strip()


def _batch_name(path: Path) -> str:
    """规范化批次名：优先取文件名中的日期（YYYY-MM-DD），否则去掉 MATR_batch_ 前缀。"""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
    return m.group(1) if m else path.stem.replace("MATR_batch_", "")


def extract_batch(path: Path) -> pd.DataFrame:
    """解析单个 MATR batch 文件，返回统一结构的 SOH 表。"""
    rows: list[pd.DataFrame] = []
    with h5py.File(str(path), "r") as f:
        batch = f["batch"]
        n_cells = batch["summary"].shape[0]
        cycle_life = [float(_deref(f, r).ravel()[0]) for r in batch["cycle_life"][()]]
        policies = [_decode_policy(f, r) for r in batch["policy_readable"][()]]
        batch_name = _batch_name(path)

        for i in range(n_cells):
            summary = f[batch["summary"][i, 0]]
            fields: dict[str, np.ndarray] = {}
            for name in SUMMARY_FIELDS:
                if name in summary:
                    fields[name] = np.asarray(summary[name][()]).ravel().astype(float)
            df = pd.DataFrame(fields)
            if df.empty:
                continue

            df = df.rename(
                columns={
                    "QDischarge": "discharge_capacity",
                    "QCharge": "charge_capacity",
                    "cycle": "cycle_index",
                }
            )
            df.columns = [c.lower() for c in df.columns]
            if "cycle_index" not in df:
                df["cycle_index"] = np.arange(1, len(df) + 1)

            # 去掉化成/初始化循环（容量为 0）
            df = df[df["discharge_capacity"] > 0].copy()
            if df.empty:
                continue
            df = df.sort_values("cycle_index").reset_index(drop=True)

            q0 = df["discharge_capacity"].head(REF_CYCLES).median()
            df["soh"] = df["discharge_capacity"] / q0
            df["cell_id"] = f"{batch_name}_c{i:03d}"
            df["batch"] = batch_name
            df["policy"] = policies[i] if i < len(policies) else ""
            df["cycle_life"] = cycle_life[i] if i < len(cycle_life) else np.nan
            rows.append(df)

    if not rows:
        raise ValueError(f"{path}: 未解析到任何电池数据")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat", type=Path, action="append", default=None, help="MATR 批次 .mat 路径，可重复；缺省扫描 data/external/matr/")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/matr_soh_table.parquet")
    args = parser.parse_args()

    mats = args.mat or sorted(
        p for p in (ROOT / "data/external/matr").glob("*.mat")
        if "MATR" in p.name or re.search(r"\d{4}-\d{2}-\d{2}", p.name)
    )
    if not mats:
        parser.error("未找到任何 MATR .mat 文件，请用 --mat 指定路径")

    table = pd.concat([extract_batch(p) for p in mats], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        table.to_parquet(args.out, index=False)
    else:
        table.to_csv(args.out, index=False)

    per_cell = table.groupby("cell_id").agg(n_cycles=("cycle_index", "max"), soh_min=("soh", "min"))
    print(f"cells: {len(per_cell)}  rows: {len(table)}")
    print(f"batches: {sorted(table['batch'].unique())}")
    print(f"cycles per cell: {per_cell['n_cycles'].min()} - {per_cell['n_cycles'].max()}")
    print(f"min SOH reached: {per_cell['soh_min'].min():.3f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())