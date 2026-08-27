"""temperature_soh SOH 标签模块（对齐 Scientific Reports 2026 原论文口径）。

本模块只负责一件事：遍历 MATR / SNL / HUST 全部电池，提取每个循环的
最终充电容量，计算原论文口径的 SOH 标签，输出一张标签表 parquet。

原论文口径（partial_soh 复现的 Scientific Reports 2026）：

    SOH(k) = Q_charge(k) / Q_nominal

即“当前循环可充电容量 / 标称容量（1.1 Ah）”，论文原文：
"SOH is defined as the ratio between the current chargeable capacity
and the nominal capacity"。对应 MATR 的 Qc、SNL/HUST pkl 的
charge_capacity_in_Ah（逐点累计充电容量，取最后一个值）。

注意：本模块**不使用**放电容量口径（Qd / Qd(cycle 2) 是 World Model
那篇论文的口径，仅作历史对照，不在本模块使用）。

HUST 注意：出厂容量约 1.19 Ah > 标称 1.1 Ah，因此其起始 SOH 约 1.08，
这是原论文口径的固有属性；训练时模型可自行学习该偏移，无需额外归一化。

数据质量口径：

  - cycle 1 排除（已知数据质量问题）；
  - MATR 应用 partial_soh 的 123-cell 口径（140 -> 124 -> 123），
    并标记坏循环（只标记、不删除整只电池）；
  - SNL / HUST 全量保留（18 / 77 只），无电池级排除名单。

输出列：

  dataset, cell_id, cycle_index, charge_capacity_ah, soh, temperature_c,
  is_bad_cycle, is_valid_label

运行：

```powershell
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/DataLoader/labels.py
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

# 把本目录加入 sys.path，便于直接运行，也便于复用 mat_io / registry。
DL_DIR = Path(__file__).resolve().parent
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from mat_io import (  # noqa: E402
    count_cells_in_mat,
    discover_batch_files,
    cell_id_for,
)
from registry import DATASETS, infer_temperature_c, list_cell_files, load_cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]

# partial_soh 的数据质量模块：MATR 的排除名单与坏循环表都维护在那里，
# 这里直接复用，避免两份名单不一致。
PARTIAL_SOH_DL = ROOT / "src" / "partial_soh" / "DataLoader"
if str(PARTIAL_SOH_DL) not in sys.path:
    sys.path.insert(0, str(PARTIAL_SOH_DL))
from quality import apply_paper_123, mark_bad_cycles  # noqa: E402

# 输出路径：data/processed/temperature_soh/soh_labels.parquet
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "temperature_soh" / "soh_labels.parquet"

# SOH 物理合理范围（兜底清洗）。
# 实测各数据集正常范围：HUST 起始约 1.126，MATR/SNL 起始约 1.0，
# 老化终点约 0.7~0.8。因此 [0.5, 1.3] 远宽于所有正常数据，只用于
# 拦截传感器尖峰（例如 SNL 某循环 Qc=19.3 Ah、MATR c018 的漂移尖峰），
# 不会误杀正常样本。
SOH_MIN_VALID = 0.5
SOH_MAX_VALID = 1.3

# ---------------------------------------------------------------------------
# 放电容量提取
# ---------------------------------------------------------------------------

def final_capacity(capacity_in_Ah: Any) -> float:
    """从“逐点累计容量”数组里提取循环的最终容量（Ah）。

    统一循环结构里 charge_capacity_in_Ah / discharge_capacity_in_Ah 都是
    逐点累计数组：
      - MATR：与 V/I/T 同长的逐点累计曲线，最后一个值 = 循环总充电量；
      - SNL / HUST（BatteryLife pkl）：同样是逐点累计数组。
    因此取最后一个**有限**值即可；若整条曲线都无效，返回 NaN，
    由上层用 is_valid_label 过滤。
    """
    arr = np.asarray(capacity_in_Ah, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def _deref_array(f: h5py.File, value: Any) -> np.ndarray:
    """解开 MATLAB v7.3 的对象引用，返回一维数组。"""
    ref = np.asarray(value).item()
    return np.asarray(f[ref][()]).ravel()


# ---------------------------------------------------------------------------
# 各数据集标签构建
# ---------------------------------------------------------------------------

def _build_matr_labels(mat_dir: Path | None = None) -> pd.DataFrame:
    """遍历 MATR 三个批次，生成每循环标签（未过滤，保留全部 channel）。

    只读取 Qc（充电容量），不读 V/I/T/Qd，最小化大文件 IO。
    """
    mat_dir = Path(mat_dir) if mat_dir is not None else None
    batches = discover_batch_files(mat_dir)
    rows: list[dict[str, Any]] = []

    for batch_name, mat_path in batches.items():
        n_cells = count_cells_in_mat(mat_path)
        with h5py.File(str(mat_path), "r") as f:
            batch = f["batch"]
            for cell_index in range(n_cells):
                cell_ref = batch["cycles"][cell_index, 0]
                cell = f[cell_ref]
                n_cycles = int(np.asarray(cell["V"]).shape[0])
                cell_id = cell_id_for(batch_name, cell_index)

                for cycle_index in range(1, n_cycles + 1):
                    qc = _deref_array(f, cell["Qc"][cycle_index - 1])
                    rows.append(
                        {
                            "dataset": "MATR",
                            "cell_id": cell_id,
                            "cycle_index": cycle_index,
                            "charge_capacity_ah": final_capacity(qc),
                            "temperature_c": 30.0,  # MATR 为 30°C 恒温箱
                        }
                    )
        print(f"[labels] MATR 批次 {batch_name}: {n_cells} channel 已读取")

    return pd.DataFrame(rows)


def _build_pkl_labels(spec_name: str) -> pd.DataFrame:
    """遍历一个 pkl 数据集（SNL / HUST），生成每循环标签。"""
    spec = DATASETS[spec_name]
    rows: list[dict[str, Any]] = []

    for path in list_cell_files(spec):
        cell = load_cell(spec, path)
        temperature_c = infer_temperature_c(spec, cell)
        for cyc in cell.get("cycle_data", []):
            rows.append(
                {
                    "dataset": spec_name,
                    "cell_id": cell["cell_id"],
                    "cycle_index": int(cyc["cycle_number"]),
                    "charge_capacity_ah": final_capacity(
                        cyc["charge_capacity_in_Ah"]
                    ),
                    "temperature_c": temperature_c,
                }
            )
    print(f"[labels] {spec_name}: {len(rows)} 个循环已读取")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 过滤与 SOH 计算
# ---------------------------------------------------------------------------

def compute_soh(
    table: pd.DataFrame, nominal_capacity_ah: float = 1.1
) -> pd.DataFrame:
    """计算原论文口径 SOH，并标记标签有效性。

    SOH(k) = Q_charge(k) / Q_nominal（1.1 Ah），对所有数据集统一。
    """
    out = table.copy().sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    out["soh"] = out["charge_capacity_ah"] / nominal_capacity_ah

    # 标签有效：充电容量为正、SOH 有限且在物理合理范围内。
    out["is_valid_label"] = (
        out["charge_capacity_ah"].notna()
        & np.isfinite(out["charge_capacity_ah"])
        & (out["charge_capacity_ah"] > 0)
        & out["soh"].notna()
        & np.isfinite(out["soh"])
        & (out["soh"] >= SOH_MIN_VALID)
        & (out["soh"] <= SOH_MAX_VALID)
    )
    return out


def build_all_labels(mat_dir: Path | None = None) -> pd.DataFrame:
    """构建三个数据集统一标签表，应用全部质量口径。"""
    frames = [_build_matr_labels(mat_dir)]
    for spec_name in ("SNL", "HUST"):
        frames.append(_build_pkl_labels(spec_name))

    table = pd.concat(frames, ignore_index=True)

    # MATR：140 -> 123 cell（含 cycle 1 排除）；SNL/HUST 不适用电池排除。
    matr = table[table["dataset"] == "MATR"]
    others = table[table["dataset"] != "MATR"]
    matr = apply_paper_123(matr)
    matr = mark_bad_cycles(matr)

    # 非 MATR 数据集：统一排除 cycle 1，坏循环列置 False。
    others = others[others["cycle_index"] != 1].copy()
    others["is_bad_cycle"] = False

    merged = pd.concat([matr, others], ignore_index=True)
    return compute_soh(merged)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main() -> None:
    """构建标签表并保存 parquet，同时打印每数据集的统计摘要。"""
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    table = build_all_labels()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)

    print(f"\n标签表已保存: {args.out}")
    print("=" * 64)
    for dataset, group in table.groupby("dataset", sort=False):
        n_cells = group["cell_id"].nunique()
        n_rows = len(group)
        valid = int(group["is_valid_label"].sum())
        soh = group.loc[group["is_valid_label"], "soh"]
        print(
            f"[{dataset}] 电池 {n_cells} 只, 循环 {n_rows} 个, "
            f"有效标签 {valid} ({valid / n_rows:.1%}), "
            f"SOH 范围 [{soh.min():.3f}, {soh.max():.3f}]"
        )


if __name__ == "__main__":
    main()
