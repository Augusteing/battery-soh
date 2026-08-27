"""temperature_soh SOH 标签模块（Severson 训练集，原论文口径）。

本模块只负责一件事：遍历 Severson（MATR）全部电池，提取每个循环的
最终充电容量，计算原论文口径的 SOH 标签，输出一张标签表 parquet。

**训练集口径（用户决策）**：与 partial_soh（片段预测）完全一致，
只用 Severson 数据集训练；本模块与片段预测的唯一区别是后续
片段输入会增加温度通道。SNL / HUST 不在训练集内（测试期另行考虑）。

原论文口径（partial_soh 复现的 Scientific Reports 2026）：

    SOH(k) = Q_charge(k) / Q_nominal

即“当前循环可充电容量 / 标称容量（1.1 Ah）”，论文原文：
"SOH is defined as the ratio between the current chargeable capacity
and the nominal capacity"。对应 MATR 的 Qc（逐点累计充电容量，
取最后一个值）。

注意：本模块**不使用**放电容量口径（Qd / Qd(cycle 2) 是 World Model
那篇论文的口径，仅作历史对照，不在本模块使用）。

数据质量口径：

  - cycle 1 排除（已知数据质量问题）；
  - MATR 应用 partial_soh 的 123-cell 口径（140 -> 124 -> 123），
    并标记坏循环（只标记、不删除整只电池）；

输出列：

  cell_id, cycle_index, charge_capacity_ah, soh, temperature_c,
  policy, is_bad_cycle, is_valid_label

policy 列来自 Severson .mat 的 batch 级字段（原始编码，如 "1C_4PER_6C"），
供 splits.py 做“按协议留出”的工况泛化划分。

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
# 实测 MATR 正常范围：起始约 1.0，老化终点约 0.73。
# 因此 [0.5, 1.3] 远宽于所有正常数据，只用于拦截传感器尖峰
# （例如 20170512_c018 的充电容量漂移尖峰），不会误杀正常样本。
SOH_MIN_VALID = 0.5
SOH_MAX_VALID = 1.3

# ---------------------------------------------------------------------------
# 放电容量提取
# ---------------------------------------------------------------------------

def final_capacity(capacity_in_Ah: Any) -> float:
    """从“逐点累计容量”数组里提取循环的最终容量（Ah）。

    MATR 的 Qc 与 V/I/T 同长（逐点累计曲线），最后一个值 = 循环总充电量。
    取最后一个**有限**值即可；若整条曲线都无效，返回 NaN，
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


def _decode_ascii(f: h5py.File, value: Any) -> str:
    """把 MATLAB 存成 uint16/uint8 的 ASCII 字符串数组解码为 str。"""
    arr = np.asarray(f[value][()]).ravel()
    text = bytes(int(x) for x in arr).decode("ascii", errors="replace")
    return text.replace("\x00", "").strip()


# ---------------------------------------------------------------------------
# Severson 标签构建
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
            # batch 级 policy 字段：shape (n_cells, 1) 的对象引用数组。
            policy_refs = np.asarray(batch["policy"])
            for cell_index in range(n_cells):
                cell_ref = batch["cycles"][cell_index, 0]
                cell = f[cell_ref]
                n_cycles = int(np.asarray(cell["V"]).shape[0])
                cell_id = cell_id_for(batch_name, cell_index)
                policy = _decode_ascii(f, policy_refs[cell_index, 0])

                for cycle_index in range(1, n_cycles + 1):
                    qc = _deref_array(f, cell["Qc"][cycle_index - 1])
                    rows.append(
                        {
                            "cell_id": cell_id,
                            "cycle_index": cycle_index,
                            "charge_capacity_ah": final_capacity(qc),
                            "temperature_c": 30.0,  # MATR 为 30°C 恒温箱
                            "policy": policy,
                        }
                    )
        print(f"[labels] MATR 批次 {batch_name}: {n_cells} channel 已读取")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 过滤与 SOH 计算
# ---------------------------------------------------------------------------

def compute_soh(
    table: pd.DataFrame, nominal_capacity_ah: float = 1.1
) -> pd.DataFrame:
    """计算原论文口径 SOH，并标记标签有效性。

    SOH(k) = Q_charge(k) / Q_nominal（1.1 Ah）。
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
    """构建 Severson 标签表，应用全部质量口径。

    处理顺序：140 channel -> 123 cell（apply_paper_123 内含 cycle 1
    排除）-> 标记坏循环 -> 计算 SOH。
    """
    table = _build_matr_labels(mat_dir)
    table = apply_paper_123(table)
    table = mark_bad_cycles(table)
    return compute_soh(table)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main() -> None:
    """构建标签表并保存 parquet，同时打印统计摘要。"""
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    table = build_all_labels()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)

    print(f"\n标签表已保存: {args.out}")
    print("=" * 64)
    n_cells = table["cell_id"].nunique()
    n_rows = len(table)
    valid = int(table["is_valid_label"].sum())
    soh = table.loc[table["is_valid_label"], "soh"]
    print(
        f"[Severson/MATR] 电池 {n_cells} 只, 循环 {n_rows} 个, "
        f"有效标签 {valid} ({valid / n_rows:.1%}), "
        f"SOH 范围 [{soh.min():.3f}, {soh.max():.3f}]"
    )


if __name__ == "__main__":
    main()
