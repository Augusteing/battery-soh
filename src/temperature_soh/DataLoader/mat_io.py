"""Severson .mat 批次读取与统一结构转换。

职责：把 MATR（Severson 2019）的 MATLAB v7.3 数据，转换为与
BatteryLife 标准化 pkl（SNL / HUST）**完全相同的循环结构**，
让下游片段切分、标签、训练模块能无差别地处理三个数据集。

为什么需要转换而不是直接用原始字段？

BatteryLife pkl 的循环结构是：

    {
        "cycle_number": int,
        "current_in_A": np.ndarray,          # 安培，充电为正、放电为负
        "voltage_in_V": np.ndarray,
        "charge_capacity_in_Ah": np.ndarray,  # 循环内累计充电容量
        "discharge_capacity_in_Ah": np.ndarray,
        "time_in_s": np.ndarray,              # 秒
        "temperature_in_C": np.ndarray,
        "internal_resistance_in_ohm": np.ndarray | None,
    }

而 Severson 原始 .mat 的字段是 t（分钟）、V、I（**C-rate**）、T、Qc、Qd。
单位不一致：I 必须乘标称容量转成安培，t 必须乘 60 转成秒。
本模块负责这些转换（对齐 partial_soh 的 123 只电池口径，但不做过滤，
过滤名单由 splits 阶段使用）。

运行冒烟测试：

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/temperature_soh/DataLoader/mat_io.py"
```
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# 项目根目录：src/temperature_soh/DataLoader/mat_io.py 向上 3 级。
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAT_DIR = ROOT / "data" / "external" / "matr"

# MATR 电池标称容量（Ah），用于把 C-rate 电流换算成安培。
MATR_NOMINAL_CAPACITY_AH = 1.1

# 原始 .mat 里我们关心的字段。
RAW_FIELDS = ("t", "V", "I", "T", "Qc", "Qd")


# ---------------------------------------------------------------------------
# 批次发现
# ---------------------------------------------------------------------------

def discover_batch_files(mat_dir: Path | None = None) -> dict[str, Path]:
    """扫描 .mat 文件，返回“批次名 -> 文件路径”的字典。

    批次名尽量从文件名中的日期提取（与 partial_soh 一致）：
      - 2017-05-12_batchdata...mat -> 2017-05-12
      - 2018-04-12_batchdata...mat -> 2018-04-12
      - MATR_batch_20170512.mat    -> 20170512

    如果解析不出日期，就退化为文件名本身。
    """
    mat_dir = Path(mat_dir or DEFAULT_MAT_DIR)
    mapping: dict[str, Path] = {}
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{8})")

    for path in sorted(mat_dir.glob("*.mat")):
        match = pattern.search(path.name)
        batch_name = match.group(1) if match else path.stem
        mapping[batch_name] = path

    if not mapping:
        raise FileNotFoundError(f"目录下没有找到 .mat 文件: {mat_dir}")
    return mapping


def count_cells_in_mat(mat_path: Path) -> int:
    """返回一个 .mat 批次内的电池（channel）数。"""
    with h5py.File(str(mat_path), "r") as f:
        return int(f["batch"]["cycles"].shape[0])


def cell_id_for(batch_name: str, cell_index: int) -> str:
    """生成与 partial_soh 一致的电池 ID，例如 "2017-06-30_c001"。

    保持 ID 格式一致，是为了让旧的排除名单（140→124→123）能直接复用。
    """
    return f"{batch_name}_c{int(cell_index):03d}"


# ---------------------------------------------------------------------------
# 原始读取（不做单位转换）
# ---------------------------------------------------------------------------

def _deref(f: h5py.File, value: Any) -> np.ndarray:
    """解开 MATLAB v7.3 的 HDF5 对象引用。

    这类 .mat 中，struct/cell 数组的元素通常不是直接数值，
    而是一个 `<HDF5 object reference>`。这里把引用解析成真正的数组。
    """
    ref = np.asarray(value).item()
    return np.asarray(f[ref][()])


def read_raw_cycle_from_file(
    f: h5py.File,
    cell_index: int,
    cycle_index: int,
) -> dict[str, np.ndarray]:
    """从“已经打开”的 h5py 句柄读取一个循环的原始字段。

    cycle_index 是 1-based（与论文一致，第 2 个循环 -> 2）。
    返回的 dict 单位是原始的：t 分钟、I C-rate、V 伏、T 摄氏度、
    Qc/Qd 安时。
    """
    batch = f["batch"]
    n_cells = int(batch["cycles"].shape[0])
    if not (0 <= cell_index < n_cells):
        raise IndexError(f"cell_index={cell_index} 超出文件内电池数 {n_cells}")

    cell_ref = batch["cycles"][cell_index, 0]
    cell = f[cell_ref]

    n_cycles = int(np.asarray(cell["V"]).shape[0])
    if not (1 <= cycle_index <= n_cycles):
        raise IndexError(f"cycle_index={cycle_index} 超出电池循环数 {n_cycles}")

    raw_index = cycle_index - 1
    out: dict[str, np.ndarray] = {}
    for name in RAW_FIELDS:
        if name not in cell:
            raise KeyError(f"cell 中缺少字段 {name}")
        out[name] = _deref(f, cell[name][raw_index]).ravel()

    lengths = {name: len(arr) for name, arr in out.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"原始字段长度不一致: {lengths}")
    out["n_cycles"] = np.array(n_cycles, dtype=np.int64)
    return out


def load_raw_cycle(
    mat_path: Path,
    cell_index: int,
    cycle_index: int,
) -> dict[str, np.ndarray]:
    """打开文件并读取一个循环的原始字段（封装，方便外部调用）。"""
    with h5py.File(str(mat_path), "r") as f:
        return read_raw_cycle_from_file(f, cell_index, cycle_index)


# ---------------------------------------------------------------------------
# 统一结构转换
# ---------------------------------------------------------------------------

def convert_cycle_to_unified(
    raw: dict[str, np.ndarray],
    cycle_number: int,
    nominal_capacity_ah: float = MATR_NOMINAL_CAPACITY_AH,
) -> dict[str, Any]:
    """把原始字段转换为 BatteryLife 统一循环结构（含单位换算）。

    换算规则：
      - I（C-rate）-> current_in_A：乘标称容量；
      - t（分钟）  -> time_in_s：乘 60；
      - V / T / Qc / Qd 单位已一致，直接映射；
      - MATR 没有逐点内阻，internal_resistance_in_ohm 置 None。
    """
    return {
        "cycle_number": int(cycle_number),
        "current_in_A": raw["I"] * nominal_capacity_ah,
        "voltage_in_V": raw["V"],
        "charge_capacity_in_Ah": raw["Qc"],
        "discharge_capacity_in_Ah": raw["Qd"],
        "time_in_s": raw["t"] * 60.0,
        "temperature_in_C": raw["T"],
        "internal_resistance_in_ohm": None,
    }


def load_unified_cycle(
    mat_path: Path,
    cell_index: int,
    cycle_index: int,
    nominal_capacity_ah: float = MATR_NOMINAL_CAPACITY_AH,
) -> dict[str, Any]:
    """读取一个循环并返回统一结构（惰性：一次只读一个循环）。"""
    raw = load_raw_cycle(mat_path, cell_index, cycle_index)
    return convert_cycle_to_unified(
        raw, cycle_number=cycle_index, nominal_capacity_ah=nominal_capacity_ah
    )


def load_unified_cell(
    mat_path: Path,
    cell_index: int,
    batch_name: str | None = None,
) -> dict[str, Any]:
    """读取整只电池，返回与 pkl `load_cell` 对齐的统一 cell 结构。

    返回：
    {
        "cell_id": str,                 # 例如 "2017-06-30_c001"
        "cycle_data": list[统一循环结构],
        "nominal_capacity_in_Ah": float,
    }

    注意：整只电池可能有 1000+ 循环，此函数适合统计/预览；
    训练时请用 `load_unified_cycle` 按需读取，避免整包进内存。
    """
    mat_path = Path(mat_path)
    if batch_name is None:
        # 从文件名推断批次名（与 discover_batch_files 同规则）。
        match = re.search(r"(\d{4}-\d{2}-\d{2}|\d{8})", mat_path.name)
        batch_name = match.group(1) if match else mat_path.stem

    with h5py.File(str(mat_path), "r") as f:
        n_cycles = read_raw_cycle_from_file(f, cell_index, 1)["n_cycles"]
        cycles = []
        for cyc_idx in range(1, int(n_cycles) + 1):
            raw = read_raw_cycle_from_file(f, cell_index, cyc_idx)
            cycles.append(convert_cycle_to_unified(raw, cycle_number=cyc_idx))

    return {
        "cell_id": cell_id_for(batch_name, cell_index),
        "cycle_data": cycles,
        "nominal_capacity_in_Ah": MATR_NOMINAL_CAPACITY_AH,
    }


# ---------------------------------------------------------------------------
# 冒烟测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """读取第一个批次的第一只电池、第 2 个循环，验证单位转换。"""
    sys.stdout.reconfigure(encoding="utf-8")

    batches = discover_batch_files()
    print("发现批次:")
    for name, path in batches.items():
        print(f"  {name}: {path.name} ({count_cells_in_mat(path)} channels)")

    first_batch = sorted(batches)[0]
    cycle = load_unified_cycle(batches[first_batch], cell_index=0, cycle_index=2)
    print(f"\n批次 {first_batch} 电池 0 第 2 个循环（统一结构）:")
    for key, value in cycle.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, "
                  f"min={value.min():.4f}, max={value.max():.4f}")
        else:
            print(f"  {key}: {value}")

    # 单位验证：MATR 充电电流约 0.5~4 C-rate，乘 1.1 后应为安培。
    i_amp = cycle["current_in_A"]
    print(f"\n单位验证: I 范围 {i_amp.min():.3f} ~ {i_amp.max():.3f} A "
          f"（C-rate × {MATR_NOMINAL_CAPACITY_AH}）")
    t_sec = cycle["time_in_s"]
    print(f"时间范围: {t_sec.min():.1f} ~ {t_sec.max():.1f} s（分钟 × 60）")
