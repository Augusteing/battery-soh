"""统一数据集注册表。

职责（只做登记与发现，不做切片/标签）：

1. 登记本项目用到的所有外部电池数据集（MATR / SNL / HUST）；
2. 扫描每个数据集下的电池文件；
3. 提供统一的电池加载入口（目前支持 pkl，mat 留给 mat_io 模块）；
4. 推断每只电池的工作温度（供训练时打温度标签、划分验证集使用）。

为什么要单独一个注册表？

- 三个数据集的原始格式不同（MATR 是 .mat，SNL/HUST 是 .pkl）；
- 如果每个模块都自己写“路径 + 格式”的判断，代码会散落满 if-else；
- 把“数据集有哪些、电池在哪、什么温度、SOH 怎么算”集中在一处，
  下游所有模块只依赖这里导出的常量与函数。

运行：

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/temperature_soh/DataLoader/registry.py"
```
"""

from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

# 本文件位于 src/temperature_soh/DataLoader/registry.py，
# 向上 3 级得到项目根目录。
ROOT = Path(__file__).resolve().parents[3]
DATA_EXTERNAL = ROOT / "data" / "external"


# ---------------------------------------------------------------------------
# 数据集规格
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """一个数据集的静态元数据。

    字段说明
    --------
    name              : 数据集短名（MATR / SNL / HUST）。
    chemistry         : 正极化学体系，本项目统一为 LFP。
    form_factor       : 电池规格，例如 18650。
    nominal_capacity_ah: 标称容量（Ah），用于展示与兜底归一化。
    temps_c           : 该数据集包含的温度档位（℃）。恒温数据集只有一档。
    raw_dir_name      : 在 data/external/ 下的目录名。
    file_glob         : 扫描电池文件的 glob 模式（相对 raw_dir_name）。
    file_format       : "mat" 表示 Severson 的 MATLAB v7.3 文件；
                        "pkl" 表示 BatteryLife 标准化 pkl。
    soh_reference     : SOH 参考口径：
                        - "cycle2"：以第 2 个循环的放电容量为参考（本项目统一口径）；
                        - "nominal"：以标称容量为参考（部分旧实验口径）。
    notes             : 备注，记录口径细节与注意事项。
    """

    name: str
    chemistry: str
    form_factor: str
    nominal_capacity_ah: float
    temps_c: tuple[float, ...]
    raw_dir_name: str
    file_glob: str
    file_format: Literal["mat", "pkl"]
    soh_reference: Literal["cycle2", "nominal"]
    notes: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "MATR": DatasetSpec(
        name="MATR",
        chemistry="LFP",
        form_factor="18650",
        nominal_capacity_ah=1.1,
        temps_c=(30.0,),
        raw_dir_name="matr",
        file_glob="*.mat",
        file_format="mat",
        soh_reference="cycle2",
        notes=(
            "Severson 2019 快充老化数据。本仓库 3 个 .mat 共 140 channel；"
            "Severson 124 口径排除 16 只；Scientific Reports 123 口径再排除 "
            "b2c1（2017-06-30_c001）。片段/SOH 口径见 partial_soh/DataLoader。"
        ),
    ),
    "SNL": DatasetSpec(
        name="SNL",
        chemistry="LFP",
        form_factor="18650",
        nominal_capacity_ah=1.1,
        temps_c=(15.0, 25.0, 35.0),
        raw_dir_name="SNL",
        file_glob="unpacked/SNL/SNL_18650_LFP_*.pkl",
        file_format="pkl",
        soh_reference="cycle2",
        notes=(
            "Sandia LFP 18650，15/25/35°C，A123 1.1Ah。"
            "18 只 LFP 是公开数据里唯一的变温度 LFP 来源，"
            "负责提供温度信号。"
        ),
    ),
    "HUST": DatasetSpec(
        name="HUST",
        chemistry="LFP",
        form_factor="18650",
        nominal_capacity_ah=1.1,
        temps_c=(30.0,),
        raw_dir_name="HUST",
        file_glob="unpacked/HUST/*.pkl",
        file_format="pkl",
        soh_reference="cycle2",
        notes=(
            "华科 77 只 LFP，30°C 恒温，相同充电协议 + 10 种多级放电协议。"
            "注意出厂容量约 1.19Ah > 标称 1.1Ah，SOH 必须用 cycle2 参考口径。"
        ),
    ),
}


# ---------------------------------------------------------------------------
# 电池文件发现
# ---------------------------------------------------------------------------

def raw_dir(spec: DatasetSpec) -> Path:
    """返回该数据集的原始数据目录。"""
    return DATA_EXTERNAL / spec.raw_dir_name


def list_cell_files(spec: DatasetSpec) -> list[Path]:
    """扫描并返回该数据集的所有电池文件（按文件名排序）。

    对 pkl 数据集，一个文件就是一只电池；
    对 mat 数据集，一个文件是一个批次（内含多只电池），
    这里的“电池文件”指批次文件本身，电池粒度由 mat_io 展开。
    """
    paths = sorted(raw_dir(spec).glob(spec.file_glob))
    if not paths:
        raise FileNotFoundError(
            f"数据集 {spec.name} 在 {raw_dir(spec)} 下没有匹配文件: {spec.file_glob}"
        )
    return paths


def count_cells_in_mat(mat_path: Path) -> int:
    """轻量统计一个 .mat 批次内的电池（channel）数。

    Severson 的 .mat 是 MATLAB v7.3（HDF5）格式，
    batch["cycles"] 的第一个维度就是电池数。
    这里只读形状、不加载数据，因此很快。
    """
    with h5py.File(str(mat_path), "r") as f:
        return int(f["batch"]["cycles"].shape[0])


def scan_all() -> dict[str, list[Path]]:
    """扫描全部数据集，返回“数据集名 -> 电池文件列表”。"""
    return {name: list_cell_files(spec) for name, spec in DATASETS.items()}


# ---------------------------------------------------------------------------
# 统一加载
# ---------------------------------------------------------------------------

def load_cell(spec: DatasetSpec, path: Path) -> dict[str, Any]:
    """按数据集格式加载一只电池，返回统一结构。

    当前实现：
    - pkl（SNL / HUST）：直接 pickle.load，得到 BatteryLife 标准化 dict，
      含 cell_id、cycle_data（list[dict]）、标称容量等；
    - mat（MATR）：暂不在此实现，由 mat_io.py 按“批次 + channel 下标”
      读取，避免把大文件整包读入内存。

    统一结构（pkl）：
    {
        "cell_id": str,
        "cycle_data": [
            {
                "cycle_number": int,
                "current_in_A": np.ndarray,        # 充电为正，放电为负
                "voltage_in_V": np.ndarray,
                "charge_capacity_in_Ah": np.ndarray,
                "discharge_capacity_in_Ah": np.ndarray,
                "time_in_s": np.ndarray,
                "temperature_in_C": np.ndarray | None,
                "internal_resistance_in_ohm": np.ndarray | None,
            },
            ...
        ],
        "nominal_capacity_in_Ah": float,
    }
    """
    if spec.file_format == "pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise NotImplementedError(
        f"数据集 {spec.name} 是 mat 格式，请用 mat_io.py 按批次+下标读取"
    )


# ---------------------------------------------------------------------------
# 温度推断
# ---------------------------------------------------------------------------

def infer_temperature_c(spec: DatasetSpec, cell: dict[str, Any]) -> float:
    """推断一只电池的工作温度（℃）。

    策略（按优先级）：
    1. 遍历循环，取第一个“非空、非全 NaN”的 temperature_in_C 序列中位数
       ——最可信，恒温箱温度有 ±1°C 波动，中位数能抗离群点；
    2. 若全部循环都没有温度序列，尝试从文件名解析温度档位
       （例如 SNL_18650_LFP_15C_... -> 15）；
    3. 解析不到且数据集只有一个温度档位，退回注册表。
    """
    for cycle in cell.get("cycle_data", []):
        temps = cycle.get("temperature_in_C")
        if temps is not None and len(temps) > 0:
            arr = np.asarray(temps, dtype=float)
            if np.isfinite(arr).any():
                return float(np.median(arr[np.isfinite(arr)]))

    # 文件名解析：SNL_18650_LFP_15C_0-100_... -> 15
    match = re.search(r"(\d+)C", Path(cell.get("cell_id", "")).name)
    if match is not None:
        return float(match.group(1))

    if len(spec.temps_c) == 1:
        return spec.temps_c[0]
    raise ValueError(
        f"无法推断温度，且数据集 {spec.name} 有多个温度档位: {spec.temps_c}"
    )


# ---------------------------------------------------------------------------
# 冒烟测试入口
# ---------------------------------------------------------------------------

def _main() -> None:
    """打印三个数据集的电池清单与统计，验证注册表可用。"""
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 64)
    print("统一数据集注册表 — 扫描结果")
    print("=" * 64)

    for name, spec in DATASETS.items():
        print(f"\n[{name}] {spec.chemistry} {spec.form_factor} "
              f"标称 {spec.nominal_capacity_ah}Ah, 温度档位 {spec.temps_c}")
        print(f"  目录: {raw_dir(spec)}")

        if spec.file_format == "mat":
            # mat：按批次统计电池数
            total = 0
            for batch_path in list_cell_files(spec):
                n = count_cells_in_mat(batch_path)
                total += n
                print(f"  - {batch_path.name}: {n} channels")
            print(f"  合计 {total} channels（口径见 partial_soh: 140→124→123）")
            continue

        # pkl：逐只电池读取元数据 + 推断温度
        cells = list_cell_files(spec)
        print(f"  电池文件数: {len(cells)}")
        temps: list[float] = []
        for path in cells[:3]:
            cell = load_cell(spec, path)
            t = infer_temperature_c(spec, cell)
            temps.append(t)
            n_cycles = len(cell["cycle_data"])
            print(f"  - {path.name}: 循环数={n_cycles}, 温度≈{t:.1f}°C")
        if len(cells) > 3:
            print(f"  ... 其余 {len(cells) - 3} 只省略（冒烟测试只看前 3 只）")
        # 用全部电池的温度中位数验证注册表的温度档位是否覆盖
        all_temps = {
            infer_temperature_c(spec, load_cell(spec, p)) for p in cells
        }
        print(f"  全部电池推断温度集合: {sorted(all_temps)}")


if __name__ == "__main__":
    _main()
