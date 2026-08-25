"""MAT 原始数据读取模块。

本模块只负责 I/O：

1. 扫描 data/external/matr/ 下的批次 .mat 文件；
2. 读取某个电池、某个循环的原始 V/I/T/t/Qc/Qd 曲线；
3. 处理 MATLAB v7.3 文件里的 HDF5 引用问题。

它不应该做充电/放电判断、切片、特征计算或标签计算。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# 项目根目录：src/partial_soh/DataLoader/mat_io.py 向上 3 级得到项目根。
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAT_DIR = ROOT / "data" / "external" / "matr"

# 本复现只关心这些字段；其余字段留给其他模块。
RAW_FIELDS = ("t", "V", "I", "T", "Qc", "Qd")


def discover_batch_files(mat_dir: Path | None = None) -> dict[str, Path]:
    """扫描 .mat 文件，返回“批次名 -> 文件路径”的字典。

    批次名尽量从文件名中的日期提取，例如：
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
        if match:
            batch_name = match.group(1)
        else:
            batch_name = path.stem
        mapping[batch_name] = path

    if not mapping:
        raise FileNotFoundError(f"目录下没有找到 .mat 文件: {mat_dir}")
    return mapping


def _deref(f: h5py.File, value: Any) -> np.ndarray:
    """解开 MATLAB v7.3 的 HDF5 对象引用。

    在这类 .mat 文件中，struct/cell 数组的元素通常不是直接数值，
    而是一个 `<HDF5 object reference>`。这里把引用解析成真正的数组。

    例如 `batch["cycles"][cell_index, 0]` 或 `cell["V"][cycle_index]`
    得到的可能是引用，需要交给这个函数处理。
    """
    ref = np.asarray(value).item()
    return np.asarray(f[ref][()])


def load_raw_cycle(
    mat_path: Path,
    cell_index: int,
    cycle_index: int,
) -> dict[str, np.ndarray]:
    """读取一只电池的一个循环，返回原始曲线字典。

    参数
    ----
    mat_path   : 该批次 .mat 文件路径。
    cell_index : 电池在批次内的 0-based 下标，例如 c000 -> 0。
    cycle_index: 1-based 循环编号，例如第 2 个循环 -> 2。

    返回
    ----
    dict:
      - t  : 时间数组，单位分钟；
      - V  : 电压数组，单位 V；
      - I  : 电流数组，单位 C-rate（正数表示充电）；
      - T  : 温度数组，单位 °C；
      - Qc : 累计充电容量，单位 Ah；
      - Qd : 累计放电容量，单位 Ah。

    说明：这里不修改数组、不做插值、不做单位转换，只负责读出来。
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"找不到 MAT 文件: {mat_path}")

    # 文件句柄只在 with 块内存在，函数返回后自动关闭。
    # 这样调用方拿到的 numpy 数组已经是完整内存副本，不会依赖已关闭的文件。
    cell_index = int(cell_index)
    cycle_index = int(cycle_index)

    with h5py.File(str(mat_path), "r") as f:
        return read_raw_cycle_from_file(f, cell_index, cycle_index)


def read_raw_cycle_from_file(
    f: h5py.File,
    cell_index: int,
    cycle_index: int,
) -> dict[str, np.ndarray]:
    """从“已经打开”的 h5py 文件句柄读取一个循环。

    与 load_raw_cycle 的逻辑完全相同，区别是不负责打开/关闭文件。
    这样调用方可以一次性打开批次文件并复用句柄，避免反复开关文件的
    I/O 开销（对全量预加载尤其重要）。
    """
    batch = f["batch"]
    n_cells = int(batch["cycles"].shape[0])
    if not (0 <= cell_index < n_cells):
        raise IndexError(f"cell_index={cell_index} 超出文件内电池数 {n_cells}")

    # batch["cycles"] 是 HDF5 引用数组；先取引用，再解引用成 cell 结构。
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

    # 所有字段必须等长，否则说明原始数据有问题。
    lengths = {name: len(arr) for name, arr in out.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"原始字段长度不一致: {lengths}")

    out["n_cycles"] = np.array(n_cycles, dtype=np.int64)
    return out


if __name__ == "__main__":
    """小规模冒烟测试：读取第一个批次的第一只电池、第 2 个循环。"""
    files = discover_batch_files()
    first_batch = sorted(files)[0]
    raw = load_raw_cycle(files[first_batch], cell_index=0, cycle_index=2)

    print(f"batch: {first_batch}")
    print(f"n_points: {len(raw['t'])}")
    print(f"V range: {raw['V'].min():.3f} ~ {raw['V'].max():.3f} V")
    print(f"I range: {raw['I'].min():.3f} ~ {raw['I'].max():.3f} C-rate")
    print(f"Qc max: {raw['Qc'].max():.4f} Ah")
    print(f"Qd max: {raw['Qd'].max():.4f} Ah")
