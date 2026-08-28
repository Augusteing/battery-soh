"""SIT（新加坡理工）电池老化数据读取与统一结构转换。

职责：把 SIT Battery Degradation Dataset（figshare 10.25447/sit.32101523）
的 xlsx + CSV 原始格式，转换为与 ``mat_io.py`` / BatteryLife pkl **完全一致**
的统一循环结构，让下游的片段切分（segments.py）、标签（labels.py）、
训练（Trainer/）模块无差别地处理 Severson 与 SIT 两个数据集。

SIT 数据组织（下载后解压到 data/external/SIT/Data）::

    Data/
    ├── Documentation/                 # 列说明、热电偶映射、设备手册
    ├── Cycle_Summary/<cell>.csv       # 每循环两行（discharge/charge），含容量
    ├── Repower_001/  Cycle_Data/<cell>/*.xlsx   # 8 只电池，环境温度
    ├── Repower_002/  Cycle_Data/<cell>/*.xlsx   # 6 只电池，40°C 恒温箱
    ├── Repower_003/  Cycle_Data/<cell>/*.xlsx   # 4 只电池，40°C 恒温箱
    └── Chroma_101/   Cycle_Data/<cell>/*.xlsx   # 2 只电池，环境温度

每个 xlsx = 一个完整循环，含 **Discharge** 和 **Charge** 两个 sheet：

    Repower 设备（6 列）:
        Relative Time(Sec), Voltage(V), Current(A), Capacity(Ah),
        Energy(Wh), MTV Celsius T*N*          # 内嵌表面温度
    Chroma 设备（7 列）: 多一路 MTV 温度（T1 / T2）

本模块与 mat_io.py 的对应关系：

    - discover_batch_files()  -> discover_sit_cells()（电池级，含温度组信息）；
    - load_unified_cycle()    -> read_charge_cycle()（一个循环的充电段）；
    - Cycle_Summary           -> 提供 SOH 标签所需的逐循环放电容量。

注意：SIT 电芯是 50 Ah 方形 LFP（Severson 是 1.1 Ah 18650），跨电芯训练时
输入通道需要在 Dataset 层归一化为 SOC / C-rate，本模块只负责"读出来"，不负责
归一化。

运行冒烟测试：

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/temperature_soh/DataLoader/sit_io.py"
```
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 项目根目录：src/temperature_soh/DataLoader/sit_io.py 向上 3 级。
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SIT_DIR = ROOT / "data" / "external" / "SIT" / "Data"

# SIT 方形 LFP 电芯标称容量（Ah）。Cycle_Summary 实测放电容量约 52~55 Ah。
SIT_NOMINAL_CAPACITY_AH = 50.0

# 设备 -> 温度条件组。环境温度组（001/101）在室温下循环（温度随环境波动），
# 恒温箱组（002/003）在 40°C 下循环。这个分组是训练/测试划分的依据之一。
DEVICE_INFO: dict[str, dict[str, Any]] = {
    "Repower_001": {"temp_group": "ambient",   "n_temp_channels": 1},
    "Repower_002": {"temp_group": "chamber40", "n_temp_channels": 1},
    "Repower_003": {"temp_group": "chamber40", "n_temp_channels": 1},
    "Chroma_101":  {"temp_group": "ambient",   "n_temp_channels": 2},
}

# xlsx 里的列名（Charge/Discharge 两个 sheet 相同）。
COL_TIME = "Relative Time(Sec)"
COL_VOLTAGE = "Voltage(V)"
COL_CURRENT = "Current(A)"
COL_CAPACITY = "Capacity(Ah)"
COL_ENERGY = "Energy(Wh)"


# ---------------------------------------------------------------------------
# 电池发现
# ---------------------------------------------------------------------------

def discover_sit_cells(data_dir: Path | None = None) -> pd.DataFrame:
    """扫描 SIT 数据目录，返回电池清单 DataFrame。

    返回列：
        cell_id                 电池 ID，例如 "001-1"（与 Cycle_Summary 同名）；
        device                  所属测试设备（Repower_001 / ... / Chroma_101）；
        temp_group              "ambient"（环境温度）或 "chamber40"（40°C 恒温箱）；
        nominal_capacity_in_Ah  标称容量（50.0）；
        n_cycles                Cycle_Data 里的 xlsx 文件数（即循环数）；
        summary_path            Cycle_Summary/<cell>.csv 的路径（可能缺失）。
    """
    data_dir = Path(data_dir or DEFAULT_SIT_DIR)
    rows: list[dict[str, Any]] = []

    for device, info in DEVICE_INFO.items():
        cycle_dir = data_dir / device / "Cycle_Data"
        if not cycle_dir.is_dir():
            continue
        for cell_dir in sorted(cycle_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            n_cycles = len(list(cell_dir.glob("*.xlsx")))
            summary = data_dir / "Cycle_Summary" / f"{cell_dir.name}.csv"
            rows.append(
                {
                    "cell_id": cell_dir.name,
                    "device": device,
                    "temp_group": info["temp_group"],
                    "nominal_capacity_in_Ah": SIT_NOMINAL_CAPACITY_AH,
                    "n_cycles": n_cycles,
                    "summary_path": str(summary) if summary.exists() else None,
                }
            )

    if not rows:
        raise FileNotFoundError(f"目录下没有找到 SIT 电池数据: {data_dir}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 循环汇总（Cycle_Summary）与文件名映射
# ---------------------------------------------------------------------------

def load_cycle_summary(cell_id: str, data_dir: Path | None = None) -> pd.DataFrame:
    """读取 Cycle_Summary/<cell>.csv（每循环两行：discharge / charge）。

    关键列：
        Cycle            循环号（从 1 开始；同文件的 discharge/charge 共享）；
        Type             "discharge" 或 "charge"；
        Capacity_Ah      该半循环最终容量（放电容量 = SOH 标签的来源）；
        Energy_Wh / Duration_s / Start_Timestamp / Filename。
    """
    data_dir = Path(data_dir or DEFAULT_SIT_DIR)
    path = data_dir / "Cycle_Summary" / f"{cell_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到循环汇总文件: {path}")
    return pd.read_csv(path)


def find_cycle_file(
    cell_id: str, cycle_number: int, data_dir: Path | None = None
) -> Path:
    """根据 Cycle_Summary 把循环号映射到 Cycle_Data 里的 xlsx 文件。

    SIT 的文件名（如 2022_03_01_14_34_40__001_1_1_1.xlsx）里的数字是
    导出批次的 step 计数，**不是循环号**，因此必须查 Cycle_Summary 的
    Filename 列来定位文件。
    """
    summary = load_cycle_summary(cell_id, data_dir)
    match = summary[summary["Cycle"] == cycle_number]
    if match.empty:
        raise KeyError(f"{cell_id} 没有循环 {cycle_number}（共 {summary['Cycle'].max()} 个循环）")
    filename = str(match.iloc[0]["Filename"])
    # 从设备目录定位：cell_id 前缀（如 "001-1" -> "001_1"）用于文件名解析，
    # 但文件实际在 discover 到的设备目录里，这里按 cell_id 匹配设备。
    data_dir = Path(data_dir or DEFAULT_SIT_DIR)
    for device in DEVICE_INFO:
        candidate = data_dir / device / "Cycle_Data" / cell_id / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"找不到 {cell_id} 循环 {cycle_number} 的文件: {filename}"
    )


# ---------------------------------------------------------------------------
# xlsx -> 统一循环结构
# ---------------------------------------------------------------------------

def _pick_temperature_columns(columns: list[str]) -> list[str]:
    """从 xlsx 列名中挑出温度列（MTV Celsius 开头）。

    Repower 设备每只电池一列；Chroma 有两列（T1/T2），本函数返回全部，
    由调用方决定用哪一列（默认取第一个非空）。
    """
    return [c for c in columns if str(c).startswith("MTV")]


def read_charge_cycle(
    cell_id: str,
    cycle_number: int,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """读取一个循环的充电段，返回统一循环结构（对齐 mat_io.convert_cycle_to_unified）。

    返回：
    {
        "cycle_number": int,                  # 与 Cycle_Summary 对齐的循环号；
        "current_in_A": np.ndarray,           # 充电电流（正），单位安培；
        "voltage_in_V": np.ndarray,           # 端电压；
        "charge_capacity_in_Ah": np.ndarray,  # 充电段累计容量（从 0 起）；
        "discharge_capacity_in_Ah": None,     # Charge sheet 无放电曲线，置 None；
        "time_in_s": np.ndarray,              # 相对时间（秒）；
        "temperature_in_C": np.ndarray,       # 内嵌表面温度；
        "internal_resistance_in_ohm": None,
    }

    注意：SIT 的 Charge sheet 本身就是充电段（电流全为正），因此该结构
    与 mat_io 输出的"完整循环"等价，segments.extract_charge_curve 可无差别
    处理（提取 I > 0 的点即全部点）。
    """
    xlsx_path = find_cycle_file(cell_id, cycle_number, data_dir)
    df = pd.read_excel(xlsx_path, sheet_name="Charge")

    # 挑温度列；Chroma 双通道时取第一个非空列（同一电池两只热电偶）。
    temp_cols = _pick_temperature_columns(list(df.columns))
    if not temp_cols:
        raise ValueError(f"{xlsx_path.name} 缺少 MTV 温度列")
    temp_series = df[temp_cols[0]]
    for col in temp_cols[1:]:
        if temp_series.isna().all():
            temp_series = df[col]
    temperature = temp_series.to_numpy(dtype=float)

    return {
        "cycle_number": int(cycle_number),
        "current_in_A": df[COL_CURRENT].to_numpy(dtype=float),
        "voltage_in_V": df[COL_VOLTAGE].to_numpy(dtype=float),
        "charge_capacity_in_Ah": df[COL_CAPACITY].to_numpy(dtype=float),
        "discharge_capacity_in_Ah": None,
        "time_in_s": df[COL_TIME].to_numpy(dtype=float),
        "temperature_in_C": temperature,
        "internal_resistance_in_ohm": None,
    }


def read_discharge_capacity(
    cell_id: str, cycle_number: int, data_dir: Path | None = None
) -> float:
    """从 Cycle_Summary 读取指定循环的放电容量（Ah），SOH 标签用。"""
    summary = load_cycle_summary(cell_id, data_dir)
    match = summary[
        (summary["Cycle"] == cycle_number) & (summary["Type"] == "discharge")
    ]
    if match.empty:
        raise KeyError(f"{cell_id} 循环 {cycle_number} 没有 discharge 行")
    return float(match.iloc[0]["Capacity_Ah"])


# ---------------------------------------------------------------------------
# 电池级温度（循环级标量，供温度嵌入 / 分层划分）
# ---------------------------------------------------------------------------

def cell_temperature_c(
    cell_id: str,
    device: str | None = None,
    data_dir: Path | None = None,
) -> float:
    """返回一只电池的代表温度（摄氏度），用于电池级分层与温度标签。

    取该电池第一个循环的内嵌温度中位数作为代表值：
      - 环境温度组（001/101）：约为 25~35°C（随季节与自热波动）；
      - 40°C 恒温箱组（002/003）：接近 40°C + 自热。
    """
    data_dir = Path(data_dir or DEFAULT_SIT_DIR)
    if device is None:
        cell_row = discover_sit_cells(data_dir)
        cell_row = cell_row[cell_row["cell_id"] == cell_id]
        if cell_row.empty:
            raise KeyError(f"未知电池: {cell_id}")
        device = str(cell_row.iloc[0]["device"])
    cycle = read_charge_cycle(cell_id, 1, data_dir)
    temps = np.asarray(cycle["temperature_in_C"], dtype=float)
    temps = temps[np.isfinite(temps)]
    if temps.size == 0:
        raise ValueError(f"{cell_id} 第一个循环没有有效温度")
    return float(np.median(temps))


# ---------------------------------------------------------------------------
# 冒烟测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """列出所有电池，并读取第一只电池的前 3 个循环验证结构。"""
    sys.stdout.reconfigure(encoding="utf-8")

    cells = discover_sit_cells()
    print("发现 SIT 电池:")
    for _, row in cells.iterrows():
        print(
            f"  {row['cell_id']:<8} {row['device']:<12} "
            f"{row['temp_group']:<10} {int(row['n_cycles']):>5} 循环"
        )
    print(f"共 {len(cells)} 只电池")

    first = str(cells.iloc[0]["cell_id"])
    print(f"\n读取 {first} 的前 3 个充电循环:")
    for cyc in (1, 2, 3):
        cycle = read_charge_cycle(first, cyc)
        q_dis = read_discharge_capacity(first, cyc)
        print(
            f"  cycle {cyc}: {len(cycle['time_in_s'])} 点, "
            f"V {cycle['voltage_in_V'].min():.3f}~{cycle['voltage_in_V'].max():.3f} V, "
            f"I {cycle['current_in_A'].min():.2f}~{cycle['current_in_A'].max():.2f} A, "
            f"Q 充到 {cycle['charge_capacity_in_Ah'].max():.2f} Ah, "
            f"T {cycle['temperature_in_C'].min():.1f}~{cycle['temperature_in_C'].max():.1f} °C, "
            f"放电容量 {q_dis:.2f} Ah"
        )

    print(f"\n代表温度: {first} = {cell_temperature_c(first):.1f} °C")
