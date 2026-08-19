"""充电阶段提取模块。

论文的输入是部分充电数据，因此需要先从完整循环中把充电段挑出来。

本模块只做“筛选 I > 0 的充电阶段”，不做片段切分、插值或特征计算。
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 判断充电段的电流阈值。原始数据中，充电电流为正，放电为负。
CHARGE_CURRENT_THRESHOLD = 0.0


def extract_charge_curve(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """从原始循环字典中提取充电阶段。

    参数
    ----
    raw : mat_io.load_raw_cycle 返回的字典，至少需要 "V", "I", "T", "t", "Qc"。

    返回
    ----
    dict:
      - t  : 充电阶段时间；
      - V  : 充电阶段电压；
      - I  : 充电阶段电流；
      - T  : 充电阶段温度；
      - Qc : 充电阶段累计充电容量。

    这里的 I 是正数，表示充电。
    """
    required = {"V", "I", "T", "t", "Qc"}
    missing = required - set(raw)
    if missing:
        raise KeyError(f"缺少充电阶段所需字段: {sorted(missing)}")

    lengths = {name: len(np.asarray(raw[name])) for name in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"充电阶段输入字段长度不一致: {lengths}")

    # I > 阈值 表示充电阶段。
    mask = np.asarray(raw["I"]) > CHARGE_CURRENT_THRESHOLD
    if not np.any(mask):
        raise ValueError("该循环没有充电段（I > 0 的点数为 0）")

    return {
        "t": np.asarray(raw["t"])[mask],
        "V": np.asarray(raw["V"])[mask],
        "I": np.asarray(raw["I"])[mask],
        "T": np.asarray(raw["T"])[mask],
        "Qc": np.asarray(raw["Qc"])[mask],
    }


if __name__ == "__main__":
    """冒烟测试：从第一个批次的第一只电池读取并提取充电阶段。"""
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")

    # DataLoader 包内直接运行时要导入同目录模块。
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mat_io import discover_batch_files, load_raw_cycle

    files = discover_batch_files()
    batch_name = sorted(files)[0]
    raw = load_raw_cycle(files[batch_name], cell_index=0, cycle_index=2)
    charge = extract_charge_curve(raw)

    print(f"batch: {batch_name}")
    print(f"raw_points: {len(raw['V'])}, charge_points: {len(charge['V'])}")
    print(f"charge Qc range: {charge['Qc'].min():.4f} ~ {charge['Qc'].max():.4f} Ah")
    print(f"charge end voltage: {charge['V'][-1]:.3f} V")
