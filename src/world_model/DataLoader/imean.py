"""逐循环平均充电电流模块（M2 · 第二步）。

论文口径（World Model, arXiv:2603.10527）:
  "The charging current I_mean is computed per cycle and used as the action
   vector u(k) for the dynamics transition."

关键理解
---------
- 不是整个循环的平均电流：充电(+C) 与放电(-4C) 相消，全循环均值接近 0，
  没有信息量；
- 是"充电段"的平均电流（I > 0 的部分）：对应快充协议的倍率，
  是跨电池/跨协议间唯一变化的运行条件（放电对所有电池恒为 4C）；
- 单位与原始 I 一致（C-rate，1C ≈ 1.1 A，M1 已验证）。

设计说明（软件工程）
---------------------
- 单一职责：本模块只计算标量 I_mean，不读 .mat、不写文件；
- 纯函数 API：输入一维电流数组，输出 float，易于单元测试；
- 防御性检查：空数组 / 无充电段（I > 0 点数为 0）直接抛异常。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def mean_charging_current(I: np.ndarray) -> float:
    """计算一个循环的平均充电电流（充电段 I > 0 的均值）。

    参数
    ----
    I : 一维 float 数组，该循环的电流序列（C-rate 单位，正=充电）。

    返回
    ----
    float: 平均充电电流（C-rate）。例如 3.6C 快充协议约为 3.6。

    异常
    ----
    ValueError: 输入为空数组，或不存在充电段（无 I > 0 的点）。
    """
    I = np.asarray(I, dtype=np.float64)

    # 防御：空数组直接报错
    if I.size == 0:
        raise ValueError("输入电流数组为空")

    # 充电段 = 电流为正的点
    charge = I[I > 0]
    if charge.size == 0:
        raise ValueError("该循环不存在充电段（无 I > 0 的点），无法计算平均充电电流")

    return float(charge.mean())


def _demo() -> None:
    """演示：对比"全循环均值（接近 0）"与"充电段均值 I_mean（≈协议倍率）"。"""
    from standardize_cycle import load_raw_cycle

    repo_root = Path(__file__).resolve().parents[3]
    mat = repo_root / "data/external/matr/MATR_batch_20170512.mat"

    for cell, cycle in [(0, 100), (5, 200)]:
        raw = load_raw_cycle(mat, cell=cell, cycle=cycle)
        i_mean = mean_charging_current(raw["I"])
        full_mean = float(raw["I"].mean())
        print(f"cell {cell}, cycle {cycle} | 全循环均值 = {full_mean:+.3f} C-rate"
              f"（接近0） | I_mean = {i_mean:.3f} C-rate | 协议 {raw['policy']}")


if __name__ == "__main__":
    _demo()