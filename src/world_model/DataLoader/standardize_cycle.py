"""单循环曲线标准化模块（M2 · 第一步）。

职责（单一职责原则）
---------------------
本模块只做一件事：把 MATR 原始数据中"一个循环"的 V/I/T 曲线
标准化为固定长度 (3, Tmax) 的浮点数组，供后续窗口构建与模型使用。

论文口径（World Model, arXiv:2603.10527）
-----------------------------------------
- 模型输入 = 每循环的原始电压 V、电流 I、温度 T 时间序列；
- 每条序列 pad 或 truncate 到 Tmax = 1000 点；
- 放电容量 Qd 与内阻 IR 不进输入（仅作标签，属后续模块）。

两种变体（论文措辞存在歧义，见 docs/m1_data_exploration.md 第 6 节）
-------------------------------------------------------------------
- mode="full"      ：完整循环（充电+放电）截断/填充到 1000 点；
- mode="discharge"：只取放电段（I < 阈值）再填充/截断到 1000 点。

边界决策（可配置，默认值见 CycleStandardizer.__init__）
-------------------------------------------------------
- 序列超过 Tmax：保留前 Tmax 个点（时间序列习惯取开头）；
- 序列不足 Tmax：末尾填充，默认用"最后观测值"（pad_mode="edge"），可选填 0；
- 放电段判定：I < discharge_threshold（默认 -3.0，C-rate 单位）。

设计说明（软件工程）
---------------------
- 单一职责：本模块不读 SOH 表、不建窗口、不训练，只做"循环 -> (3, Tmax)"；
- IO 与变换分离：load_raw_cycle 负责读取，CycleStandardizer 负责变换；
- 纯函数式 API：不写文件、不改全局状态，方便单元测试与复用；
- 防御性检查：mode、数组长度、维度全部校验。

用法（直接运行演示）:
    python "src/world model/DataLoader/standardize_cycle.py"
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# 常量：论文口径与默认阈值
DEFAULT_TMAX = 1000                    # 论文的固定序列长度
DEFAULT_DISCHARGE_THRESHOLD = -3.0     # 判定放电段的电流阈值（C-rate 单位）
VALID_MODES = ("full", "discharge")    # 两种输入变体
VALID_PAD_MODES = ("edge", "zero")     # 两种填充方式

# 数据行顺序：与论文"V, I, T"一致，模型侧可直接使用
CHANNEL_NAMES = ("V", "I", "T")


def _deref(f: h5py.File, value) -> np.ndarray:
    """解引用：MATLAB v7.3 中 cell/struct 数组元素是 HDF5 引用，需取出真值。"""
    ref = np.asarray(value).item()
    return np.asarray(f[ref][()])


def load_raw_cycle(mat_path: str | Path, cell: int, cycle: int) -> dict:
    """从 .mat 读取一个循环的原始曲线（IO 职责）。

    参数
    ----
    mat_path : .mat 文件路径
    cell     : 电池下标（0 起）
    cycle    : 循环下标（0 起）

    返回
    ----
    dict: {"V", "I", "T", "t", "n_cycles", "policy"}
      V/I/T/t 为同长度的 float64 数组；n_cycles 为该电池总循环数。
    """
    with h5py.File(str(mat_path), "r") as f:
        batch = f["batch"]
        cyc = f[batch["cycles"][cell, 0]]

        # 防御：循环号越界直接报错，而不是返回空数据
        n_cycles = np.asarray(cyc["V"]).shape[0]
        if not (0 <= cycle < n_cycles):
            raise ValueError(f"cell={cell} 只有 {n_cycles} 个循环，cycle={cycle} 越界")

        # 读取四条约等长的原始序列
        raw = {k: _deref(f, cyc[k][cycle]).ravel().astype(np.float64)
               for k in ("t", "V", "I", "T")}

        # 防御：同族长度必须一致（防止把错位数组喂给模型）
        lengths = {k: len(v) for k, v in raw.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"raw 族长度不一致: {lengths}")

        # 协议名用于元数据/图标题（UTF-16 编码的字符串）
        policy = _deref(f, batch["policy_readable"][cell]).tobytes()
        policy = policy.decode("utf-16-le", errors="ignore").strip("\x00").strip()

    return {"V": raw["V"], "I": raw["I"], "T": raw["T"], "t": raw["t"],
            "n_cycles": n_cycles, "policy": policy}


class CycleStandardizer:
    """把一个循环的原始曲线标准化为 (3, Tmax) 数组（变换职责）。

    使用方式:
        std = CycleStandardizer(mode="full")
        X = std(raw)          # raw 来自 load_raw_cycle
    """

    def __init__(self,
                 tmax: int = DEFAULT_TMAX,
                 mode: str = "full",
                 discharge_threshold: float = DEFAULT_DISCHARGE_THRESHOLD,
                 pad_mode: str = "edge"):
        # 防御：构造参数合法性检查（fail fast）
        if mode not in VALID_MODES:
            raise ValueError(f"mode 必须是 {VALID_MODES} 之一，得到 {mode!r}")
        if pad_mode not in VALID_PAD_MODES:
            raise ValueError(f"pad_mode 必须是 {VALID_PAD_MODES} 之一，得到 {pad_mode!r}")
        if tmax <= 0:
            raise ValueError(f"tmax 必须为正整数，得到 {tmax}")

        self.tmax = tmax
        self.mode = mode
        self.discharge_threshold = discharge_threshold
        self.pad_mode = pad_mode

    def __call__(self, raw: dict) -> np.ndarray:
        """标准化单个循环。

        参数
        ----
        raw : load_raw_cycle 的返回字典（至少含 V, I, T）

        返回
        ----
        np.ndarray: 形状 (3, Tmax)，float32，行顺序 [V, I, T]。
        """
        # 防御：输入完整性检查
        missing = [k for k in CHANNEL_NAMES if k not in raw]
        if missing:
            raise KeyError(f"raw 缺少字段: {missing}")

        V, I, T = raw["V"], raw["I"], raw["T"]

        # 变体选择：先决定"用完整循环"还是"只取放电段"
        if self.mode == "discharge":
            V, I, T = self._extract_discharge(V, I, T)

        # 逐通道固定长度后堆叠成 (3, Tmax)
        X = np.stack([self._fix_length(V),
                      self._fix_length(I),
                      self._fix_length(T)])
        return X.astype(np.float32)

    # ------------------------------------------------------------------
    # 私有方法（不对外暴露，避免调用方依赖实现细节）
    # ------------------------------------------------------------------

    def _extract_discharge(self, V, I, T):
        """取放电段：电流低于阈值（恒流 4C 放电）的连续点。"""
        mask = I < self.discharge_threshold
        n = int(mask.sum())
        if n == 0:
            raise ValueError(
                f"未找到放电段（I < {self.discharge_threshold} 的点数为 0）")
        return V[mask], I[mask], T[mask]

    def _fix_length(self, arr: np.ndarray) -> np.ndarray:
        """把一维序列截断或填充到 self.tmax。"""
        n = len(arr)

        # 超长：截断，保留开头 Tmax 点
        if n > self.tmax:
            return arr[: self.tmax]

        # 恰好：直接返回
        if n == self.tmax:
            return arr

        # 不足：末尾填充
        pad = self.tmax - n
        if self.pad_mode == "edge":
            fill = float(arr[-1]) if n > 0 else 0.0   # 用最后观测值填充
        else:
            fill = 0.0                                 # 填 0
        return np.pad(arr, (0, pad), mode="constant", constant_values=fill)


def _demo() -> None:
    """演示：读取电池 0 的第 100 个循环，输出两种变体的标准化结果。"""
    # 仓库根目录 = 本文件向上 4 级: src/world model/DataLoader -> 仓库根
    repo_root = Path(__file__).resolve().parents[3]
    mat = repo_root / "data/external/matr/MATR_batch_20170512.mat"

    raw = load_raw_cycle(mat, cell=0, cycle=100)
    print(f"cell 0, cycle 100 | 原始长度 {len(raw['V'])} | 协议 {raw['policy']}")

    for mode in VALID_MODES:
        std = CycleStandardizer(mode=mode)
        X = std(raw)
        print(f"\n[mode={mode}] X.shape = {X.shape}")
        print(f"  V 范围: {X[0].min():.3f} ~ {X[0].max():.3f} V")
        print(f"  I 范围: {X[1].min():.3f} ~ {X[1].max():.3f} C-rate")
        print(f"  T 范围: {X[2].min():.3f} ~ {X[2].max():.3f} °C")


if __name__ == "__main__":
    _demo()