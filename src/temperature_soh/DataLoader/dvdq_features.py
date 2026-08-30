"""dV/dQ 差分电压特征提取（片段级，物理预处理）。

背景（导师建议 + 增量容量分析文献）
------------------------------------
LFP 电池老化的核心可观测信号之一是差分电压 dV/dQ（其倒数 dQ/dV 即
增量容量 ICA）：充电曲线上代表正负极材料相变的特征峰，其位置、宽度
和强度会随 SOH 衰减发生明显变化，即使平台区绝对电压几乎不变。

实车传感器噪声大、片段短，直接让轻量级 LSTM 从噪声里学高阶导数
既不稳定也不经济。本模块把"求导"这件事放在预处理阶段完成：

  1. 在等容量网格上直接用 Savitzky–Golay 滤波器的 deriv=1 输出
     解析导数。网格步长均匀（20% 窗口 / 100 区间 = 0.002 SOC），
     满足 S-G 的均匀采样假设；先平滑再前向差分会引入两步误差，
     故不采用；
  2. tanh() 把 dV/dQ 软压缩到 [-1, 1]，防止充电起止端极化剧烈时
     导数数值爆炸、反向传播把 LSTM 梯度炸飞；
  3. 窗口内三个统计矩（均值 / 方差 / 偏度）作为显式特征：
     窗口含相变峰时方差剧增、偏度刻画峰的倾斜方向；不含峰时自然
     趋近背景噪声水平——比硬性 find_peaks 鲁棒得多，避开"局部
     片段丢峰陷阱"（主峰在 SOC 29~56% 游走，窗口处于 0~20% 或
     10~30% 时峰根本不在片段内）。

单位说明
--------
容量坐标通道 Q 已被归一化为 SOC（容量 ÷ 标称容量，无量纲），因此
dV/dQ 的单位是 V/SOC。与 dT/dSOC 的设计一致：Severson（1.1 Ah）
与 SIT（50 Ah）在同一量纲下可比，模型才能跨电芯迁移。

输出
----
对 tanh(dV/dQ) 序列计算 3 个矩（全部有界），与 12 维温度形状特征
拼接后一起送入条件调制（FiLM）特征层。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

# 20% 容量窗口 = 0.2 SOC，101 个等容量点 -> 100 个区间。
SOC_STEP = 0.2 / 100.0  # 0.002 SOC / 点

DVDQ_MOMENT_NAMES = ("dvdq_mean", "dvdq_var", "dvdq_skew")
N_DVDQ_MOMENTS = len(DVDQ_MOMENT_NAMES)

# 矩特征在特征向量里的归一化常数（物理定标，非统计量）。
# tanh 输出有界：mean、skew 均在 (-1, 1) 内，取尺度 0.3；
# var 在 [0, ~0.3]（纯平台线趋近 0），取尺度 0.1、中心 0.05。
DVDQ_MOMENT_CENTER = (0.0, 0.05, 0.0)
DVDQ_MOMENT_SCALE = (0.3, 0.1, 0.3)


def compute_dvdq_moments(
    v: np.ndarray,
    q: np.ndarray,
    window: int = 21,
    polyorder: int = 3,
    soc_step: float = SOC_STEP,
) -> np.ndarray:
    """计算片段窗口内 tanh(dV/dQ) 的均值 / 方差 / 偏度。

    参数
    ----
    v         : (..., T) 电压曲线（伏），T 为等容量网格点数（101）。
    q         : (..., T) 容量坐标（SOC，无量纲），仅用于校验网格，
                导数按均匀步长 soc_step 计算。
    window    : S-G 滤波窗口长度（奇数，默认 21 点 ≈ 0.04 SOC）。
    polyorder : S-G 拟合多项式阶数（默认 3，低于窗口长度）。
    soc_step  : 等容量网格步长（SOC/点，默认 0.002）。

    返回
    ----
    (..., 3) float32，最后一维为 [mean, var, skew]：
      - mean : tanh(dV/dQ) 序列均值（峰整体越强越接近 ±1）；
      - var  : 方差（窗口含相变峰时剧增，纯平台线趋近 0）；
      - skew : 偏度（刻画峰的倾斜方向，无峰时趋近 0）。
    """
    v = np.asarray(v, dtype=float)
    q = np.asarray(q, dtype=float)
    if v.shape != q.shape:
        raise ValueError(f"v 与 q 形状不一致: {v.shape} vs {q.shape}")

    # deriv=1：S-G 在拟合局部多项式的同时直接输出解析导数。
    # delta=soc_step 把导数从"每网格点"换算成"每单位 SOC"。
    dvdq = savgol_filter(
        v,
        window_length=window,
        polyorder=polyorder,
        deriv=1,
        delta=soc_step,
        axis=-1,
    )
    x = np.tanh(dvdq)  # 软压缩到 [-1, 1]，防导数爆炸

    mean = x.mean(axis=-1)
    var = x.var(axis=-1)  # 总体方差（ddof=0）
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(
            std > 1e-12,
            np.mean(((x - mean[..., None]) / std[..., None]) ** 3, axis=-1),
            0.0,
        )
    return np.stack([mean, var, skew], axis=-1).astype(np.float32)


if __name__ == "__main__":
    """冒烟测试：合成含峰曲线 vs 平坦曲线，检查矩特征的行为。"""
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    n = 101
    soc = np.linspace(0.0, 0.2, n)
    # 含峰曲线：S 形电压（dV/dQ 在中间出现峰）
    v_peak = 3.3 + 0.2 / (1.0 + np.exp(-(soc - 0.1) / 0.02))
    # 平坦曲线：近似 CV 段（接近恒压，仅微小线性漂移）
    v_flat = np.linspace(3.6, 3.601, n)
    m_peak = compute_dvdq_moments(v_peak, soc)
    m_flat = compute_dvdq_moments(v_flat, soc)
    print(f"含峰: mean={m_peak[0]:.3f} var={m_peak[1]:.3f} skew={m_peak[2]:.3f}")
    print(f"平坦: mean={m_flat[0]:.3f} var={m_flat[1]:.3f} skew={m_flat[2]:.3f}")
    assert m_peak[1] > m_flat[1], "含峰窗口方差应显著大于平坦窗口"
