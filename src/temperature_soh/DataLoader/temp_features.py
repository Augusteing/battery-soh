"""温度曲线形状特征提取（片段级）。

背景
----
之前温度嵌入的输入是"循环级温度标量"（充电段平均温度）。在恒温数据
（Severson 30°C）上它几乎没有判别力，在变温数据（SIT 环境温度组）上
也把"温度在充电过程中怎么变化"这个信息抹掉了——均值相同的两个片段，
一个可能升温快、一个可能升温慢，老化行为完全不同。

文献调研结论（见 docs/ 里的温度表征论文笔记）：
    温度嵌入应使用**曲线形状特征**，而不是均值标量。常用特征包括：
      - 温度水平：T_mean / T_start / T_end / T_max / T_min；
      - 温度变化量：ΔT = T_end - T_start，T_range = T_max - T_min；
      - 温升速率：dT/dSOC（线性拟合斜率 + 平滑后的最大/最小差分）；
      - 形状位置：最高温升率出现的位置、温度峰值的位置（0~1）。

本模块把"物理量 -> 特征向量"这一件事单独抽出来：
  - 输入：插值后的片段 dict（interpolate_segment 的返回值，含 101 点
    T 曲线和 capacity 网格）；
  - 输出：12 维 float32 特征向量，**全部是物理单位**（°C、°C/SOC、
    无量纲位置），不做统计归一化；
  - 归一化交给 Trainer/model.py 的 TemperatureEmbedding 完成
    （固定物理常数，不是数据驱动的 z-score，保证跨数据集可迁移）。

注意：SIT 的温度传感器分辨率约 0.1°C，而 101 点容量网格的 SOC
间距只有 0.002，逐点差分会被量化噪声主导（实测出现 ±50、±100
这类整齐极值）。因此差分特征先对 T 做 5 点（0.01 SOC）滑动平均
再求导；整体趋势则用线性拟合斜率，对噪声天然鲁棒。

为什么 dT/dSOC 而不是 dT/dQ：
    SOC = 容量坐标 ÷ 标称容量，无量纲。这样 Severson（1.1 Ah 18650）
    和 SIT（50 Ah 方形电芯）的温升率在同一量纲下可比，模型才能跨电芯
    迁移。若用 dT/dQ（°C/Ah），50 Ah 电芯的特征数值会小 45 倍。

用法：见文件底部冒烟测试。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# 把本目录放进 sys.path，便于像 scripts 一样直接运行。
DL_DIR = Path(__file__).resolve().parent
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from segments import (  # noqa: E402
    OBSERVED_CAPACITY_PCT,
    extract_charge_curve,
    interpolate_segment,
)
from sit_io import DEFAULT_SIT_DIR, SIT_NOMINAL_CAPACITY_AH, read_charge_cycle  # noqa: E402

# ---------------------------------------------------------------------------
# 特征定义
# ---------------------------------------------------------------------------

# 特征顺序（模型侧按此顺序拼接输入，修改顺序会破坏已训练权重）。
FEATURE_NAMES = (
    "T_mean",           # 片段平均温度（°C）
    "T_start",          # 片段起始温度（°C）
    "T_end",            # 片段结束温度（°C）
    "T_max",            # 片段最高温度（°C）
    "T_min",            # 片段最低温度（°C）
    "T_range",          # T_max - T_min（°C），片段内温度摆幅
    "dT_end_start",     # T_end - T_start（°C），片段内净升温
    "slope_T_vs_soc",   # 温度对 SOC 线性拟合斜率（°C/SOC）
    "dTdSOC_max",       # 最大温升率（°C/SOC），差分峰值
    "dTdSOC_min",       # 最小温升率（°C/SOC），差分谷值（可为负=降温）
    "pos_dTdSOC_max",   # 最大温升率出现位置（0~1，片段内相对位置）
    "pos_T_max",        # 温度峰值位置（0~1）
)

# 每个特征在 TemperatureEmbedding 里的归一化常数（物理定标，非统计量）。
# center 和 scale 与 FEATURE_NAMES 一一对应：
#   - 温度水平类（0~4）：以 25°C 为物理零点、10°C 为尺度（与 dataset.py
#     的 TEMP_CENTER_C / TEMP_SCALE_C 一致）；
#   - 温差类（5~6）：20% 容量窗口内温升通常 1~3°C，取尺度 3；
#   - 温升率类（7~9）：dT/dSOC 实测 5~30，取尺度 8；
#   - 位置类（10~11）：本来就落在 [0,1]，居中 0.5、尺度 0.25。
FEATURE_CENTER = (
    25.0, 25.0, 25.0, 25.0, 25.0,   # T_mean/T_start/T_end/T_max/T_min
    0.0, 0.0,                        # T_range / dT_end_start
    0.0, 0.0, 0.0,                   # slope / max / min
    0.5, 0.5,                        # pos_dTdSOC_max / pos_T_max
)
FEATURE_SCALE = (
    10.0, 10.0, 10.0, 10.0, 10.0,
    3.0, 3.0,
    8.0, 8.0, 8.0,
    0.25, 0.25,
)

N_FEATURES = len(FEATURE_NAMES)


def extract_temp_shape_features(
    seg: dict[str, np.ndarray],
    nominal_capacity: float,
    fallback_temp_c: float | None = None,
) -> np.ndarray:
    """从插值后的片段里提取 12 维温度形状特征（物理单位）。

    参数
    ----
    seg              : interpolate_segment 的返回值，必须含
                       "T"（101 点温度，°C）与 "capacity"（容量网格，Ah）。
    nominal_capacity : 标称容量（Ah），用于把容量坐标转成 SOC。
    fallback_temp_c  : 温度曲线全 NaN 时的兜底值（°C）；不传则抛错。

    返回
    ----
    np.ndarray 形状 (N_FEATURES,)，float32，顺序见 FEATURE_NAMES。
    """
    t = np.asarray(seg["T"], dtype=float)
    q = np.asarray(seg["capacity"], dtype=float)

    if t.size < 2:
        raise ValueError(f"温度曲线点数不足: {t.size}")
    if not np.isfinite(t).all():
        if fallback_temp_c is None:
            raise ValueError("温度曲线含 NaN 且未提供 fallback_temp_c")
        # 兜底：整段常数温度 -> 温差/斜率全为 0，位置取中点。
        t = np.full_like(t, float(fallback_temp_c))

    # 容量坐标转 SOC（无量纲），使温升率跨电芯可比。
    soc = q / float(nominal_capacity)
    # 平滑后再差分：0.02 SOC 窗口的滑动平均，抑制 0.1°C 传感器量化噪声。
    # 网格间距 = 0.2 SOC / 100 区间 = 0.002 SOC，11 点窗口恰好 0.02 SOC。
    soc_step = float(soc[-1] - soc[0]) / max(t.size - 1, 1)
    smooth_k = max(1, int(round(0.02 / soc_step)))
    # 边界用边缘复制填充（np.convolve 默认补零会在左端产生假跳变）。
    kernel = np.ones(smooth_k) / smooth_k
    pad_left = smooth_k // 2
    pad_right = smooth_k - 1 - pad_left
    t_pad = np.pad(t, (pad_left, pad_right), mode="edge")
    t_sm = np.convolve(t_pad, kernel, mode="valid")
    dtdq = np.diff(t_sm) / np.diff(soc)  # 平滑后差分温升率，°C/SOC

    # 线性拟合斜率：整体升温趋势（比端点差分对噪声更鲁棒）。
    slope = float(np.polyfit(soc, t, 1)[0]) if t.size >= 2 else 0.0

    # 位置特征：索引 / (n-1) 映射到 [0,1]；差分序列少一个点，用 n-2。
    n_t = t.size
    pos_t_max = float(np.argmax(t)) / (n_t - 1) if n_t > 1 else 0.5
    n_d = dtdq.size
    pos_dtdq_max = (
        float(np.argmax(dtdq)) / (n_d - 1) if n_d > 1 else 0.5
    )

    return np.asarray(
        [
            t.mean(),
            t[0],
            t[-1],
            t.max(),
            t.min(),
            t.max() - t.min(),
            t[-1] - t[0],
            slope,
            float(dtdq.max()) if n_d else 0.0,
            float(dtdq.min()) if n_d else 0.0,
            pos_dtdq_max,
            pos_t_max,
        ],
        dtype=np.float32,
    )


def _smoke_test() -> None:
    """对 SIT 一只电池的若干循环提取特征，打印取值范围供定标。"""
    sys.stdout.reconfigure(encoding="utf-8")
    cell_id = "001-1"
    all_feats: list[np.ndarray] = []
    for cycle_number in (50, 150, 300):
        cycle = read_charge_cycle(cell_id, cycle_number, DEFAULT_SIT_DIR)
        charge = extract_charge_curve(cycle)
        # 取 1%~21% 容量窗口（起点 1%）和 30%~50%（起点 30%）两类。
        # 注意：SIT 充电曲线 Qc 起点略大于 0（约 0.01 Ah），0% 起点
        # 会被 is_valid_soh 判为不合法，因此冒烟测试从 1% 起点开始。
        for start_pct in (0.01, 0.30):
            seg = interpolate_segment(
                charge,
                start_ah=start_pct * SIT_NOMINAL_CAPACITY_AH,
                end_ah=(start_pct + OBSERVED_CAPACITY_PCT) * SIT_NOMINAL_CAPACITY_AH,
                nominal_capacity=SIT_NOMINAL_CAPACITY_AH,
            )
            feat = extract_temp_shape_features(
                seg, nominal_capacity=SIT_NOMINAL_CAPACITY_AH
            )
            all_feats.append(feat)
            print(
                f"cycle {cycle_number:>4}  起点 {start_pct * 100:>2.0f}%: "
                f"T_mean {feat[0]:5.1f}°C  ΔT {feat[6]:+5.2f}°C  "
                f"dT/dSOC_max {feat[8]:5.2f}  dT/dSOC_min {feat[9]:+5.2f}  "
                f"峰位 {feat[10]:.2f}"
            )

    feats = np.stack(all_feats)
    print("\n各特征取值范围（供核对 FEATURE_CENTER / FEATURE_SCALE）:")
    for name, vals in zip(FEATURE_NAMES, feats.T):
        print(f"  {name:<14} {vals.min():9.3f} ~ {vals.max():9.3f}")


if __name__ == "__main__":
    _smoke_test()
