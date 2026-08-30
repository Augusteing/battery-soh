"""演示 dV/dQ 的计算流程（给导师展示用）。

步骤：
  1. 从 Severson 原始 .mat 读一个循环的充电段；
  2. 按等容量网格插值到 101 点（20% 容量窗口，步长 0.002 SOC）；
  3. Savitzky–Golay 滤波（窗口 21 点、3 阶多项式、deriv=1）直接输出
     解析导数 dV/dQ；
  4. tanh 软压缩到 [-1, 1]，即第 4 物理通道。

输出：
  - 中间数值打印（前 8 个点 + 峰值附近）；
  - 图：V(SOC) 与 tanh(dV/dQ)(SOC) 双面板 -> results/figures/demo_dvdq.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, savgol_coeffs

ROOT = Path(__file__).resolve().parents[1]
DL_DIR = ROOT / "src" / "temperature_soh" / "DataLoader"
sys.path.insert(0, str(DL_DIR))

from mat_io import discover_batch_files, load_unified_cycle  # noqa: E402
from segments import extract_charge_curve, interpolate_segment  # noqa: E402

SG_WINDOW = 21
SG_POLYORDER = 3
SOC_STEP = 0.2 / 100.0  # 0.002 SOC/点

files = discover_batch_files(ROOT / "data" / "external" / "matr")
# 选一块电池的新鲜循环，窗口 [0.44, 0.66] Ah 覆盖 LFP 主相变峰。
cycle = load_unified_cycle(files["20170512"], 0, 5)
charge = extract_charge_curve(cycle)
seg = interpolate_segment(
    charge, start_ah=0.44, end_ah=0.66, nominal_capacity=1.1
)
V = seg["V"]
Q_soc = seg["capacity"] / 1.1  # 容量坐标 -> SOC

# S-G 求导：deriv=1 在拟合局部多项式的同时输出解析导数。
dvdq = savgol_filter(
    V, window_length=SG_WINDOW, polyorder=SG_POLYORDER,
    deriv=1, delta=SOC_STEP,
)
dvdq_tanh = np.tanh(dvdq)

print("=" * 64)
print("步骤 1-2：等容量插值（20% 窗口 -> 101 点，步长 0.002 SOC）")
print("前 8 个网格点：")
for i in range(8):
    print(f"  SOC={Q_soc[i]:.4f}  V={V[i]:.4f} V")

print("\n步骤 3：S-G 滤波 deriv=1 的卷积系数（前 8 个，共 21 个）")
coeffs = savgol_coeffs(
    SG_WINDOW, SG_POLYORDER, deriv=1, delta=SOC_STEP
)
print("  " + " ".join(f"{c:+.2f}" for c in coeffs[:8]))
print(f"  中心点导数 = Σ_k c_k·V_k = {dvdq[10]:.4f} V/SOC")

print("\n步骤 3-4：dV/dQ 与 tanh(dV/dQ)（峰值附近）")
pk = int(np.argmax(dvdq_tanh))
for i in range(pk - 2, pk + 3):
    print(
        f"  SOC={Q_soc[i]:.4f}  V={V[i]:.4f}  "
        f"dV/dQ={dvdq[i]:+.3f}  tanh={dvdq_tanh[i]:+.3f}"
    )
print(f"\n窗口内：dV/dQ ∈ [{dvdq.min():.3f}, {dvdq.max():.3f}]，"
      f"tanh ∈ [{dvdq_tanh.min():.3f}, {dvdq_tanh.max():.3f}]")
print(f"峰值位于 SOC={Q_soc[pk]:.3f}（V={V[pk]:.3f} V），"
      f"tanh(dV/dQ)={dvdq_tanh[pk]:.3f}")

# 双面板图
fig, axes = plt.subplots(
    2, 1, figsize=(8, 6), sharex=True,
    gridspec_kw={"height_ratios": [1, 1]},
)
axes[0].plot(Q_soc, V, lw=1.5, color="#1f77b4")
axes[0].set_ylabel("V (V)")
axes[0].set_title("Step 2: equi-capacity interpolated voltage curve V(SOC)")
axes[0].grid(alpha=0.3)
axes[0].axvline(Q_soc[pk], color="gray", ls="--", lw=1)
axes[1].plot(Q_soc, dvdq_tanh, lw=1.5, color="#d62728")
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].axvline(Q_soc[pk], color="gray", ls="--", lw=1,
                label=f"主峰 SOC={Q_soc[pk]:.2f}")
axes[1].set_ylabel("tanh(dV/dQ)")
axes[1].set_xlabel("SOC")
axes[1].set_title("Step 3-4: S-G deriv=1 derivative + tanh (channel 4)")
axes[1].grid(alpha=0.3)
axes[1].legend()
fig.tight_layout()
out = ROOT / "results" / "figures" / "demo_dvdq.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\n图已保存: {out}")
