"""实验章 5.3（SIT 跨电芯 / 跨温度）绘图脚本。

输入：
    results/metrics/temperature_soh/sec53/<配置>_preds.parquet
        （finetune_sit.py --save-preds 的产物，含
        cell_id / cycle_index / soh_true / soh_pred）
    data/external/SIT/Data（画"单电池单循环"详情图时需要原始充电曲线）

输出（docs/report/figures/）：
    fig53_b1_scatter.png    B1 预测 vs 真实散点图（默认）
    fig53_cycle_detail.png  某电池某循环的详情图（充电曲线 + 片段预测分布）

运行：
```powershell
# 默认：B1 条件调制版（B1-film-rel）
& "E:\conda\envs\battery-soh\python.exe" scripts/make_sec53_figures.py

# 指定其他配置 / 输出文件名 / 标题（例如画 A 或 B2）
& "E:\conda\envs\battery-soh\python.exe" scripts/make_sec53_figures.py `
    --preds results/metrics/temperature_soh/sec53/B2-trans-nt_preds.parquet `
    --out fig53_b2_nt_scatter.png `
    --title "B2 跨温度外推（无温度）"

# 画某电池某循环的详情图（默认 002-1 循环 564）
& "E:\conda\envs\battery-soh\python.exe" scripts/make_sec53_figures.py `
    --cell 002-1 --cycle 564 --out fig53_cycle_002-1_564.png
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取 SIT 原始充电曲线需要 DataLoader 模块。
DL_DIR = Path(__file__).resolve().parents[1] / "src" / "temperature_soh" / "DataLoader"
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from sit_io import (  # noqa: E402
    DEFAULT_SIT_DIR,
    SIT_NOMINAL_CAPACITY_AH,
    read_charge_cycle,
)

# 中文字体（Windows 自带），负号用 ASCII 防止缺字形。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "report" / "figures"
DEFAULT_PREDS = (
    ROOT / "results" / "metrics" / "temperature_soh" / "sec53"
    / "B1-film-rel_preds.parquet"
)

# 与 5.2 图一致的 SOH 区间与配色。
SOH_BINS = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70]
BIN_LABELS = ["0.95–1.00", "0.90–0.95", "0.85–0.90", "0.80–0.85", "0.70–0.80"]
BIN_COLORS = ["#2f6fb3", "#4c9be8", "#7fc4f5", "#f2b134", "#d9534f"]
# 散点图固定坐标范围（真实 SOH 过滤后 0.75~1.0，留一点边距）。
PLOT_LIM = (0.70, 1.05)


def _soh_bin(value: float) -> str:
    """把 SOH 值映射到区间标签（pd.cut 的辅助函数）。"""
    for label, lo, hi in zip(BIN_LABELS, SOH_BINS[1:], SOH_BINS[:-1]):
        if lo <= value <= hi:
            return label
    return BIN_LABELS[-1]


def fig53_scatter(
    df: pd.DataFrame,
    out_name: str,
    title: str,
) -> None:
    """预测 vs 真实散点图（抽样绘制，45° 线 + 指标标注）。

    26.8 万片段全部画会严重重叠，随机抽 8 万点；按真实 SOH 所在
    老化阶段着色（与 5.2 图配色一致），便于看出"哪一段老化误差大"。
    """
    rng = np.random.default_rng(0)
    sample = df.sample(n=min(80_000, len(df)), random_state=rng)
    colors = sample["soh_true"].map(_soh_bin).map(
        dict(zip(BIN_LABELS, BIN_COLORS))
    )

    err = df["soh_pred"] - df["soh_true"]
    mae = np.abs(err).mean() * 100
    rmse = np.sqrt((err**2).mean()) * 100
    bias = err.mean() * 100
    pearson = df["soh_pred"].corr(df["soh_true"])
    # 统计预测越界比例（真实 SOH 恒在 [0,1]，预测偶尔会出界）。
    frac_oob = float(
        ((df["soh_pred"] < PLOT_LIM[0]) | (df["soh_pred"] > PLOT_LIM[1])).mean()
    ) * 100

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.scatter(sample["soh_true"], sample["soh_pred"], s=4, c=colors, alpha=0.35)
    ax.plot(PLOT_LIM, PLOT_LIM, "k--", lw=1.2, label="y = x")
    ax.text(
        0.71, 1.025,
        f"MAE = {mae:.2f}%\nRMSE = {rmse:.2f}%\n偏差 = {bias:+.2f}%\n"
        f"r = {pearson:.3f}",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#bbb"),
    )
    ax.set_xlim(PLOT_LIM)
    ax.set_ylim(PLOT_LIM)
    ax.set_xlabel("真实 SOH")
    ax.set_ylabel("预测 SOH")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(
        f"已保存 {FIG_DIR / out_name}\n"
        f"  MAE={mae:.2f}%  RMSE={rmse:.2f}%  偏差={bias:+.2f}%  "
        f"Pearson r={pearson:.3f}  越界占比={frac_oob:.2f}%"
    )


def fig53_cycle_detail(
    df: pd.DataFrame,
    cell_id: str,
    cycle_number: int,
    data_dir: Path,
    out_name: str,
) -> None:
    """某电池某循环的详情图：充电曲线 + 该循环各片段预测分布。

    上半部分画模型的原始输入（电压、电流、温度随 SOC 的变化），
    让"这个循环的数据长什么样"一目了然；下半部分画该循环所有
    片段（不同容量起点）的预测 SOH 分布，与真实 SOH 对照。
    一个循环内真实 SOH 是常数（一条水平线），预测散开说明
    单片段信息不完整 -> 实际部署应对同循环多片段取平均。
    """
    sub = df[(df["cell_id"] == cell_id) & (df["cycle_index"] == cycle_number)]
    if len(sub) == 0:
        raise ValueError(f"{cell_id} 循环 {cycle_number} 在预测表中没有片段")

    # 原始充电曲线：V / I / T 随充电容量（转为 SOC，跨电芯可比）。
    cycle = read_charge_cycle(cell_id, int(cycle_number), data_dir)
    soc = np.asarray(cycle["charge_capacity_in_Ah"]) / SIT_NOMINAL_CAPACITY_AH
    volt = np.asarray(cycle["voltage_in_V"])
    curr = np.asarray(cycle["current_in_A"]) / SIT_NOMINAL_CAPACITY_AH  # C-rate
    temp = np.asarray(cycle["temperature_in_C"])

    y_true = sub["soh_true"].mean()
    y_pred_mean = sub["soh_pred"].mean()
    mae = (sub["soh_pred"] - sub["soh_true"]).abs().mean() * 100

    fig = plt.figure(figsize=(8.2, 9.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 1.5], hspace=0.38)

    # 面板 1：电压 + 温度（双 y 轴）
    ax_v = fig.add_subplot(gs[0])
    ax_v.plot(soc, volt, color="#2f6fb3", lw=1.4, label="电压 V")
    ax_v.set_ylabel("电压 (V)", color="#2f6fb3")
    ax_v.tick_params(axis="y", labelcolor="#2f6fb3")
    ax_t = ax_v.twinx()
    ax_t.plot(soc, temp, color="#d9534f", lw=1.4, label="温度 T")
    ax_t.set_ylabel("温度 (°C)", color="#d9534f")
    ax_t.tick_params(axis="y", labelcolor="#d9534f")
    ax_v.set_xlabel("SOC")
    ax_v.set_title(f"{cell_id} 循环 {cycle_number} 充电曲线（模型输入）")
    lines = ax_v.get_lines() + ax_t.get_lines()
    ax_v.legend(lines, [ln.get_label() for ln in lines], loc="lower right",
                fontsize=9)
    ax_v.grid(alpha=0.25)

    # 面板 2：电流（C-rate）
    ax_i = fig.add_subplot(gs[1])
    ax_i.plot(soc, curr, color="#2e8b57", lw=1.2)
    ax_i.fill_between(soc, curr, color="#2e8b57", alpha=0.15)
    ax_i.set_xlabel("SOC")
    ax_i.set_ylabel("电流 (C)")
    ax_i.set_title("充电电流（C-rate，恒流阶段为平台）")
    ax_i.grid(alpha=0.25)

    # 面板 3：该循环各片段的预测 vs 真实
    ax_p = fig.add_subplot(gs[2])
    idx = np.arange(len(sub))
    ax_p.axhline(y_true, color="#d9534f", lw=1.8,
                 label=f"真实 SOH = {y_true:.4f}")
    ax_p.scatter(idx, sub["soh_pred"], s=20, color="#2f6fb3", alpha=0.55,
                 label="各片段预测")
    ax_p.axhline(y_pred_mean, color="#2f6fb3", ls="--", lw=1.2,
                 label=f"预测均值 = {y_pred_mean:.4f}")
    ax_p.set_xlabel("片段序号（同一循环的不同容量起点）")
    ax_p.set_ylabel("SOH")
    ax_p.set_title(f"该循环 {len(sub)} 个片段的预测 vs 真实"
                   f"（循环 MAE = {mae:.2f}%）")
    ax_p.legend(loc="best", fontsize=9)
    ax_p.grid(alpha=0.25)

    fig.savefig(FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(
        f"已保存 {FIG_DIR / out_name}\n"
        f"  {cell_id} 循环 {cycle_number}: 片段数 {len(sub)}  "
        f"真实 {y_true:.4f}  预测均值 {y_pred_mean:.4f}  "
        f"MAE {mae:.2f}%"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, default=DEFAULT_PREDS,
                        help="预测 parquet 路径（默认 B1-film-rel）")
    parser.add_argument("--out", default="fig53_b1_scatter.png",
                        help="输出文件名（docs/report/figures/ 下）")
    parser.add_argument("--title",
                        default="B1 跨温度外推：预测 vs 真实\n"
                                "（10 环境温训练 → 10 恒温箱测试 · 条件调制 + 相对特征）",
                        help="图标题")
    parser.add_argument("--cell", default=None,
                        help="指定电池号（如 002-1），同时给 --cycle 则画循环详情图")
    parser.add_argument("--cycle", type=int, default=None,
                        help="指定循环号（需同时给 --cell）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SIT_DIR,
                        help="SIT 原始数据目录（画循环详情图时需要）")
    args = parser.parse_args()

    if not args.preds.exists():
        raise FileNotFoundError(
            f"找不到预测表 {args.preds}，请先运行 finetune_sit.py --save-preds"
        )
    df = pd.read_parquet(args.preds)
    print(f"预测表: {len(df):,} 行, {df['cell_id'].nunique()} 只电池")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.cell is not None or args.cycle is not None:
        if args.cell is None or args.cycle is None:
            raise ValueError("--cell 与 --cycle 必须同时提供")
        fig53_cycle_detail(df, args.cell, args.cycle, args.data_dir, args.out)
    else:
        fig53_scatter(df, args.out, args.title)


if __name__ == "__main__":
    main()
