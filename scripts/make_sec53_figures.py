"""实验章 5.3（SIT 跨电芯 / 跨温度）绘图脚本。

输入：
    results/metrics/temperature_soh/sec53/<配置>_preds.parquet
        （finetune_sit.py --save-preds 的产物，含
        cell_id / cycle_index / soh_true / soh_pred）

输出（docs/report/figures/）：
    fig53_b1_scatter.png    B1 预测 vs 真实散点图（默认）

运行：
```powershell
# 默认：B1 条件调制版（B1-film-rel）
& "E:\conda\envs\battery-soh\python.exe" scripts/make_sec53_figures.py

# 指定其他配置 / 输出文件名 / 标题（例如画 A 或 B2）
& "E:\conda\envs\battery-soh\python.exe" scripts/make_sec53_figures.py `
    --preds results/metrics/temperature_soh/sec53/B2-trans-nt_preds.parquet `
    --out fig53_b2_nt_scatter.png `
    --title "B2 跨温度外推（无温度）"
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
    args = parser.parse_args()

    if not args.preds.exists():
        raise FileNotFoundError(
            f"找不到预测表 {args.preds}，请先运行 finetune_sit.py --save-preds"
        )
    df = pd.read_parquet(args.preds)
    print(f"预测表: {len(df):,} 行, {df['cell_id'].nunique()} 只电池")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig53_scatter(df, args.out, args.title)


if __name__ == "__main__":
    main()
