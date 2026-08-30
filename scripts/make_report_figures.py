"""实验章 5.2（Severson 同分布基线）绘图脚本。

输入：
    data/processed/temperature_soh/severson_test_preds_3ch.parquet
        （eval_severson_preds.py 的产物，含 cell_id/cycle_index/soh_true/soh_pred）

输出（docs/report/figures/）：
    fig52_scatter.png          预测 vs 真实散点图
    fig52_trajectories.png     4 只典型测试电池的 SOH 轨迹对比
    fig52_aging_buckets.png    误差 vs 老化阶段（SOH 分桶）
    fig52_per_cell_mae.png     按电池误差条形图
    fig52_temp_channel.png     3ch vs 4ch 温度通道消融（README 受控记录）

运行：
```powershell
& "E:\conda\envs\battery-soh\python.exe" scripts/make_report_figures.py
```
"""

from __future__ import annotations

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
PREDS_3CH = ROOT / "data" / "processed" / "temperature_soh" / "severson_test_preds_3ch.parquet"
FIG_DIR = ROOT / "docs" / "report" / "figures"

# 散点/桶的 SOH 区间与配色。
SOH_BINS = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70]
BIN_LABELS = ["0.95–1.00", "0.90–0.95", "0.85–0.90", "0.80–0.85", "0.70–0.80"]
BIN_COLORS = ["#2f6fb3", "#4c9be8", "#7fc4f5", "#f2b134", "#d9534f"]


def _load_preds() -> pd.DataFrame:
    """读取预测表；不存在时给出明确的运行指引。"""
    if not PREDS_3CH.exists():
        raise FileNotFoundError(
            f"找不到 {PREDS_3CH}，请先运行 "
            "src/temperature_soh/Trainer/eval_severson_preds.py"
        )
    return pd.read_parquet(PREDS_3CH)


def _soh_bin(value: float) -> str:
    """把 SOH 值映射到区间标签（pd.cut 的辅助函数）。"""
    for label, lo, hi in zip(BIN_LABELS, SOH_BINS[1:], SOH_BINS[:-1]):
        if lo <= value <= hi:
            return label
    return BIN_LABELS[-1]


def fig52_scatter(df: pd.DataFrame) -> None:
    """图 5-1：预测 vs 真实散点（抽样绘制，45° 线 + MAE 标注）。"""
    rng = np.random.default_rng(0)
    sample = df.sample(n=min(80_000, len(df)), random_state=rng)
    colors = sample["soh_true"].map(_soh_bin).map(
        dict(zip(BIN_LABELS, BIN_COLORS))
    )

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(sample["soh_true"], sample["soh_pred"], s=4, c=colors, alpha=0.35)
    lim = (0.70, 1.03)
    ax.plot(lim, lim, "k--", lw=1.2, label="y = x")
    err = df["soh_pred"] - df["soh_true"]
    mae, rmse = np.abs(err).mean() * 100, np.sqrt((err**2).mean()) * 100
    ax.text(0.72, 1.00, f"MAE = {mae:.2f}%\nRMSE = {rmse:.2f}%",
            fontsize=11, bbox=dict(facecolor="white", alpha=0.8))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("真实 SOH")
    ax.set_ylabel("预测 SOH")
    ax.set_title("Severson 测试集预测 vs 真实（24 只电池，86.9 万片段）")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig52_scatter.png", dpi=150)
    plt.close(fig)
    print(f"图5-1 散点图: MAE={mae:.2f}% RMSE={rmse:.2f}%")


def _pick_representative_cells(df: pd.DataFrame) -> list[str]:
    """选 4 只代表电池：最深退化 / 最长寿命 / 中位 / 最浅退化。"""
    stats = (
        df.groupby("cell_id")
        .agg(min_soh=("soh_true", "min"), n_cycles=("cycle_index", "nunique"))
        .reset_index()
    )
    deepest = stats.loc[stats["min_soh"].idxmin(), "cell_id"]
    longest = stats.loc[stats["n_cycles"].idxmax(), "cell_id"]
    shallow = stats.loc[stats["min_soh"].idxmax(), "cell_id"]
    mid = stats.loc[
        (stats["min_soh"] - stats["min_soh"].median()).abs().idxmin(), "cell_id"
    ]
    return [deepest, longest, mid, shallow]


def fig52_trajectories(df: pd.DataFrame) -> None:
    """图 5-2：4 只典型电池的 SOH 轨迹（循环级均值，真实 vs 预测）。"""
    cells = _pick_representative_cells(df)
    per_cycle = (
        df.groupby(["cell_id", "cycle_index"])[["soh_true", "soh_pred"]]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, cell in zip(axes.ravel(), cells):
        sub = per_cycle[per_cycle["cell_id"] == cell].sort_values("cycle_index")
        ax.plot(sub["cycle_index"], sub["soh_true"], "-", color="#d9534f",
                lw=1.6, label="真实 SOH")
        ax.plot(sub["cycle_index"], sub["soh_pred"], "--", color="#2f6fb3",
                lw=1.4, label="预测 SOH")
        ax.set_title(cell)
        ax.set_xlabel("循环号")
        ax.set_ylabel("SOH")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("典型测试电池的 SOH 退化轨迹（循环级均值）", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig52_trajectories.png", dpi=150)
    plt.close(fig)
    print(f"图5-2 轨迹图: 代表电池 = {cells}")


def fig52_aging_buckets(df: pd.DataFrame) -> None:
    """图 5-3：误差 vs 老化阶段（按真实 SOH 分桶的 MAE）。"""
    df = df.copy()
    df["bucket"] = df["soh_true"].map(_soh_bin)
    grouped = df.groupby("bucket").agg(
        mae=("soh_pred", lambda s: np.abs(s - df.loc[s.index, "soh_true"]).mean()),
        n=("soh_pred", "size"),
    ).reindex(BIN_LABELS)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(BIN_LABELS, grouped["mae"] * 100, color=BIN_COLORS, alpha=0.85)
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}%\n(n={row['n']:,})",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("SOH 区间（老化阶段）")
    ax.set_ylabel("MAE (%)")
    ax.set_title("误差随老化阶段的变化（越老越难？）")
    ax.set_ylim(0, grouped["mae"].max() * 100 * 1.35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig52_aging_buckets.png", dpi=150)
    plt.close(fig)
    print(f"图5-3 老化分桶: {grouped['mae'].round(4).mul(100).to_dict()}")


def fig52_per_cell_mae(df: pd.DataFrame) -> None:
    """图 5-4：按电池误差条形图（24 只测试电池，按批次着色）。"""
    df = df.copy()
    df["err"] = (df["soh_pred"] - df["soh_true"]).abs()
    per_cell = (
        df.groupby("cell_id")
        .agg(mae=("err", "mean"), n=("err", "size"))
        .reset_index()
        .sort_values("mae")
    )
    per_cell["batch"] = per_cell["cell_id"].str.rsplit("_", n=1).str[0]
    batch_colors = {
        "2017-06-30": "#2f6fb3",
        "20170512": "#7fc4f5",
        "2018-04-12": "#f2b134",
    }
    colors = per_cell["batch"].map(batch_colors)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(range(len(per_cell)), per_cell["mae"] * 100, color=colors)
    mean_mae = per_cell["mae"].mean() * 100
    ax.axhline(mean_mae, color="k", ls="--", lw=1.0,
               label=f"平均 {mean_mae:.2f}%")
    for bar, (_, row) in zip(bars, per_cell.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(per_cell)))
    ax.set_xticklabels(per_cell["cell_id"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("电池级 MAE (%)")
    ax.set_title("各测试电池的 MAE（颜色=批次）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig52_per_cell_mae.png", dpi=150)
    plt.close(fig)
    print(f"图5-4 按电池误差: 平均电池级 MAE={mean_mae:.2f}%")


def fig52_temp_channel() -> None:
    """图 5-5：3ch vs 4ch 温度通道消融（README 受控实验记录）。

    说明：1.40% 与 1.63% 来自 8/28 同一受控消融（同协议同种子，只开关
    温度通道）；实际部署模型 normalized_3ch.pt 复测为 1.47%。
    """
    labels = ["3 通道\n(I, V, Q)", "4 通道\n(I, V, Q, T)"]
    maes = [1.40, 1.63]
    colors = ["#2f6fb3", "#d9534f"]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    bars = ax.bar(labels, maes, color=colors, width=0.55)
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}%", ha="center", fontsize=12)
    ax.set_ylabel("Severson 测试集 MAE (%)")
    ax.set_title("温度通道消融（恒温 30°C 数据）")
    ax.set_ylim(0, 2.0)
    ax.text(0.5, -0.28,
            "受控消融（同协议同种子，2026-08-28 记录）\n"
            "恒温数据下温度通道无增益 → 温度模块仅在变温微调阶段启用",
            transform=ax.transAxes, ha="center", fontsize=9, color="#444")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig52_temp_channel.png", dpi=150)
    plt.close(fig)
    print("图5-5 温度通道消融: 3ch 1.40% vs 4ch 1.63%（受控记录）")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_preds()
    print(f"预测表: {len(df):,} 行, {df['cell_id'].nunique()} 只电池")
    fig52_scatter(df)
    fig52_trajectories(df)
    fig52_aging_buckets(df)
    fig52_per_cell_mae(df)
    fig52_temp_channel()
    print(f"全部保存至: {FIG_DIR}")


if __name__ == "__main__":
    main()
