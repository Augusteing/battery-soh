"""组会汇报用图生成脚本。

基于已保存的基线模型（models/ablation_baseline.pt）与基线训练日志，
生成 5 张可直接放进 PPT 的图（输出到 results/figures/）：

    1. meeting_method_overview.png      方法总览（架构 + 两个创新点）
    2. meeting_loss_curves.png          预训练 / 微调损失曲线
    3. meeting_soh_scatter.png          测试集 SOH 预测 vs 真值
    4. meeting_cell_trajectories.png    典型测试电池的 SOH 轨迹
    5. meeting_example_segments.png     模型输入片段示例（V / I vs 容量）

说明：评估在 CPU 上完成（模型很小），不占用正在跑消融的 GPU。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TR_DIR = ROOT / "src" / "partial_soh" / "Trainer"
if str(TR_DIR) not in sys.path:
    sys.path.insert(0, str(TR_DIR))

from dataset import MemmapSohDataset  # noqa: E402
from model import PartialSohLSTM  # noqa: E402

FIG_DIR = ROOT / "results" / "figures"
INDEX = ROOT / "data" / "processed" / "partial_segments_index.parquet"
CACHE = ROOT / "data" / "processed" / "segments_cache"
LOG = ROOT / "results" / "runs" / "ablation_baseline.log"
MODEL = ROOT / "models" / "ablation_baseline.pt"

# Windows 中文字体：避免图上中文变方块。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def _test_index() -> pd.DataFrame:
    """加载与磁盘缓存行序一致的测试集片段索引（soh 有效行）。"""
    index = pd.read_parquet(INDEX)
    index = index[(index["split"] == "test") & (index["is_valid_soh"])].copy()
    return index.reset_index(drop=True)


def _predict_rows(model: PartialSohLSTM, ds: MemmapSohDataset, rows: np.ndarray) -> np.ndarray:
    """对给定行号批量预测 SOH（CPU，不占 GPU）。"""
    model.eval()
    preds: list[np.ndarray] = []
    batch = 4096
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            idx = [int(r) for r in rows[i : i + batch]]
            x, _ = ds.__getitems__(idx)  # (B, 101, 3)
            preds.append(model.soh_predict(x).numpy())
    return np.concatenate(preds)


def draw_method_overview(out: Path) -> None:
    """方法总览图：输入片段 -> 共享 LSTM 编码器 -> 三个头。"""
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#dbe9f6", ec="#2f5597", fs=10, dashed=False):
        style = "round,pad=0.08"
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.5,
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#2f5597", lw=1.6),
        )

    # 顶部：两个创新点说明。
    box(
        0.2, 5.2, 11.6, 0.8,
        "创新1 同循环一致性：同一循环切出的多个片段，SOH 输出必须彼此接近（组内方差正则）\n"
        "创新2 扩展自监督：掩码电压重建 + 下一步电压预测（预训练阶段）",
        fc="#fdf3d8", ec="#bf9000", fs=9,
    )

    # 主流程。
    box(0.2, 2.3, 2.5, 1.5, "部分充电片段\n(101×3)\n[I, V, Q]", fc="#e2efda")
    box(3.1, 2.3, 2.5, 1.5, "共享 LSTM 编码器\n隐藏状态 h_t\n(64 维)", fc="#dbe9f6")
    arrow(2.7, 3.05, 3.1, 3.05)

    # 三个输出头。
    box(6.0, 4.3, 2.6, 1.0, "电压预测头\n预训练：预测 V_{t+1}", fc="#fbe5d6")
    box(6.0, 2.4, 2.6, 1.0, "重建头\n预训练：掩码重建", fc="#fbe5d6")
    box(6.0, 0.5, 2.6, 1.0, "SOH 头\n微调：回归标量 SOH", fc="#fbe5d6")
    arrow(5.6, 3.05, 6.0, 4.8)
    arrow(5.6, 3.05, 6.0, 2.9)
    arrow(5.6, 3.05, 6.0, 1.0)

    # 输出说明。
    box(9.0, 4.3, 2.8, 1.0, "下一步电压\n（密集监督）", fc="#ffffff")
    box(9.0, 2.4, 2.8, 1.0, "补全被遮电压\n（重建损失）", fc="#ffffff")
    box(9.0, 0.5, 2.8, 1.0, "SOH 估计值\nTest MAE ≈ 2.27%", fc="#e2efda")
    arrow(8.6, 4.8, 9.0, 4.8)
    arrow(8.6, 2.9, 9.0, 2.9)
    arrow(8.6, 1.0, 9.0, 1.0)

    # SOH 分支的一致性约束示意（虚线框）。
    box(5.8, 0.25, 6.3, 1.5, "", fc="none", ec="#c00000", dashed=True)
    ax.text(
        8.9, 0.1, "同一循环的 K 个片段 → 输出一致",
        ha="center", va="top", fontsize=8, color="#c00000",
    )

    ax.set_title("方法总览：部分充电片段 → LSTM 迁移学习 → SOH 估计", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已生成: {out}")


def draw_loss_curves(out: Path) -> None:
    """从基线日志解析并绘制预训练 / 微调损失曲线。"""
    text = LOG.read_text(encoding="utf-8")

    def parse(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        pat = re.compile(rf"\[{prefix}\] epoch\s+(\d+)/\d+\s+loss=([\d.eE+-]+)")
        found = pat.findall(text)
        epochs = np.array([int(e) for e, _ in found])
        losses = np.array([float(l) for _, l in found])
        return epochs, losses

    pt_ep, pt_loss = parse("pretrain")
    ft_ep, ft_loss = parse("finetune")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(pt_ep, pt_loss, marker="o", ms=3, color="#4c72b0")
    axes[0].set_yscale("log")
    axes[0].set_title("预训练：电压预测损失")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE（对数轴）")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ft_ep, ft_loss, marker="o", ms=3, color="#dd8452")
    axes[1].set_title("微调：SOH 回归损失")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.3)

    fig.suptitle("训练损失曲线（基线配置，10 + 10 epoch）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已生成: {out}")


def draw_soh_scatter(out: Path) -> None:
    """测试集随机子样本上的预测 vs 真值散点图。"""
    ds = MemmapSohDataset(CACHE, "test", "soh")
    model = PartialSohLSTM()
    model.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))

    rng = np.random.default_rng(0)
    rows = rng.choice(len(ds), size=30_000, replace=False)
    y_true = ds._y[rows]
    y_pred = _predict_rows(model, ds, rows)

    err = y_pred - y_true
    mae = float(np.abs(err).mean()) * 100
    rmse = float(np.sqrt(np.mean(err**2))) * 100
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(y_true * 100, y_pred * 100, s=2, alpha=0.25, color="#4c72b0")
    lims = [80, 105]
    ax.plot(lims, lims, "k--", lw=1.2, label="y = x（理想）")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("真实 SOH（%）")
    ax.set_ylabel("预测 SOH（%）")
    ax.set_title(
        f"测试集 SOH 预测 vs 真值（n=30,000）\n"
        f"MAE = {mae:.2f}%   RMSE = {rmse:.2f}%   R² = {r2:.4f}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已生成: {out}")


def draw_cell_trajectories(out: Path) -> None:
    """挑选 3 只测试电池，绘制逐循环 SOH 真值 vs 预测轨迹。"""
    ds = MemmapSohDataset(CACHE, "test", "soh")
    model = PartialSohLSTM()
    model.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
    index = _test_index()

    # 选 3 只循环数多、且有明显老化的电池。
    stats = (
        index.groupby("cell_id")
        .agg(n_cycles=("cycle_index", "nunique"), min_soh=("soh_nominal", "min"))
        .reset_index()
    )
    candidates = stats[(stats["n_cycles"] >= 80) & (stats["min_soh"] < 0.97)]
    cells = candidates.sort_values("n_cycles", ascending=False)["cell_id"].head(3).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for ax, cell_id, color in zip(axes, cells, colors):
        rows = index.index[index["cell_id"] == cell_id].to_numpy()
        y_true = ds._y[rows]
        y_pred = _predict_rows(model, ds, rows)
        cycles = index.loc[rows, "cycle_index"].to_numpy()

        # 每个循环内：画所有片段预测（淡色点）+ 循环均值（实线）。
        df = pd.DataFrame({"cycle": cycles, "pred": y_pred, "true": y_true})
        per_cycle = df.groupby("cycle").agg(pred_mean=("pred", "mean"), true=("true", "mean"))
        ax.scatter(df["cycle"], df["pred"] * 100, s=1, alpha=0.15, color=color)
        ax.plot(per_cycle.index, per_cycle["true"] * 100, color="k", lw=2, label="真实 SOH")
        ax.plot(per_cycle.index, per_cycle["pred_mean"] * 100, color=color, lw=1.6, ls="--", label="预测均值")
        ax.set_title(f"{cell_id}\n（{len(df):,} 个片段）", fontsize=9)
        ax.set_xlabel("循环序号")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("SOH（%）")
    axes[0].legend(fontsize=8)
    fig.suptitle("典型测试电池的 SOH 老化轨迹（片段级预测 vs 真实值）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已生成: {out}")


def draw_example_segments(out: Path) -> None:
    """展示同一循环里 3 个不同起点的输入片段（V / I vs 容量）。"""
    ds = MemmapSohDataset(CACHE, "test", "soh")
    index = _test_index()

    # 挑一个片段数最多的循环，取 0% / 25% / 50% 三个起点。
    counts = index.groupby(["cell_id", "cycle_index"]).size().reset_index(name="n")
    best = counts.sort_values("n", ascending=False).iloc[0]
    rows = index[
        (index["cell_id"] == best["cell_id"])
        & (index["cycle_index"] == best["cycle_index"])
    ]
    starts = [0.0, 0.275, 0.55]
    picked = rows[rows["start_ah"].isin(starts)]
    if len(picked) < 2:
        picked = rows.head(3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for ax, (_, row), color in zip(axes, picked.iterrows(), colors):
        seg = ds._x[row.name]  # (101, 3)：I, V, Q
        q = seg[:, 2] * 1000  # Ah -> mAh
        label = f"起点 {row['start_ah'] / 0.011:.0f}%"
        axes[0].plot(q, seg[:, 1], color=color, label=label)
        axes[1].plot(q, seg[:, 0], color=color, label=label)

    axes[0].set_title("片段电压 V vs 容量坐标")
    axes[0].set_xlabel("累计充电容量（mAh）")
    axes[0].set_ylabel("电压（V）")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("片段电流 I vs 容量坐标")
    axes[1].set_xlabel("累计充电容量（mAh）")
    axes[1].set_ylabel("电流（A）")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"同一循环 {best['cell_id']} cycle {best['cycle_index']} 的输入片段示例", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已生成: {out}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    draw_method_overview(FIG_DIR / "meeting_method_overview.png")
    draw_loss_curves(FIG_DIR / "meeting_loss_curves.png")
    draw_soh_scatter(FIG_DIR / "meeting_soh_scatter.png")
    draw_cell_trajectories(FIG_DIR / "meeting_cell_trajectories.png")
    draw_example_segments(FIG_DIR / "meeting_example_segments.png")
    print("全部完成")


if __name__ == "__main__":
    main()
