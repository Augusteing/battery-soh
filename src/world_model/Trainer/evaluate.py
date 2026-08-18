"""测试集评估模块（M3 · 第七步）：用训练好的模型在未见过的电池上出报告。

一句话：trainer.py 训练时只碰验证集（用于早停），evaluate.py 负责把
训练阶段从未见过的测试集（默认 split_by_cell 的 20 只电池）完整跑一遍，
输出正式指标和可视化，回答"模型到底能不能外推"。

为什么要有单独的评估模块：
  - 训练/调参过程中如果反复看测试集指标，等于把测试集"偷看"进了决策
    （信息泄漏），测试集就失去意义了。所以测试集只在最后评估一次；
  - 评估粒度比训练循环更细：不只是整体 MAE，还要看各老化阶段、各只
    电池的表现，找出模型的短板（比如只对健康电池准、对深度老化失效）。

指标口径：
  - MAE（平均绝对误差）：SOH 是归一化到 cycle 2 的比值，0.01 ≈ 1% 容量
    偏差，论文整体 MAE 0.006；
  - 窗口级：所有测试窗口一视同仁（论文主口径）；
  - 电池级：先对每只电池求平均再对所有电池求平均，避免窗口多的电池
    主导结果（诊断用，观察模型是否对某些电池系统性失效）。

用法：
    python "src/world_model/Trainer/evaluate.py" \
        --checkpoint "results/runs/run1_baseline/checkpoint.pt"

输出（results/runs/<run_name>/test_report.json + figures/）：
  - 整体 / 各老化阶段 / 各批次 / 各电池的 MAE
  - mae_by_horizon.png   误差随预测距离（1..80 步）的变化
  - mae_by_cell.png      每只测试电池的误差
  - trajectory_samples.png  几只电池的真实 vs 预测 SOH 轨迹
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OMP 冲突

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/world_model/Trainer"))
sys.path.insert(0, str(ROOT / "src/world_model/DataLoader"))

from dataset import WindowDataset                       # noqa: E402
from normalize import ChannelNormalizer                 # noqa: E402
from model import WorldModel                            # noqa: E402
from loss import WorldModelLoss                         # noqa: E402
from trainer import collate_window                      # noqa: E402


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader,
            device: torch.device) -> dict:
    """把整个 loader 跑一遍，收集预测与标签（不做任何聚合，方便细拆）。"""
    model.eval()
    out = {"s_cur": [], "s_fut": [], "y_cur": [], "y_fut": [],
           "cell_id": [], "stage": [], "batch": [], "pos": []}
    for batch in loader:
        X = batch["X"].to(device)
        u = batch["u"].to(device)
        s_cur, s_fut = model(X, u)
        out["s_cur"].append(s_cur.detach().cpu())
        out["s_fut"].append(s_fut.detach().cpu())
        out["y_cur"].append(batch["y_cur"])
        out["y_fut"].append(batch["y_fut"])
        out["cell_id"].extend(batch["cell_id"])
        out["stage"].extend(batch["stage"])
        out["batch"].extend([cid.rsplit("_", 1)[0] for cid in batch["cell_id"]])
        out["pos"].extend([int(p) for p in batch["pos"]])
    return {k: (torch.cat(v) if k in ("s_cur", "s_fut", "y_cur", "y_fut")
                else v) for k, v in out.items()}


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float((pred - true).abs().mean())


def report_metrics(res: dict) -> dict:
    """从 predict() 的结果计算多粒度指标。"""
    p_cur, p_fut = res["s_cur"], res["s_fut"]        # (N,) / (N, H)
    t_cur, t_fut = res["y_cur"], res["y_fut"]
    abs_fut = (p_fut - t_fut).abs()                  # (N, H)
    n = len(p_cur)

    report = {
        "n_windows": n,
        "n_cells": len(set(res["cell_id"])),
        "mae_cur": mae(p_cur, t_cur),
        "mae_fut": float(abs_fut.mean()),
        "rmse_fut": float(((p_fut - t_fut) ** 2).mean().sqrt()),
        "mae_h1": float(abs_fut[:, 0].mean()),
        "mae_h20": float(abs_fut[:, 19].mean()),
        "mae_h80": float(abs_fut[:, 79].mean()),
    }

    # 按老化阶段：窗口所属 stage（s1 健康 / s2 轻度 / s3 深度老化）
    report["by_stage"] = {}
    for stage in sorted(set(res["stage"])):
        mask = np.array([s == stage for s in res["stage"]])
        if mask.sum() == 0:
            continue
        report["by_stage"][stage] = {
            "n_windows": int(mask.sum()),
            "mae_cur": mae(p_cur[mask], t_cur[mask]),
            "mae_fut": float(abs_fut[mask].mean()),
            "mae_h80": float(abs_fut[mask, 79].mean()),
        }

    # 按制造批次：检查模型是否只在某个批次上有效
    report["by_batch"] = {}
    for b in sorted(set(res["batch"])):
        mask = np.array([x == b for x in res["batch"]])
        if mask.sum() == 0:
            continue
        report["by_batch"][b] = {
            "n_windows": int(mask.sum()),
            "mae_cur": mae(p_cur[mask], t_cur[mask]),
            "mae_fut": float(abs_fut[mask].mean()),
        }

    # 按电池：先逐电池平均再汇总（诊断哪些电池系统性失效）
    per_cell = {}
    for cid in sorted(set(res["cell_id"])):
        mask = np.array([c == cid for c in res["cell_id"]])
        per_cell[cid] = {
            "n_windows": int(mask.sum()),
            "mae_cur": mae(p_cur[mask], t_cur[mask]),
            "mae_fut": float(abs_fut[mask].mean()),
            "mae_h80": float(abs_fut[mask, 79].mean()),
        }
    report["by_cell"] = per_cell
    report["mae_fut_cell_avg"] = float(np.mean(
        [v["mae_fut"] for v in per_cell.values()]))
    return report


def plot_mae_by_horizon(res: dict, save_path: Path) -> None:
    """误差随预测距离 h=1..80 的曲线：rollout 误差累积的可视化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abs_fut = (res["s_fut"] - res["y_fut"]).abs()               # (N, H)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, 81), abs_fut.mean(dim=0).numpy(), "o-", ms=3,
            label="all windows", color="tab:blue")
    for stage in ("s1_healthy", "s2_mild", "s3_aged"):
        mask = np.array([s == stage for s in res["stage"]])
        if mask.sum() == 0:
            continue
        ax.plot(np.arange(1, 81), abs_fut[mask].mean(dim=0).numpy(),
                "--", label=stage)
    ax.set_xlabel("prediction horizon h (cycles ahead)")
    ax.set_ylabel("MAE")
    ax.set_title("Test-set MAE vs horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_mae_by_cell(report: dict, save_path: Path) -> None:
    """每只测试电池的整体轨迹 MAE，找最差的电池。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(report["by_cell"])
    vals = [report["by_cell"][c]["mae_fut"] for c in cells]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["tab:red" if v > np.median(vals) else "tab:blue" for v in vals]
    ax.bar(cells, vals, color=colors)
    ax.axhline(report["mae_fut"], color="k", ls="--", lw=1,
               label=f"overall {report['mae_fut']:.4f}")
    ax.set_ylabel("MAE (future trajectory)")
    ax.set_title("Per-cell test MAE")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_trajectories(res: dict, loader: DataLoader,
                      save_path: Path) -> None:
    """挑几只深度退化的测试电池，画"当前 SOH + 未来 80 步"预测 vs 真实。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 选 s3_aged 阶段且窗口数最多的 3 只电池（退化最明显、轨迹最好看）
    cells = sorted(set(res["cell_id"]))
    scores = []
    for cid in cells:
        mask = np.array([c == cid for c in res["cell_id"]])
        scores.append((int((np.array(res["stage"])[mask] == "s3_aged").sum()),
                       cid))
    chosen = [c for _, c in sorted(scores, reverse=True)[:3]]

    # 每只电池选最后（最老）的一个窗口
    pos_by_cell: dict[str, int] = {}
    for i, cid in enumerate(res["cell_id"]):
        pos_by_cell[cid] = max(pos_by_cell.get(cid, -1), res["pos"][i])

    n = min(3, len(chosen))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    for ax, cid in zip(axes, chosen):
        mask = np.array([c == cid for c in res["cell_id"]])
        p = pos_by_cell[cid]
        # 该窗口对应的真实未来轨迹：从预测结果里找（pos 对齐）
        idx = int(np.flatnonzero(mask & (np.array(res["pos"]) == p))[0])
        # pos 是排除 cycle 1 后的 0-based 位置：labels[0]=cycle 2，
        # 因此当前循环编号 k = p + 2，未来轨迹对应 cycle k+1 .. k+80
        k = p + 2
        ax.plot(range(k + 1, k + 81), res["y_fut"][idx].numpy(),
                "o-", ms=3, label="true", color="tab:green")
        ax.plot(range(k + 1, k + 81), res["s_fut"][idx].numpy(),
                "x--", ms=3, label="pred", color="tab:red")
        ax.axvline(k - 1, color="gray", ls=":", lw=1)
        ax.set_title(f"{cid} (pos={p})")
        ax.set_xlabel("cycle")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("SOH")
    fig.suptitle("Test-set future-trajectory predictions (last window per cell)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = __import__("io").TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="trainer.py 保存的 checkpoint.pt")
    parser.add_argument("--split-col", default="split_by_cell",
                        choices=["split_by_cell", "split_by_policy"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-name", default=None,
                        help="报告输出目录，默认取 checkpoint 所在目录")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_dir = (ROOT / "results/runs" / args.run_name if args.run_name
               else args.checkpoint.parent)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  checkpoint={args.checkpoint}", flush=True)
    print(f"ckpt 配置: {ckpt.get('config')}", flush=True)

    # 只加载测试集（20 只电池，全部窗口）
    splits = pd.read_parquet(ROOT / "data/processed/splits.parquet")
    windows = pd.read_parquet(ROOT / "data/processed/matr_windows.parquet")
    labels = pd.read_parquet(ROOT / "data/processed/matr_soh_labels.parquet")
    normalizer = ChannelNormalizer.load(ROOT / "data/processed/normalizer.json")

    test_cells = set(splits.loc[splits[args.split_col] == "test", "cell_id"])
    te = windows[windows["cell_id"].isin(test_cells)].reset_index(drop=True)
    test_ds = WindowDataset(te, labels, ROOT / "data/external/matr",
                            normalizer=normalizer,
                            cache_size=te["cell_id"].nunique())
    test_ds.preload_all(verbose=True)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_window, num_workers=0)
    print(f"测试集: {len(te):,} 窗口 / {te['cell_id'].nunique()} 只电池",
          flush=True)

    model = WorldModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    res = predict(model, loader, device)
    report = report_metrics(res)
    report["checkpoint"] = str(args.checkpoint)
    report["split_col"] = args.split_col

    # 画图
    plot_mae_by_horizon(res, fig_dir / "mae_by_horizon.png")
    plot_mae_by_cell(report, fig_dir / "mae_by_cell.png")
    plot_trajectories(res, loader, fig_dir / "trajectory_samples.png")

    json_path = run_dir / "test_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    print(f"\n===== 测试集结果（{report['n_windows']:,} 窗口 / "
          f"{report['n_cells']} 只电池）=====", flush=True)
    print(f"MAE 当前 SOH : {report['mae_cur']:.4f}", flush=True)
    print(f"MAE 未来轨迹 : {report['mae_fut']:.4f}  "
          f"(cell-avg {report['mae_fut_cell_avg']:.4f})", flush=True)
    print(f"MAE h=1 / 20 / 80 : {report['mae_h1']:.4f} / "
          f"{report['mae_h20']:.4f} / {report['mae_h80']:.4f}", flush=True)
    print("按老化阶段:", flush=True)
    for stage, v in report["by_stage"].items():
        print(f"  {stage:10s} n={v['n_windows']:>6,}  "
              f"mae_cur={v['mae_cur']:.4f}  mae_fut={v['mae_fut']:.4f}",
              flush=True)
    print(f"报告: {json_path}", flush=True)
    print(f"图:   {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
