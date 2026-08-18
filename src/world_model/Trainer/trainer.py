"""训练循环模块（M3 · 第六步）：把 dataset / model / loss 串成可训练管线。

一句话：trainer.py 负责"喂数据 -> 算损失 -> 反向传播 -> 更新权重 -> 每轮
在验证集上打分"，并把结果（权重、指标、学习曲线图）存到 results/ 下。

软件工程分工：
  - dataset.py   ：给一个窗口索引，返回 (X, u, y_cur, y_fut, ir_0, ir_k)；
  - model.py     ：给一个 batch，返回 (s_cur, s_fut)；
  - loss.py      ：给预测和标签，返回 total 和每个损失分量；
  - trainer.py   ：只负责把它们串起来 + 存结果，不包含模型/损失逻辑；
  - evaluate.py  ：（下一步）用测试集做完整评估报告，复用本文件的 evaluate()。

关于 max-samples：
  默认 0 = 每 epoch 采全部训练窗口（论文 §4 口径）。CPU 上全量 6.9 万
  窗口一个 epoch 约 20 分钟（实测），需要快速迭代时用 --max-samples 3000
  限制每轮样本量（逆频率采样本来就是重复采样，只是再截断而已）。

训练配置（论文 §4，全部写入默认值）：
  - 优化器 Adam，lr=1e-3，weight decay=1e-4；
  - 梯度裁剪：L2 范数 1.0；
  - 早停：验证 MAE 连续 15 轮无提升即停，最多 100 轮；
  - 逆频率采样（论文 Imbalance handling）平衡四档老化阶段；
  - 主配置单 DataLoader 全量打乱；EWC 分阶段配置（§5）后续单独实现。

用法：
    python "src/world_model/Trainer/trainer.py" --max-samples 3000 --epochs 10

输出（results/runs/<run_name>/）：
  checkpoint.pt       最优验证集的模型权重 + 配置 + 指标
  metrics.json        每 epoch 的全部指标（可画图/对比）
  learning_curve.png  训练损失 + 验证 MAE 曲线
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Windows 下 torch 与 matplotlib 各自带一份 OpenMP 运行时，直接 import 会
# 报 "OMP: Error #15"。必须在 import torch 之前声明允许重复加载。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/world_model/Trainer"))
sys.path.insert(0, str(ROOT / "src/world_model/DataLoader"))

from dataset import WindowDataset, make_stage_weights     # noqa: E402
from normalize import ChannelNormalizer                    # noqa: E402
from model import WorldModel                               # noqa: E402
from loss import WorldModelLoss                            # noqa: E402


def collate_window(batch: list[dict]) -> dict:
    """把 dataset 返回的样本列表拼成一个 batch 张量 dict。

    dataset 每个样本是 dict（X 是 numpy、标签是标量/数组），DataLoader 默认
    的拼接函数也能处理，但类型不统一（可能出现 float64 混入）。这里显式
    统一成 float32，并顺手把 cell_id / stage 以列表形式带出来备用。
    """
    return {
        "X": torch.stack([torch.as_tensor(b["X"], dtype=torch.float32)
                          for b in batch]),
        "u": torch.tensor([b["u"] for b in batch], dtype=torch.float32),
        "y_cur": torch.tensor([b["y_cur"] for b in batch], dtype=torch.float32),
        "y_fut": torch.stack([torch.as_tensor(b["y_fut"], dtype=torch.float32)
                              for b in batch]),
        "ir_0": torch.tensor([b["ir_0"] for b in batch], dtype=torch.float32),
        "ir_k": torch.tensor([b["ir_k"] for b in batch], dtype=torch.float32),
        "cell_id": [b["cell_id"] for b in batch],
        "stage": [b["stage"] for b in batch],
        "pos": [int(b["pos"]) for b in batch],
    }


def parse_stage_boost(text: str) -> dict[str, float] | None:
    """把 "s3_aged:8,s2_mild:3" 解析成 {阶段: 倍数}；空串返回 None。"""
    if not text.strip():
        return None
    boost: dict[str, float] = {}
    for item in text.split(","):
        stage, _, mult = item.strip().partition(":")
        if not stage or not mult:
            raise ValueError(f"无法解析 --stage-boost 片段: {item!r}")
        boost[stage.strip()] = float(mult)
    return boost


def build_loaders(splits: pd.DataFrame, windows: pd.DataFrame,
                  labels: pd.DataFrame, mat_dir: Path,
                  normalizer, split_col: str = "split_by_cell",
                  batch_size: int = 32, max_samples: int = 0,
                  max_val_samples: int = 2000, seed: int = 42,
                  W: int = 30, H: int = 80, cache_size: int = 0,
                  preload: bool = True,
                  stage_boost: dict[str, float] | None = None):
    """按 cell 划分构建 训练/验证/测试 三个 DataLoader。

    - 训练集：用逆频率采样权重（dataset.make_stage_weights，论文的
      Imbalance handling）做 WeightedRandomSampler；max_samples<=0 表示
      每 epoch 采全部窗口（论文 §4 口径），>0 则是 CPU 快速迭代用的上限；
    - 验证集：随机抽 max_val_samples 个窗口（固定 seed 保证可复现），
      不重采样（评估必须忠实反映数据分布）；
    - 测试集：完整保留，训练阶段不碰（避免"看着测试集调参"的信息泄漏），
      留给 evaluate.py 用。

    cache_size=0 表示"自动 = 该集合的电池数"（全部曲线驻留内存，约 2GB），
    训练前 preload=True 会一次性载完，避免随机采样导致的反复读盘。
    """
    def cell_subset(split_name: str) -> pd.DataFrame:
        cells = set(splits.loc[splits[split_col] == split_name, "cell_id"])
        return windows[windows["cell_id"].isin(cells)].reset_index(drop=True)

    tr, va, te = cell_subset("train"), cell_subset("val"), cell_subset("test")

    # 训练集：逆频率权重 + 每 epoch 限制样本量
    weights = make_stage_weights(tr, stage_boost=stage_boost)
    n_train = len(tr) if max_samples <= 0 else min(max_samples, len(tr))
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.float64),
        num_samples=n_train, replacement=True,
    )
    train_ds = WindowDataset(
        tr, labels, mat_dir, normalizer=normalizer, W=W, H=H,
        cache_size=cache_size or tr["cell_id"].nunique())
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, collate_fn=collate_window,
                              num_workers=0)

    # 验证集：固定 seed 抽一个子集，保证每次实验可比
    rng = np.random.RandomState(seed)
    val_idx = rng.choice(len(va), min(max_val_samples, len(va)),
                         replace=False)
    val_ds = WindowDataset(
        va.iloc[val_idx], labels, mat_dir, normalizer=normalizer, W=W, H=H,
        cache_size=cache_size or va["cell_id"].nunique())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_window, num_workers=0)

    test_ds = WindowDataset(
        te, labels, mat_dir, normalizer=normalizer, W=W, H=H,
        cache_size=cache_size or te["cell_id"].nunique())

    if preload:                                 # 训练前把曲线全部载入内存
        train_ds.preload_all()
        val_ds.preload_all()
        test_ds.preload_all()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_window, num_workers=0)

    return train_loader, val_loader, test_loader, {
        "train_windows": len(tr), "val_windows": len(va),
        "test_windows": len(te), "train_cells": tr["cell_id"].nunique(),
        "val_cells": va["cell_id"].nunique(), "test_cells": te["cell_id"].nunique(),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader,
             loss_fn: WorldModelLoss, device: torch.device) -> dict:
    """在给定 DataLoader 上计算损失分量 + 关键 MAE 指标。

    关键细节：评估必须 model.eval()，否则 BatchNorm 会用当前 batch 的统计量
    而不是历史滑动平均，指标会失真；torch.no_grad() 关闭梯度记录以省内存。
    MAE 指标：
      val_mae_cur  ：当前 SOH 预测的平均绝对误差
      val_mae_fut  ：未来 80 步轨迹的整体 MAE（论文报告的主指标）
      val_mae_h1 / h20 / h80 ：只看第 1/20/80 步预测，观察误差如何随
        滚动距离累积（远视距误差大是 rollout 模型的正常现象）
    """
    model.eval()
    loss_sums: dict[str, float] = {}
    n_seen = 0
    s_cur_list, s_fut_list, y_cur_list, y_fut_list = [], [], [], []

    for batch in loader:
        X = batch["X"].to(device)
        u = batch["u"].to(device)
        y_cur = batch["y_cur"].to(device)
        y_fut = batch["y_fut"].to(device)
        ir_0 = batch["ir_0"].to(device)
        ir_k = batch["ir_k"].to(device)

        s_cur, s_fut = model(X, u)
        losses = loss_fn(s_cur, s_fut, y_cur, y_fut, ir_0, ir_k)
        bs = X.shape[0]
        for k, v in losses.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach()) * bs
        n_seen += bs
        s_cur_list.append(s_cur.detach().cpu())
        s_fut_list.append(s_fut.detach().cpu())
        y_cur_list.append(y_cur.detach().cpu())
        y_fut_list.append(y_fut.detach().cpu())

    p_cur = torch.cat(s_cur_list)
    p_fut = torch.cat(s_fut_list)
    t_cur = torch.cat(y_cur_list)
    t_fut = torch.cat(y_fut_list)

    mae_fut = (p_fut - t_fut).abs()          # (N, H)
    metrics = {f"val_{k}": v / n_seen for k, v in loss_sums.items()}
    metrics["val_mae_cur"] = float((p_cur - t_cur).abs().mean())
    metrics["val_mae_fut"] = float(mae_fut.mean())
    for h in (1, 20, 80):
        metrics[f"val_mae_h{h}"] = float(mae_fut[:, h - 1].mean())

    model.train()
    return metrics


def train(model: torch.nn.Module, train_loader: DataLoader,
          val_loader: DataLoader, loss_fn: WorldModelLoss,
          device: torch.device, epochs: int, lr: float,
          weight_decay: float, run_dir: Path, seed: int = 42,
          grad_clip: float = 1.0, log_every: int = 50, patience: int = 15,
          lambda_phys: float = 0.1) -> tuple[dict, Path]:
    """主训练循环：epoch 内迭代 -> 每轮结束在验证集打分 -> 存最优权重。

    返回 (history, best_checkpoint_path)。
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    history: list[dict] = []
    best_score = float("inf")
    best_path = run_dir / "checkpoint.pt"
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        train_total, n_batches = 0.0, 0

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(device)
            u = batch["u"].to(device)
            y_cur = batch["y_cur"].to(device)
            y_fut = batch["y_fut"].to(device)
            ir_0 = batch["ir_0"].to(device)
            ir_k = batch["ir_k"].to(device)

            optimizer.zero_grad()
            s_cur, s_fut = model(X, u)
            losses = loss_fn(s_cur, s_fut, y_cur, y_fut, ir_0, ir_k)
            losses["total"].backward()
            # 梯度裁剪：rollout 80 步的梯度可能很大，限制范数防训练震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_total += float(losses["total"].detach())
            n_batches += 1

            if step % log_every == 0 or step == len(train_loader):
                speed = (time.perf_counter() - t0) / step
                eta = speed * (len(train_loader) - step) / 60
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={train_total/n_batches:.4f} "
                      f"({speed:.2f}s/step, ETA {eta:.1f}min)", flush=True)

        epoch_time = time.perf_counter() - t0
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        row = {"epoch": epoch, "train_total": train_total / n_batches,
               "epoch_time_s": round(epoch_time, 1)}
        row.update({k: round(v, 6) for k, v in val_metrics.items()})
        history.append(row)

        score = val_metrics["val_mae_fut"]          # 早停看主指标
        improved = score < best_score - 1e-5
        if improved:
            best_score = score
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {"lr": lr, "lambda_phys": lambda_phys,
                           "epoch": epoch},
                "best_metrics": row,
            }, best_path)
        else:
            no_improve += 1

        print(f"[epoch {epoch}] train_total={row['train_total']:.4f} "
              f"val_mae_cur={row['val_mae_cur']:.4f} "
              f"val_mae_fut={row['val_mae_fut']:.4f} "
              f"val_mae_h80={row['val_mae_h80']:.4f} "
              f"({epoch_time:.0f}s, best={best_score:.4f})", flush=True)

        if patience and no_improve >= patience:
            print(f"early stop: {patience} 轮无提升，停止训练", flush=True)
            break

    return {"history": history, "best_score": best_score}, best_path


def plot_learning_curve(history: list[dict], save_path: Path) -> None:
    """画训练损失与验证 MAE 的曲线，直观看到模型是否在收敛。"""
    import matplotlib
    matplotlib.use("Agg")                       # 无界面后端，只存文件
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, [h["train_total"] for h in history],
             "o-", label="train loss (total)", color="tab:blue")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()                           # 右轴：验证 MAE
    ax2.plot(epochs, [h["val_mae_fut"] for h in history],
             "s--", label="val MAE (fut, all H)", color="tab:red")
    ax2.plot(epochs, [h["val_mae_h80"] for h in history],
             "s:", label="val MAE (h=80)", color="tab:orange")
    ax2.set_ylabel("val MAE", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "buffer"):           # Windows 控制台 GBK 兼容
        sys.stdout = __import__("io").TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=100,
                        help="最大 epoch 数（论文 §4）")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="论文 §4 固定 batch size 32")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="每 epoch 训练采样数；0=全部窗口（论文口径）")
    parser.add_argument("--max-val-samples", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-col", default="split_by_cell",
                        choices=["split_by_cell", "split_by_policy"])
    parser.add_argument("--lambda-phys", type=float, default=0.1,
                        help="物理损失权重；设 0 即消融掉物理约束")
    parser.add_argument("--stage-boost", default="",
                        help="老化阶段采样放大，逗号分隔 阶段:倍数，"
                             "如 s3_aged:8,s2_mild:3")
    parser.add_argument("--patience", type=int, default=15,
                        help="验证指标连续几轮无提升就早停（论文 §4）；0=关闭")
    parser.add_argument("--log-every", type=int, default=50,
                        help="每多少步打印一次训练进度")
    parser.add_argument("--cache-size", type=int, default=0,
                        help="每集合 LRU 容量；0=自动等于电池数（全部驻留内存）")
    parser.add_argument("--no-preload", action="store_true",
                        help="不预载电池曲线（内存紧张时用）")
    parser.add_argument("--run-name", default=None,
                        help="输出目录名，默认用时间戳")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results/runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  torch={torch.__version__}", flush=True)
    print(f"run_dir={run_dir}", flush=True)

    # 读数据 + 标准化参数（normalize.py 在训练集上拟合，验证/测试直接复用）
    splits = pd.read_parquet(ROOT / "data/processed/splits.parquet")
    windows = pd.read_parquet(ROOT / "data/processed/matr_windows.parquet")
    labels = pd.read_parquet(ROOT / "data/processed/matr_soh_labels.parquet")
    normalizer = ChannelNormalizer.load(ROOT / "data/processed/normalizer.json")

    train_loader, val_loader, test_loader, info = build_loaders(
        splits, windows, labels, ROOT / "data/external/matr", normalizer,
        split_col=args.split_col, batch_size=args.batch_size,
        max_samples=args.max_samples, max_val_samples=args.max_val_samples,
        seed=args.seed, cache_size=args.cache_size,
        preload=not args.no_preload,
        stage_boost=parse_stage_boost(args.stage_boost))
    print("数据规模:", info, flush=True)

    model = WorldModel().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    loss_fn = WorldModelLoss(lambda_phys=args.lambda_phys)

    result, best_path = train(
        model, train_loader, val_loader, loss_fn, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        run_dir=run_dir,
        seed=args.seed, patience=args.patience, log_every=args.log_every,
        lambda_phys=args.lambda_phys)

    # 落盘：指标 JSON + 学习曲线图 + 训练配置
    metrics_path = run_dir / "metrics.json"
    config = vars(args) | {"device": str(device), "run_name": run_name}
    metrics_path.write_text(
        json.dumps({"config": config, **result},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    plot_learning_curve(result["history"], run_dir / "learning_curve.png")

    print(f"\n训练完成。最优验证 val_mae_fut={result['best_score']:.4f}",
          flush=True)
    print(f"checkpoint: {best_path}", flush=True)
    print(f"metrics:    {metrics_path}", flush=True)
    print(f"curve:      {run_dir / 'learning_curve.png'}", flush=True)


if __name__ == "__main__":
    main()
