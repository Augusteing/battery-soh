"""temperature_soh 训练入口：电压预测预训练 + SOH 回归微调（4 通道）。

两阶段流程（与 partial_soh / 原论文一致，仅输入改为 4 通道）：

    阶段 1（预训练）：用电压预测任务训练编码器 + 电压头。
        目标 = 预测每个时间步的“下一步电压”（密集监督）
               + 未来 7% 容量窗的电压曲线。

    阶段 2（微调）：  保留编码器权重，换 SOH 头，回归标量 SOH。

默认超参数与 partial_soh 对齐：lr=1e-3, batch_size 可调，
支持 --max-samples / --epochs 缩小规模做冒烟测试。

运行：

```powershell
# 冒烟测试（小规模，几分钟内跑完）
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/trainer.py --max-samples 300 --epochs 2

# 全量训练（默认）
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/trainer.py
```
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[3]
TR_DIR = Path(__file__).resolve().parent
if str(TR_DIR) not in sys.path:
    sys.path.insert(0, str(TR_DIR))

from dataset import TemperatureSohDataset  # noqa: E402
from model import TemperatureSohLSTM  # noqa: E402

DEFAULT_INDEX = ROOT / "data" / "processed" / "temperature_soh" / "segment_index.parquet"
DEFAULT_MAT_DIR = ROOT / "data" / "external" / "matr"
DEFAULT_OUT = ROOT / "models" / "temperature_soh" / "temperature_soh.pt"


@dataclass
class TrainingConfig:
    """训练超参数（默认对齐 partial_soh / 原论文）。"""

    batch_size: int = 512
    pretrain_epochs: int = 50
    finetune_epochs: int = 50
    lr: float = 1e-3
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    """固定随机种子，保证可复现。"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _save_model(path: Path, model: nn.Module) -> None:
    """保存训练好的模型权重。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)


def pretrain_voltage(
    model: TemperatureSohLSTM,
    loader: DataLoader,
    epochs: int,
    lr: float,
    grad_clip: float,
    device: torch.device,
) -> float:
    """在电压预测任务上预训练，返回最后一个 epoch 的平均损失。

    监督信号有两部分（对应论文）：
      1. 观测窗内“下一步电压”：pred[:, :-1] 对齐 y = V[1:101]；
      2. 未来 7% 容量窗电压：从最终状态直接预测 36 点曲线，
         监督 = 未来窗的真实电压（x_future 的 V 通道）。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        total_future = 0.0
        t0 = time.perf_counter()
        for step, (x, y, x_future) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)          # (B, 100) = V[1:101]
            x_future = x_future.to(device)  # (B, 36, 4)

            # 一次编码同时得到“下一步电压”和“未来窗电压”。
            pred, future_pred = model.voltage_and_future(x)
            loss = loss_fn(pred[:, :-1], y)
            future_loss = loss_fn(future_pred, x_future[:, :, 1])
            loss = loss + future_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += float(loss.item()) * x.size(0)
            total_future += float(future_loss.item()) * x.size(0)
            n += x.size(0)

        final_loss = total / n
        print(
            f"  [pretrain] epoch {epoch:3d}/{epochs}  "
            f"loss={final_loss:.6f}  future={total_future / n:.6f}  "
            f"({time.perf_counter() - t0:.1f}s)"
        )
    return final_loss


def finetune_soh(
    model: TemperatureSohLSTM,
    loader: DataLoader,
    epochs: int,
    lr: float,
    grad_clip: float,
    device: torch.device,
) -> float:
    """在 SOH 回归任务上微调，返回最后一个 epoch 的平均损失。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        t0 = time.perf_counter()
        for step, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)          # (B,)

            pred = model.soh_predict(x)  # (B,)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += float(loss.item()) * x.size(0)
            n += x.size(0)

        final_loss = total / n
        print(
            f"  [finetune] epoch {epoch:3d}/{epochs}  "
            f"loss={final_loss:.6f}  ({time.perf_counter() - t0:.1f}s)"
        )
    return final_loss


@torch.no_grad()
def evaluate_mae(
    model: TemperatureSohLSTM,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """在给定 DataLoader 上计算 SOH 预测的平均绝对误差（MAE）。"""
    model.eval()
    errors: list[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model.soh_predict(x)
        errors.append((pred - y).abs().cpu().numpy())
    model.train()
    if not errors:
        return float("nan")
    return float(np.concatenate(errors).mean())


def main() -> None:
    """两阶段训练：预训练 -> 微调 -> 测试 MAE。"""
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mat-dir", type=Path, default=DEFAULT_MAT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--pretrain-epochs", type=int, default=TrainingConfig.pretrain_epochs)
    parser.add_argument("--finetune-epochs", type=int, default=TrainingConfig.finetune_epochs)
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="每个 split 只取前 N 个片段（冒烟测试用）")
    parser.add_argument("--preload", action="store_true",
                        help="预加载充电曲线到内存（全量训练建议开启）")
    args = parser.parse_args()

    cfg = TrainingConfig(
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        lr=args.lr,
        seed=args.seed,
    )
    device = cfg.device
    _set_seed(cfg.seed)
    print(f"设备: {device}")

    # ---- 阶段 1：电压预测预训练（train split）----
    print("=" * 60)
    print("阶段 1：电压预测预训练（4 通道）")
    print("=" * 60)
    pretrain_ds = TemperatureSohDataset(
        args.index, args.mat_dir, split="train", task="pretrain",
        preload=args.preload,
    )
    if args.max_samples is not None:
        pretrain_ds = Subset(pretrain_ds, list(range(min(args.max_samples, len(pretrain_ds)))))
    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False,
    )

    model = TemperatureSohLSTM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}  预训练样本数: {len(pretrain_ds):,}")
    pretrain_voltage(
        model, pretrain_loader, cfg.pretrain_epochs, cfg.lr, cfg.grad_clip, device
    )

    # ---- 阶段 2：SOH 回归微调（train split）----
    print("=" * 60)
    print("阶段 2：SOH 回归微调")
    print("=" * 60)
    soh_ds = TemperatureSohDataset(
        args.index, args.mat_dir, split="train", task="soh",
        preload=args.preload,
    )
    if args.max_samples is not None:
        soh_ds = Subset(soh_ds, list(range(min(args.max_samples, len(soh_ds)))))
    soh_loader = DataLoader(
        soh_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False,
    )
    finetune_soh(
        model, soh_loader, cfg.finetune_epochs, cfg.lr, cfg.grad_clip, device
    )

    # ---- 评估：test split MAE ----
    print("=" * 60)
    print("测试集评估（test split，未见电池）")
    print("=" * 60)
    test_ds = TemperatureSohDataset(args.index, args.mat_dir, split="test", task="soh")
    if args.max_samples is not None:
        test_ds = Subset(test_ds, list(range(min(args.max_samples, len(test_ds)))))
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=False,
    )
    test_mae = evaluate_mae(model, test_loader, device)
    print(f"test MAE = {test_mae:.4f}  (样本数 {len(test_ds):,})")

    _save_model(args.out, model)
    print(f"模型已保存: {args.out}")


if __name__ == "__main__":
    main()
