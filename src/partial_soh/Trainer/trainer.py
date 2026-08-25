"""partial_soh 训练入口：电压预测预训练 + SOH 回归微调。

两阶段流程（对应论文）：

    阶段 1（预训练）：用电压预测任务训练编码器 + 电压头。
        目标 = 预测每个时间步的“下一步电压”，监督是密集的。

    阶段 2（微调）：  保留编码器权重，换 SOH 头，回归标量 SOH。

默认超参数来自论文 Table 1：lr=1e-3, batch_size=20000, 50+50 epochs。
为了能在本地冒烟测试，命令行支持 --max-samples / --epochs 缩小规模。
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

# 让本文件能 import 到 Trainer 包里的模块。
ROOT = Path(__file__).resolve().parents[3]
TR_DIR = ROOT / "src" / "partial_soh" / "Trainer"
if str(TR_DIR) not in sys.path:
    sys.path.insert(0, str(TR_DIR))

from dataset import PartialSohDataset  # noqa: E402
from model import PartialSohLSTM  # noqa: E402


@dataclass
class TrainingConfig:
    """训练超参数（默认对齐论文 Table 1）。"""

    batch_size: int = 20000
    pretrain_epochs: int = 50
    finetune_epochs: int = 50
    lr: float = 1e-3
    grad_clip: float = 1.0
    seed: int = 42

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    """固定随机种子，保证可复现。"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _ckpt_path(model_out: Path) -> Path:
    """checkpoint 文件与模型同目录、同名，后缀改为 .ckpt。"""
    return model_out.with_name(model_out.stem + ".ckpt")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    stage: str,
    epoch: int,
) -> None:
    """保存可续训的 checkpoint：模型权重 + 优化器状态 + 阶段/epoch 信息。"""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage": stage,   # "pretrain" 或 "finetune"
            "epoch": epoch,   # 刚完成的 epoch 编号
        },
        path,
    )


def _load_checkpoint(
    path: Path, model: nn.Module, lr: float
) -> tuple[dict, torch.optim.Optimizer]:
    """从 checkpoint 恢复模型权重和优化器状态。"""
    # checkpoint 是我们自己保存的（只含张量和基础类型），可以安全用 weights_only。
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt, optimizer


def pretrain_voltage(
    model: PartialSohLSTM,
    loader: DataLoader,
    epochs: int,
    lr: float,
    grad_clip: float,
    device: torch.device,
    ckpt_path: Path | None = None,
    start_epoch: int = 1,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """在电压预测任务上预训练，返回最后一个 epoch 的平均损失。

    支持断点续训：start_epoch 指定从第几个 epoch 开始，
    optimizer 为之前保存的优化器状态；每个 epoch 结束会保存 checkpoint。
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    final_loss = float("nan")
    for epoch in range(start_epoch, epochs + 1):
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)  # y 形状 (B, 100)，是 V[1:101]

            pred = model.voltage_predict(x)  # (B, 101)
            # 只用前 100 步和下一步电压比较。
            loss = loss_fn(pred[:, :-1], y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += float(loss.item()) * x.size(0)
            n += x.size(0)

        final_loss = total / n
        print(f"  [pretrain] epoch {epoch:3d}/{epochs}  loss={final_loss:.6f}")
        if ckpt_path is not None:
            _save_checkpoint(ckpt_path, model, optimizer, "pretrain", epoch)
            print(f"  [checkpoint] saved -> {ckpt_path}")

    return final_loss


def finetune_soh(
    model: PartialSohLSTM,
    loader: DataLoader,
    epochs: int,
    lr: float,
    grad_clip: float,
    device: torch.device,
    ckpt_path: Path | None = None,
    start_epoch: int = 1,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """在 SOH 回归任务上微调，返回最后一个 epoch 的平均损失。

    支持断点续训，逻辑与 pretrain_voltage 相同。
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    final_loss = float("nan")
    for epoch in range(start_epoch, epochs + 1):
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)  # y 形状 (B,)

            pred = model.soh_predict(x)  # (B,)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += float(loss.item()) * x.size(0)
            n += x.size(0)

        final_loss = total / n
        print(f"  [finetune] epoch {epoch:3d}/{epochs}  loss={final_loss:.6f}")
        if ckpt_path is not None:
            _save_checkpoint(ckpt_path, model, optimizer, "finetune", epoch)
            print(f"  [checkpoint] saved -> {ckpt_path}")

    return final_loss


@torch.no_grad()
def evaluate_soh(
    model: PartialSohLSTM,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """在 SOH 任务上评估 MAE / RMSE（以百分比报告）。"""
    model.eval()
    errs: list[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model.soh_predict(x)
        err = (pred - y).cpu().numpy()
        errs.append(err)

    err = np.concatenate(errs)
    mae = float(np.abs(err).mean()) * 100.0
    rmse = float(np.sqrt(np.mean(err**2))) * 100.0
    return {"mae_pct": mae, "rmse_pct": rmse}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "processed" / "partial_segments_index.parquet")
    parser.add_argument("--mat-dir", type=Path, default=ROOT / "data" / "external" / "matr")
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None, help="冒烟测试时只取前 N 个样本")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preload", action="store_true", help="初始化时把充电曲线全部读进内存")
    parser.add_argument(
        "--no-pretrain",
        action="store_true",
        help="跳过电压预测预训练，随机初始化直接训练 SOH（对照基线）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 model-out 同目录的 .ckpt 断点续训",
    )
    parser.add_argument("--model-out", type=Path, default=ROOT / "models" / "partial_soh_lstm.pt")
    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        lr=args.lr,
        seed=args.seed,
    )
    device = config.device
    _set_seed(config.seed)

    print(f"device: {device}")
    print(f"batch_size: {config.batch_size}")

    # 两个任务各建一个 Dataset，都用 train 划分。直接训练模式不需要预训练数据集。
    soh_ds = PartialSohDataset(
        args.index, args.mat_dir, split="train", task="soh", preload=args.preload
    )
    if not args.no_pretrain:
        pretrain_ds = PartialSohDataset(
            args.index, args.mat_dir, split="train", task="pretrain", preload=args.preload
        )

    if args.max_samples is not None:
        n = min(args.max_samples, len(soh_ds))
        soh_ds = Subset(soh_ds, range(n))
        if not args.no_pretrain:
            pretrain_ds = Subset(pretrain_ds, range(n))
        print(f"冒烟测试：只取前 {n} 个样本")

    if not args.no_pretrain:
        pretrain_loader = DataLoader(
            pretrain_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )
    soh_loader = DataLoader(
        soh_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = PartialSohLSTM().to(device)
    ckpt_path = _ckpt_path(args.model_out)
    pretrained_out = args.model_out.with_name(args.model_out.stem + "_pretrained.pt")

    # ---- 断点续训：如果存在 .ckpt 且开了 --resume，恢复模型与优化器 ----
    resume_stage: str | None = None
    resume_epoch = 0
    resume_optimizer: torch.optim.Optimizer | None = None
    if args.resume and ckpt_path.exists():
        ckpt, resume_optimizer = _load_checkpoint(ckpt_path, model, config.lr)
        resume_stage = ckpt["stage"]
        resume_epoch = int(ckpt["epoch"])
        print(f"从 checkpoint 恢复: stage={resume_stage}, 已完成 epoch={resume_epoch}")

    # ---- 阶段 1：预训练（直接训练模式跳过） ----
    if args.no_pretrain:
        print("\n== 直接训练模式（对照基线）：跳过预训练 ==")
    else:
        pretrain_done = resume_stage == "pretrain" and resume_epoch >= config.pretrain_epochs
        if pretrain_done:
            print("\n== 阶段 1：预训练已完成，跳过 ==")
        else:
            print("\n== 阶段 1：电压预测预训练 ==")
            t0 = time.perf_counter()
            start = resume_epoch + 1 if resume_stage == "pretrain" else 1
            opt = resume_optimizer if resume_stage == "pretrain" else None
            pretrain_voltage(
                model,
                pretrain_loader,
                epochs=config.pretrain_epochs,
                lr=config.lr,
                grad_clip=config.grad_clip,
                device=device,
                ckpt_path=ckpt_path,
                start_epoch=start,
                optimizer=opt,
            )
            print(f"预训练耗时: {time.perf_counter() - t0:.1f}s")
        # 预训练完成就保存一份纯权重，便于单独使用/对比。
        torch.save(model.state_dict(), pretrained_out)
        print(f"预训练权重: {pretrained_out}")
        # 预训练结束后进入微调，微调用新的优化器。
        resume_stage = "finetune"
        resume_epoch = 0
        resume_optimizer = None

    # ---- 阶段 2：SOH 回归微调 ----
    finetune_done = resume_stage == "finetune" and resume_epoch >= config.finetune_epochs
    if finetune_done:
        print("\n== 阶段 2：微调已完成，跳过 ==")
    else:
        print("\n== 阶段 2：SOH 回归微调 ==")
        t0 = time.perf_counter()
        start = resume_epoch + 1 if resume_stage == "finetune" else 1
        opt = resume_optimizer if resume_stage == "finetune" else None
        finetune_soh(
            model,
            soh_loader,
            epochs=config.finetune_epochs,
            lr=config.lr,
            grad_clip=config.grad_clip,
            device=device,
            ckpt_path=ckpt_path,
            start_epoch=start,
            optimizer=opt,
        )
        print(f"微调耗时: {time.perf_counter() - t0:.1f}s")

    print("\n== 训练集评估 ==")
    train_metrics = evaluate_soh(model, soh_loader, device)
    print(f"  MAE  = {train_metrics['mae_pct']:.4f}%")
    print(f"  RMSE = {train_metrics['rmse_pct']:.4f}%")

    print("\n== 测试集评估 ==")
    test_ds = PartialSohDataset(
        args.index, args.mat_dir, split="test", task="soh", preload=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_metrics = evaluate_soh(model, test_loader, device)
    print(f"  MAE  = {test_metrics['mae_pct']:.4f}%")
    print(f"  RMSE = {test_metrics['rmse_pct']:.4f}%")
    print(f"  测试样本数 = {len(test_ds)}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    print(f"\nsaved: {args.model_out}")


if __name__ == "__main__":
    main()
