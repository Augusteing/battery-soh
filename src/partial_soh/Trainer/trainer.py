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
import math
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

from consistency import (  # noqa: E402
    SameCycleBatchSampler,
    same_cycle_consistency_loss,
)
from dataset import MemmapSohDataset, PartialSohDataset  # noqa: E402
from model import PartialSohLSTM  # noqa: E402
from ssl_tasks import mask_voltage, masked_reconstruction_loss  # noqa: E402


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


def _get_group_ids(ds) -> np.ndarray:
    """从 Dataset（或 Subset）取出每个样本的循环编号。

    Subset 是 PyTorch 对 Dataset 的切片包装（--max-samples 冒烟测试时
    会用到），需要先还原到原 Dataset，再按子集下标切片，
    才能拿到与子集对应的 group_ids。
    """
    if isinstance(ds, Subset):
        return ds.dataset.group_ids()[ds.indices]
    return ds.group_ids()


def _memmap_safe_collate(batch):
    """兼容两种 batch 形态的 collate（配合 MemmapSohDataset）。

    默认的 default_collate 会逐个样本地组装一个 4096 元素的列表，
    Python 开销约 50~80ms/step。memmap 数据集已经在 __getitems__
    里一次性堆叠成张量，所以可能收到两种输入：

      1) (x, y) 或 (x, y, x_future) 已堆叠张量：batch[0] 形状 (B, 101, 3)；
      2) [(x_i, y_i), ...] 样本列表（如 Subset 逐样本回退）：
         batch[0] 形状 (101, 3) —— 走 default_collate。

    用 batch[0].ndim == 3 区分这两种情况。
    """
    if (
        isinstance(batch, (tuple, list))
        and torch.is_tensor(batch[0])
        and batch[0].ndim == 3
        and all(torch.is_tensor(b) for b in batch)
    ):
        return batch
    return torch.utils.data.default_collate(batch)


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
    recon_lambda: float = 0.0,
    mask_ratio: float = 0.3,
    seed: int = 0,
    accum_steps: int = 1,
) -> float:
    """在电压预测任务上预训练，返回最后一个 epoch 的平均损失。

    支持断点续训：start_epoch 指定从第几个 epoch 开始，
    optimizer 为之前保存的优化器状态；每个 epoch 结束会保存 checkpoint。

    创新 2（扩展自监督）：当 recon_lambda > 0 时，额外执行掩码电压
    重建任务——随机遮掉一部分电压点，用上下文把它们补回来。

    论文对齐：预训练监督 = 观测窗内“下一步电压” + 未来 7% 容量窗的
    自回归电压 rollout（voltage_rollout）。accum_steps 用于用较小的
    batch 梯度累积模拟论文的 batch=20,000。
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    # 掩码重建的随机数生成器：固定种子，保证可复现。
    mask_gen = torch.Generator().manual_seed(seed)

    final_loss = float("nan")
    for epoch in range(start_epoch, epochs + 1):
        total, n = 0.0, 0
        total_recon = 0.0
        total_future = 0.0
        optimizer.zero_grad()
        for step, (x, y, x_future) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)  # y 形状 (B, 100)，是 V[1:101]
            x_future = x_future.to(device)  # (B, 36, 3)

            recon_loss_val = 0.0
            future_loss_val = 0.0
            pred = model.voltage_predict(x)  # (B, 101)
            # 损失 1：观测窗内“下一步电压”的密集监督。
            # 除以 accum_steps：累积 N 步后再更新，等效于大 batch。
            loss = loss_fn(pred[:, :-1], y) / accum_steps
            # 损失 2：未来 7% 容量窗的自回归电压预测（论文对齐）。
            rollout = model.voltage_rollout(x, x_future)  # (B, 36)
            future_loss_val = loss_fn(rollout, x_future[:, :, 1])
            loss = loss + future_loss_val / accum_steps
            if recon_lambda > 0:
                # 1) 随机遮掉 mask_ratio 比例的电压点；
                # 2) 用损坏后的输入走一次前向，重建被遮住的电压；
                # 3) 只统计掩码位置的重建误差。
                x_c, mask = mask_voltage(x, mask_ratio, mask_gen)
                recon = model.reconstruct(x_c)  # (B, T)
                recon_loss_val = masked_reconstruction_loss(recon, x, mask)
                loss = loss + recon_lambda * recon_loss_val / accum_steps

            loss.backward()
            if step % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            total += float(loss.item()) * x.size(0) * accum_steps
            n += x.size(0)
            total_recon += float(recon_loss_val) * x.size(0)
            total_future += float(future_loss_val) * x.size(0)

        # epoch 末尾：如果剩余累积梯度不足 accum_steps，做最后一次更新。
        if len(loader) % accum_steps != 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        final_loss = total / n
        avg_recon = total_recon / n if recon_lambda > 0 else float("nan")
        avg_future = total_future / n
        print(
            f"  [pretrain] epoch {epoch:3d}/{epochs}  "
            f"loss={final_loss:.6f}  future={avg_future:.6f}  recon={avg_recon:.6f}"
        )
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
    consist_lambda: float = 0.0,
    group_size: int = 4,
    accum_steps: int = 1,
) -> float:
    """在 SOH 回归任务上微调，返回最后一个 epoch 的平均损失。

    支持断点续训，逻辑与 pretrain_voltage 相同。

    创新 1（同循环一致性）：当 consist_lambda > 0 时，批次由
    SameCycleBatchSampler 提供（每组是同一循环的 K 个片段），
    在数据损失之外额外惩罚组内预测的离散程度。
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    final_loss = float("nan")
    for epoch in range(start_epoch, epochs + 1):
        total, n = 0.0, 0
        total_consist = 0.0
        optimizer.zero_grad()
        for step, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)  # y 形状 (B,)

            consist_val = 0.0
            pred = model.soh_predict(x)  # (B,)
            loss = loss_fn(pred, y) / accum_steps
            if consist_lambda > 0:
                # 批次由 SameCycleBatchSampler 构成：每 G 组、每组是
                # 同一循环的 K 个片段。把预测 reshape 成 (G, K)，
                # 惩罚每组内部的离散程度（组内方差）。
                g = pred.numel() // group_size
                pred_g = pred[: g * group_size].view(g, group_size)
                consist_val = same_cycle_consistency_loss(pred_g)
                loss = loss + consist_lambda * consist_val / accum_steps

            loss.backward()
            if step % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            total += float(loss.item()) * x.size(0) * accum_steps
            n += x.size(0)
            total_consist += float(consist_val) * x.size(0)

        if len(loader) % accum_steps != 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        final_loss = total / n
        avg_consist = total_consist / n if consist_lambda > 0 else float("nan")
        print(
            f"  [finetune] epoch {epoch:3d}/{epochs}  "
            f"loss={final_loss:.6f}  consist={avg_consist:.6f}"
        )
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
    # line_buffering=True：每行立即输出，便于后台监控训练进度。
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "processed" / "partial_segments_index.parquet")
    parser.add_argument("--mat-dir", type=Path, default=ROOT / "data" / "external" / "matr")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="若提供，则从 build_cache.py 生成的 memmap 缓存直接读数据（训练提速）",
    )
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
    # ---- 创新特性开关（本分支新增）----
    parser.add_argument(
        "--consistency",
        action="store_true",
        help="启用同循环一致性约束：同一循环的多个片段输出应一致",
    )
    parser.add_argument(
        "--consist-lambda",
        type=float,
        default=1.0,
        help="一致性损失权重；设为 0 等于“只改采样方式、不加约束”的对照",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="每个循环抽取 K 个片段组成一组（K>=2 才有意义）",
    )
    parser.add_argument(
        "--batch-groups",
        type=int,
        default=512,
        help="每个批次包含多少个循环；有效 batch = batch-groups × group-size",
    )
    parser.add_argument(
        "--recon-loss",
        action="store_true",
        help="启用扩展自监督：掩码电压重建",
    )
    parser.add_argument(
        "--recon-lambda",
        type=float,
        default=1.0,
        help="掩码重建损失权重",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.3,
        help="重建任务中遮掉电压点的比例（0~1）",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="梯度累积步数；batch 4096 × 5 可模拟论文的 batch 20,000",
    )
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
    print(f"梯度累积: {args.accum_steps} 步（等效 batch = {config.batch_size * args.accum_steps}）")
    print(f"创新开关: consistency={args.consistency}, recon_loss={args.recon_loss}")
    if args.consistency:
        effective_batch = args.batch_groups * args.group_size
        print(
            f"  一致性: batch_groups={args.batch_groups} × group_size={args.group_size} "
            f"= 有效 batch {effective_batch}, λ={args.consist_lambda}"
        )
    if args.recon_loss:
        print(f"  重建: mask_ratio={args.mask_ratio}, λ={args.recon_lambda}")

    # 两个任务各建一个 Dataset，都用 train 划分。直接训练模式不需要预训练数据集。
    if args.cache_dir is not None:
        print(f"数据源: memmap 缓存 {args.cache_dir}（跳过 MAT 读取与插值）")
        soh_ds = MemmapSohDataset(args.cache_dir, split="train", task="soh")
        if not args.no_pretrain:
            pretrain_ds = MemmapSohDataset(args.cache_dir, split="train", task="pretrain")
    else:
        print(f"数据源: 惰性读取 {args.mat_dir}")
        soh_ds = PartialSohDataset(
            args.index, args.mat_dir, split="train", task="soh", preload=args.preload
        )
        if not args.no_pretrain:
            pretrain_ds = PartialSohDataset(
                args.index,
                args.mat_dir,
                split="train",
                task="pretrain",
                preload=args.preload,
            )

    if args.max_samples is not None:
        n = min(args.max_samples, len(soh_ds))
        soh_ds = Subset(soh_ds, range(n))
        if not args.no_pretrain:
            pretrain_ds = Subset(pretrain_ds, range(n))
        print(f"冒烟测试：只取前 {n} 个样本")

    # 一致性模式需要“每个片段属于哪个循环”的映射来构造分组批次。
    if args.consistency:
        group_ids = _get_group_ids(soh_ds)
        print(f"一致性采样器：{len(group_ids)} 个片段，按循环分组")

    # memmap 缓存模式：用安全 collate 让 __getitems__ 返回的整块张量
    # 直接成为 batch（也兼容 Subset 冒烟测试的逐样本回退）。
    collate_fn = _memmap_safe_collate if args.cache_dir is not None else None

    if not args.no_pretrain:
        pretrain_loader = DataLoader(
            pretrain_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
    if args.consistency:
        # 用“循环分组”的批次采样器替代普通 shuffle：
        # 每个批次 = batch_groups 个循环 × group_size 个同循环片段。
        # steps_per_epoch 与普通模式的更新次数对齐（按样本数折算），
        # 保证消融对比公平。
        steps_per_epoch = math.ceil(
            len(soh_ds) / (args.batch_groups * args.group_size)
        )
        batch_sampler = SameCycleBatchSampler(
            group_ids,
            group_size=args.group_size,
            batch_groups=args.batch_groups,
            seed=config.seed + 1,
            steps_per_epoch=steps_per_epoch,
        )
        soh_loader = DataLoader(
            soh_ds, batch_sampler=batch_sampler, num_workers=0, collate_fn=collate_fn
        )
    else:
        soh_loader = DataLoader(
            soh_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
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
                recon_lambda=args.recon_lambda if args.recon_loss else 0.0,
                mask_ratio=args.mask_ratio,
                seed=config.seed + 2,
                accum_steps=args.accum_steps,
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
            consist_lambda=args.consist_lambda if args.consistency else 0.0,
            group_size=args.group_size,
            accum_steps=args.accum_steps,
        )
        print(f"微调耗时: {time.perf_counter() - t0:.1f}s")

    print("\n== 训练集评估 ==")
    train_metrics = evaluate_soh(model, soh_loader, device)
    print(f"  MAE  = {train_metrics['mae_pct']:.4f}%")
    print(f"  RMSE = {train_metrics['rmse_pct']:.4f}%")

    print("\n== 测试集评估 ==")
    if args.cache_dir is not None:
        test_ds = MemmapSohDataset(args.cache_dir, split="test", task="soh")
    else:
        test_ds = PartialSohDataset(
            args.index, args.mat_dir, split="test", task="soh", preload=True
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_memmap_safe_collate if args.cache_dir is not None else None,
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
