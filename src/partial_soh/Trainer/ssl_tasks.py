"""扩展自监督训练：掩码电压重建（denoising reconstruction）。

论文原版的预训练只有“下一步电压预测”一个任务。我们在此基础上加入
第二个自监督任务，组成多任务预训练：

    任务 1（原版）：预测下一步电压。
        每个时间步都有监督目标，属于“密集监督”；

    任务 2（新增）：随机遮掉 30% 的电压点，用上下文重建它们。

任务 2 的动机
-------------
- 遮掉电压、保留电流与容量坐标，迫使模型把“该时刻电压大概是多少”
  编码进隐藏状态，而不是直接照抄输入；
- 重建被遮住的电压需要理解电压曲线在容量轴上的平滑结构（LFP 的
  电压平台期尤其明显），相当于让模型先学会“电化学曲线长什么样”；
- 与 xPatch 等 2025 年工作的“重建 + 去噪 + 预测”多任务预训练思路一致。

实现要点
--------
- 只遮电压通道 V（第 1 列），不遮电流 I 和容量坐标 Q：
  I 和 Q 是“已知的操作条件 / 坐标”，只有 V 是被测信号；
- 被遮住的电压用“该样本的电压均值”填充，而不是填 0。
  填 0 会在曲线里制造一个明显的异常跳变，模型容易靠“看到 0 就知道
  这里被遮了”来偷懒；填均值更像真实传感器噪声/缺失的形态。
- 重建损失只统计掩码位置，避免模型把没遮的位置也强行重建（那部分
  由任务 1 负责）。
"""

from __future__ import annotations

import torch


def mask_voltage(
    x: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机遮掉一部分电压点，返回 (损坏输入, 掩码)。

    参数
    ----
    x         : (B, T, 3) 原始输入，第 1 列是电压 V；
    mask_ratio: 遮掉电压点的比例（0~1）；
    generator : torch.Generator，固定种子保证可复现。

    返回
    ----
    x_corrupted : (B, T, 3)，掩码位置上的 V 被替换为样本电压均值；
    mask        : (B, T) bool，True 表示该位置被遮住。
    """
    if mask_ratio <= 0.0:
        zero_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        return x, zero_mask

    b, t, _ = x.shape
    # 均匀随机掩码：每个点独立地以 mask_ratio 概率被遮住。
    # 先在 CPU 上用 CPU generator 生成，再搬到目标设备（cuda/cpu），
    # 避免“cuda 张量 + cpu generator”的类型不匹配。
    mask = (torch.rand(b, t, generator=generator) < mask_ratio).to(x.device)

    x_corrupted = x.clone()
    # 每个样本的电压均值（沿时间轴取平均），形状 (B, 1)。
    v_mean = x[:, :, 1].mean(dim=1, keepdim=True)
    # torch.where(条件, 填充值, 原值)：mask 为 True 的位置换成均值。
    x_corrupted[:, :, 1] = torch.where(mask, v_mean, x[:, :, 1])
    return x_corrupted, mask


def masked_reconstruction_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """掩码重建损失：只统计被遮住的电压点。

    参数
    ----
    recon : (B, T) 重建头输出（每个时间步一个电压值，与输入对齐）；
    x     : (B, T, 3) 原始（未损坏）输入，第 1 列是电压 V；
    mask  : (B, T) bool，True 的位置参与损失。

    返回
    ----
    标量损失 = 掩码位置上 (重建值 - 真实值)^2 的平均。
    """
    # recon[mask] 取出所有被遮位置的预测，x[:, :, 1][mask] 取出对应真实值。
    return torch.mean((recon[mask] - x[:, :, 1][mask]) ** 2)


if __name__ == "__main__":
    """冒烟测试：演示掩码、重建损失的形状。"""
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4, 101, 3)  # batch=4, seq=101, channels=3

    x_c, mask = mask_voltage(x, mask_ratio=0.3, generator=gen)
    print(f"损坏输入 x_c.shape = {tuple(x_c.shape)}")
    print(f"掩码 mask.shape    = {tuple(mask.shape)}，遮住比例 = {mask.float().mean():.2f}")
    print(f"重建头输出形状（假设）= {tuple(x_c.shape[:2])}")

    fake_recon = torch.randn(4, 101)
    loss = masked_reconstruction_loss(fake_recon, x, mask)
    print(f"掩码重建损失 = {loss.item():.6f}")
