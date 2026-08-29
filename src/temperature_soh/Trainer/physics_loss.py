"""车辆级微调的物理约束损失（第一梯队三件套）。

三个约束分别对应一条电池物理常识：

1. 单调性（monotonicity）：正常老化下 SOH 不可逆，预测的循环级 SOH
   轨迹不允许回升。容差 eps 容忍标签测量噪声（循环容量本身有起伏）。

        L_mono = mean( relu( s(k+1) - s(k) + eps )^2 )

2. 有界性（boundedness）：SOH 是 0~1 之间的无量纲比例（我们以该电池
   最大充电容量为参考，标签天然 ≤ 1）。软惩罚把预测钉在物理范围内，
   防止出现 ">1 的健康度" 这种无意义输出。

        L_bounds = mean( relu(s - 1)^2 + relu(-s)^2 )

3. 同循环一致性（consistency）：SOH 是"循环级"物理量，同一个循环切出
   的多个片段（最多 51 个，起点不同）必须给出同一个 SOH。约束等于要求
   循环内预测方差尽量小，把片段噪声摊平，让模型学稳定的循环级读数。

        L_cons = mean_k( var( {s_j : j in cycle k} ) )

用法：先对训练集做一次完整前向，拿到每个片段的预测 pred 和对应的
循环号 cycle_ids，然后调用 cycle_physics_losses() 一次得到三个损失。
三个损失都是标量张量、保持计算图，可直接 backward。

与 world_model/loss.py 的关系：那里是"未来 80 循环 rollout 轨迹"的
单调性（世界模型用）；这里是"同一辆车训练段逐循环"的单调性（车辆微调用）。
公式同源，容差 eps 沿用 0.005。
"""

from __future__ import annotations

import torch

# 单调性容差：容忍循环容量标签的测量起伏（与 world_model 的 EPS_MONO 一致）。
EPS_MONO = 0.005


def cycle_physics_losses(
    pred: torch.Tensor,
    cycle_ids: torch.Tensor,
    eps: float = EPS_MONO,
) -> dict[str, torch.Tensor]:
    """由片段级预测聚合出三个物理损失。

    参数
    ----
    pred      : (N,) 片段级 SOH 预测（保持计算图）。
    cycle_ids : (N,) 每个片段所属的循环号（int64，任意顺序）。
    eps       : 单调性容差。

    返回
    ----
    dict，键 = "mono" / "bounds" / "consistency"，值 = 标量损失张量。
    其中 mono/bounds 基于"循环均值轨迹"，consistency 基于循环内方差。
    """
    if pred.shape != cycle_ids.shape:
        raise ValueError(
            f"pred 与 cycle_ids 形状不一致: {tuple(pred.shape)} vs "
            f"{tuple(cycle_ids.shape)}"
        )
    if pred.numel() == 0:
        raise ValueError("没有片段，无法计算物理损失")

    device = pred.device
    cycle_ids = cycle_ids.to(device)

    # torch.unique 返回升序排列的循环号 -> cycle_means 天然按时间排序。
    unique_cycles = torch.unique(cycle_ids)
    cycle_means: list[torch.Tensor] = []
    cycle_vars: list[torch.Tensor] = []
    for c in unique_cycles:
        p = pred[cycle_ids == c]
        m = p.mean()
        cycle_means.append(m)
        # 单片段循环方差为 0，不影响 mean()。
        cycle_vars.append(((p - m) ** 2).mean())

    traj = torch.stack(cycle_means)   # (K,) 按循环号升序的 SOH 轨迹
    vars_ = torch.stack(cycle_vars)   # (K,) 每循环内片段预测方差

    # 1) 单调性：相邻循环不允许回升（超过容差 eps 才惩罚）。
    diff = traj[1:] - traj[:-1]
    mono = (torch.relu(diff + eps) ** 2).mean()

    # 2) 有界性：循环均值不允许超出 [0, 1]（软 hinge，不硬裁剪）。
    bounds = (
        (torch.relu(traj - 1.0) ** 2).mean()
        + (torch.relu(-traj) ** 2).mean()
    )

    # 3) 同循环一致性：循环内预测方差越小越好。
    consistency = vars_.mean()

    return {"mono": mono, "bounds": bounds, "consistency": consistency}
