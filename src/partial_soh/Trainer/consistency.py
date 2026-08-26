"""同循环一致性约束：本项目的核心创新模块之一。

背景
----
同一个循环（cell_id, cycle_index）最多可以切出 51 个部分充电片段
（起点从 0% 到 50% 额定容量、步长 1%）。这些片段的起点不同、电压轨迹
不同，但它们的 SOH 标签完全相同——因为 SOH 是“按循环”定义的，
而不是“按片段”定义的。

物理直觉
--------
老化是一个慢过程：同一个循环内部的几十个片段之间，电池容量几乎没有
变化。因此，一个合理的 SOH 估计器必须满足：

    同一循环的所有片段 -> 输出相同的 SOH。

这就是“同循环一致性约束”。论文原版只给每个片段一个 SOH 标签（稀疏
监督），而一致性约束额外要求“同循环的多个输出互相靠近”，把每个循环的
监督变得更密集，也迫使模型学习与“片段起点无关”的老化特征。

实现
----
1. SameCycleBatchSampler：每次采样 G 个循环，每个循环随机抽 K 个片段，
   组成一个 (G*K, 101, 3) 的批次；
2. same_cycle_consistency_loss：把该批次的预测 reshape 成 (G, K)，
   惩罚每组 K 个预测之间的离散程度（组内方差）。

设计说明
--------
一致性损失只负责“让同循环的输出彼此靠近”，不负责把输出拉到正确位置；
SOH 的绝对大小仍由常规的数据损失（MSE 到标签）决定。这样两个损失各司
其职，不会出现“所有输出塌缩到一个错误常数”的问题。
"""

from __future__ import annotations

from math import ceil

import numpy as np
import torch
from torch.utils.data import Sampler


class SameCycleBatchSampler(Sampler[list[int]]):
    """按循环分组的批次采样器。

    每次迭代返回一个长度为 G*K 的下标列表：

        G = batch_groups：本批包含多少个循环；
        K = group_size ：每个循环抽取多少个片段。

    参数
    ----
    group_ids   : 一维 int64 数组，长度 = 样本数；
                  第 i 个样本属于哪个循环（用 0, 1, 2, ... 编号）。
    group_size  : 每个循环抽 K 个片段（K >= 2 时一致性损失才有意义）。
    batch_groups: 每个批次包含多少个循环。
                  有效 batch = batch_groups × group_size。
    seed        : 随机种子，保证可复现。

    注意：epoch 的定义和普通训练不同。普通训练一个 epoch 覆盖所有片段；
    这里一个 epoch 覆盖所有“片段数 >= K 的循环”，每个循环只贡献
    K 个片段（随机抽）。因此单个 epoch 的更新次数更少，但每个循环都会
    被一致性地观测到。
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        group_size: int = 4,
        batch_groups: int = 512,
        seed: int = 0,
    ) -> None:
        if group_size < 2:
            raise ValueError("group_size 至少为 2，否则无法计算组内一致性")
        if batch_groups < 1:
            raise ValueError("batch_groups 至少为 1")

        self.group_size = int(group_size)
        self.batch_groups = int(batch_groups)
        self.rng = np.random.default_rng(seed)

        # 第一步：统计“每个循环有哪些片段”（片段在数据集中的下标）。
        # 用普通 dict 而不是 pandas，省掉建表开销。
        cycles: dict[int, list[int]] = {}
        for idx, gid in enumerate(group_ids):
            cycles.setdefault(int(gid), []).append(idx)

        # 第二步：只保留片段数 >= K 的循环，否则抽不满 K 个。
        # （部分循环可能因为坏周期被过滤后剩下的有效片段很少。）
        self.cycles = {
            gid: idxs for gid, idxs in cycles.items() if len(idxs) >= self.group_size
        }
        if not self.cycles:
            raise ValueError(
                f"没有任何循环包含 >= {group_size} 个有效片段，"
                "请调小 group_size 或检查片段索引。"
            )

        self.cycle_keys = list(self.cycles.keys())
        # 一个 epoch 的迭代次数 ≈ 循环数 / 每批循环数。
        self.n_batches = max(1, ceil(len(self.cycle_keys) / self.batch_groups))

    def __iter__(self):
        """每个批次：有放回地抽 G 个循环，每个循环无放回地抽 K 个片段。"""
        for _ in range(self.n_batches):
            # 有放回抽样：即使循环数少于 batch_groups 也能凑满一个批次。
            chosen = self.rng.choice(
                self.cycle_keys, size=self.batch_groups, replace=True
            )
            batch: list[int] = []
            for gid in chosen:
                segs = self.cycles[gid]
                # 从该循环的片段里无放回抽 K 个（同一批内不重复）。
                picked = self.rng.choice(segs, size=self.group_size, replace=False)
                batch.extend(int(i) for i in picked)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


def same_cycle_consistency_loss(pred_grouped: torch.Tensor) -> torch.Tensor:
    """同循环一致性损失。

    参数
    ----
    pred_grouped : (G, K) 的张量，第 g 行的 K 个值是同一循环的 K 个预测。

    返回
    ----
    标量损失 = 所有组内方差的平均。

    组内方差定义：先算本组的均值 mean_g，再算每个预测与 mean_g 的
    平方偏差的平均。方差为 0 表示同一循环的 K 个片段输出完全相同。
    """
    mean = pred_grouped.mean(dim=1, keepdim=True)  # (G, 1) 每组的均值
    return ((pred_grouped - mean) ** 2).mean()  # 组内方差，再对所有组取平均


if __name__ == "__main__":
    """冒烟测试：演示采样器输出与损失形状。"""
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    # 造一个 3 个循环、共 10 个片段的假数据。
    fake_group_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=np.int64)
    sampler = SameCycleBatchSampler(
        fake_group_ids, group_size=2, batch_groups=2, seed=0
    )
    print(f"n_batches = {len(sampler)}")
    batch = next(iter(sampler))
    print(f"第一个批次下标 = {batch}（长度 {len(batch)}）")

    # 假预测：两组、每组 2 个预测。
    preds = torch.tensor([[0.95, 0.97], [0.91, 0.90]])
    loss = same_cycle_consistency_loss(preds)
    print(f"一致性损失 = {loss.item():.6f}")
    print("预期：第一组方差小、第二组方差更小，损失为两者平均。")
