# partial_soh：片段级在线 SOH 估计复现

本目录用于复现并逐步改进论文：

- **论文**：Transfer learning-based SOH estimation from partial charging data with polarization-aware modeling
- **期刊/DOI**：Scientific Reports, 2026, 16: 23076；10.1038/s41598-026-48906-4
- **中文译稿**：`docs/papers/s41598-026-48906-4_zh.md`

## 当前目标

把论文中的“部分充电片段 -> 当前循环 SOH”在线估计流程在我们的 Severson LFP 数据上复现出来，然后再逐步加入物理约束、温度条件、多片段历史和鲁棒性验证。

与 `src/world_model/` 的区别：

- `world_model` 是“30 个完整历史循环 -> 未来 80 个循环 SOH”的实验室级循环模型；
- `partial_soh` 是“单个充电片段 -> 当前 SOH”的在线估计模型，更贴近比赛要求。

## 已完成的准备

1. 原文 PDF 已保存到 `docs/papers/s41598-026-48906-4.pdf`。
2. 中文译稿已保存到 `docs/papers/s41598-026-48906-4_zh.md`。

## 下一步计划

1. 确认论文的数据切分口径：20% 容量观测窗口、起点 0%–50%、1% 间隔、每 1% 容量插值为 5 步。
2. 设计并实现 `DataLoader/` 中的片段级数据构建脚本。
3. 设计 `Trainer/` 中的电压预测预训练和 SOH 微调流程。
4. 先做小规模冒烟测试，再全量训练。

