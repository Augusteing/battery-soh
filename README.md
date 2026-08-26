# battery-soh

**挑战者杯「揭榜挂帅」CP-202612：复杂动态工况下混合动力车辆磷酸铁锂电池 SOC/SOH 高精度估计研究**

本仓库面向比赛交付物（技术报告 + 答辩）组织实验代码、数据流水线与结果，遵循可复现的计算机科学实验项目结构。

## 项目背景

磷酸铁锂（LFP）电池的开路电压曲线在很宽的 SOC 范围内近乎"平坦"，电压随电量变化极小；混合动力车辆又存在纯电、增程、并联驱动等多种工况的高频切换。两者叠加，使得传统安时积分或简单查表的状态估计算法极易失效。本课题的目标是开发一套兼顾新电池与老化电池、在复杂动态工况下仍能高精度估计 SOC/SOH 的算法方案。

## 当前进展

### 主线：片段级在线 SOH 估计（`src/partial_soh/`，进行中）

复现 Scientific Reports 2026（DOI 10.1038/s41598-026-48906-4）的迁移学习方案：
LSTM 先做“下一步电压预测”自监督预训练，再微调 SOH 回归头；模型输入是
部分充电片段（拿到当前充电片段即可在线估计，不需要完整循环）。

- 数据：Severson et al. (2019) LFP 快速充电数据集（MATR，A123 18650，1.1 Ah）；
  按 Severson 2019 口径排除坏电池后剩 124 只，再排除 1 只短寿命电池，共 123 只；
- 片段：每个循环在 0%–50% 额定容量区间内以 1% 步长滑动 20% 容量窗口，
  重采样到 101 点，输入通道 [I, V, Q]；SOH 标签 = 充电容量 / 1.1；
- 基线结果（10 预训练 + 10 微调 epoch，测试集）：
  - 迁移学习 LSTM：MAE **1.80%**、RMSE 2.35%；
  - 直接训练 LSTM（对照）：MAE **1.87%**、RMSE 2.43%；
  - 论文报告（50+50 epoch）：迁移 0.91% / 1.30%，直接 1.75% / 2.35%。

### 创新：同循环一致性 + 扩展自监督（分支 `codex/feature/consistency-ssl`）

- 创新 1（同循环一致性约束）：同一循环切出的多个片段共享同一个 SOH，
  用分组采样 + 组内方差损失让模型输出彼此一致；
- 创新 2（扩展自监督）：在原“下一步电压预测”之外增加掩码电压重建任务；
- 消融驱动脚本 `run_ablation.py` 已跑通冒烟验证，5 配置全量消融进行中
  （结果将写入 `results/metrics/ablation_consistency_ssl.json`）。

### 世界模型复现（旧主线，独立保留 `src/world_model/`）

arXiv 2603.10527：完整循环 V/I/T -> 1D-CNN 循环编码器 -> PatchTST -> 动力学
滚动预测未来 80 个循环 SOH（W=30, H=80）。与片段级主线数据口径分开，不再混用。

## 目录结构

```
battery-soh/
├── data/                  # 数据目录（内容不入库，仅保留结构）
│   ├── raw/               # 原始数据（实车/台架采集，只读）
│   ├── interim/           # 清洗、对齐等中间产物
│   ├── processed/         # 可直接用于训练/标定的最终数据
│   └── external/          # 外部公开数据集（MATR 等）
├── docs/                  # 项目文档（比赛方案、调研笔记、论文与翻译）
├── models/                # 训练好的模型权重（不入库）
├── notebooks/             # 演示与分析 notebook
├── references/            # 参考文献与资料
├── reports/               # 技术报告（比赛交付物）及图表素材
├── results/               # 实验输出（不入库）
│   ├── figures/           # 图表
│   ├── metrics/           # 指标记录（JSON）
│   └── runs/              # 训练日志
├── scripts/               # 通用脚本（数据探索、绘图）
├── src/
│   ├── partial_soh/       # 当前主线：片段级在线 SOH（DataLoader / Trainer）
│   └── world_model/       # 世界模型复现（旧主线，独立保留）
└── tests/                 # 单元测试
```

## 快速开始

推荐使用 conda 环境 `battery-soh`（Python 3.11）：

```powershell
conda activate battery-soh
```

依赖：pytorch、pandas、numpy、scikit-learn、pyarrow、h5py、matplotlib、nbformat、nbclient、ipykernel。

## 数据流水线

### 数据获取

Severson LFP 数据集（MATR，3 个批次共 140 个通道）从官网直链手动下载后放入：

```
data/external/matr/
```

> 数据、模型权重等大文件不入库（见 `.gitignore`）。

### 处理与训练（主线 partial_soh）

```powershell
# 1. 构建片段索引（labels -> splits -> segments）
#    -> data/processed/partial_segments_index.parquet
python src/partial_soh/DataLoader/build_dataset.py

# 2. 训练（电压预测预训练 + SOH 微调）
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/trainer.py" `
  --preload --pretrain-epochs 50 --finetune-epochs 50

# 3. 消融实验（5 配置：基线 / 只改采样 / 一致性 / 重建 / 完整方案）
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/run_ablation.py" `
  --epochs 10 --preload
```

## 技术路线

1. **片段级在线 SOH 估计（主线）**：部分充电片段 -> 共享 LSTM 编码器
   （“下一步电压预测”自监督预训练）-> SOH 回归头；
2. **创新 1（同循环一致性约束）**：同一循环的多个片段输出彼此一致，
   缓解单片段观测噪声，把稀疏的逐循环监督变成组内稠密约束；
3. **创新 2（扩展自监督）**：掩码电压重建，增强编码器对 LFP 电压平台
   与曲线平滑结构的表征；
4. **数据扩展（待接入）**：补充带温度变化的 LFP 数据集
   （SNL/Preger、UMR AMPERE 等），增加温度输入通道并做跨温度验证；
5. 最终以技术报告 + 答辩 PPT 交付。

## 关键节点

- 2026-09-15：作品（技术报告）提交截止
- 2026-09-30：初审，确定终审擂台赛入围名单
- 2026-11：终审擂台赛

## 数据集与引用

- Severson LFP 快速充电数据集（MATR，124 只可用）：Severson, K.A. et al. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy* (2019). <https://data.matr.io/>
- 片段级 SOH 估计（复现对象）：*Scientific Reports* (2026)，DOI 10.1038/s41598-026-48906-4（原文 PDF 与中文翻译见 `docs/papers/`）
- 世界模型（复现对象，旧主线）：arXiv 2603.10527（完整循环 -> PatchTST -> 未来 SOH 滚动预测）

使用上述数据集发表成果时请按各自要求引用原文。

## 许可

[MIT](LICENSE)
