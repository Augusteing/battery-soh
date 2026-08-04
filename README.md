# battery-soh

**挑战者杯「揭榜挂帅」CP-202612：复杂动态工况下混合动力车辆磷酸铁锂电池 SOC/SOH 高精度估计研究**

本仓库面向比赛交付物（技术报告 + 答辩）组织实验代码、数据流水线与结果，遵循可复现的计算机科学实验项目结构。

## 项目背景

磷酸铁锂（LFP）电池的开路电压曲线在很宽的 SOC 范围内近乎"平坦"，电压随电量变化极小；混合动力车辆又存在纯电、增程、并联驱动等多种工况的高频切换。两者叠加，使得传统安时积分或简单查表的状态估计算法极易失效。本课题的目标是开发一套兼顾新电池与老化电池、在复杂动态工况下仍能高精度估计 SOC/SOH 的算法方案。

## 当前进展

### A 基线（SOH 估计，已完成）

- 数据：MATR 快速充电数据集（Severson et al., *Nature Energy* 2019，A123 LFP/graphite），batch 20170512 共 46 只电池、23 种快充协议；
- 数据处理：逐循环 SOH 标签表（化成循环剔除、前 10 循环中位数归一化）；
- 特征工程：15 个逐循环特征（时序位置 / 当前观测 / 派生量 / 历史滚动窗口四类，只用过去信息）；
- 模型：Ridge（线性对照）与 HistGradientBoostingRegressor（主力）；
- 验证：按电池分组 5 折 / 按协议分组 5 折 / 时间切分三种方案；
- 结果：跨电池 MAE 0.52%、跨协议 MAE 0.59%（R²≈0.90）；时间外推 MAE 5.21%（暴露逐点回归无法外推的短板，作为 B 方案对照）；
- 演示 notebook：`notebooks/01_matr_a_baseline.ipynb`（数据集协议 → 数据处理 → 特征 → 模型 → 验证 → 结果，已预执行）。

### B 方案（SOH 预测，进行中）

- 任务设定（已与导师确认）：**在线滚动前瞻**——用第 t 个循环及之前的数据，预测第 t+Δ 个循环的 SOH（Δ=50/100/200），并配套早期寿命（RUL）预测；
- 路线：移位标签回归基线（误差 vs 预测距离曲线）→ 退化趋势建模外推（幂律/拐点）→ 序列模型（LSTM/GRU/Transformer）→ 不确定性量化（GPR/分位数回归）；
- 数据扩展：MATR 其余批次、Stanford 动态循环数据集（NCA，动态工况验证）。

## 目录结构

```
battery-soh/
├── data/                  # 数据目录（内容不入库，仅保留结构）
│   ├── raw/               # 原始数据（实车/台架采集，只读）
│   ├── interim/           # 清洗、对齐等中间产物
│   ├── processed/         # 可直接用于训练/标定的最终数据
│   └── external/          # 外部公开数据集（MATR、Stanford 等）
├── docs/                  # 项目文档（比赛方案、调研笔记、论文）
├── models/                # 训练好的模型权重（不入库）
├── notebooks/             # 演示与分析 notebook
├── references/            # 参考文献与资料
├── reports/               # 技术报告（比赛交付物）及图表素材
├── results/               # 实验输出（不入库）
│   ├── figures/           # 图表
│   └── metrics/           # 指标记录（JSON/CSV）
├── scripts/               # 可执行脚本（数据流水线、训练、评估入口）
├── src/battery_soh/       # 核心代码包（部分为空壳，待 B 方案重构）
└── tests/                 # 单元测试
```

## 快速开始

推荐使用 conda 环境 `battery-soh`（Python 3.11）：

```powershell
conda activate battery-soh
```

依赖：pandas、numpy、scikit-learn、pyarrow、h5py、matplotlib、nbformat、nbclient、ipykernel。

## 数据流水线

### 数据获取

MATR 批次文件体积较大（约 3 GB/批），从官网直链手动下载后放入：

```
data/external/matr/MATR_batch_20170512.mat
```

> 数据、模型权重等大文件不入库（见 `.gitignore`）。

### 处理与建模（A 基线）

```powershell
# 1. 逐循环 SOH 标签表 -> data/processed/matr_soh_table.parquet
python scripts/build_matr_soh_table.py

# 2. 特征工程 -> data/processed/matr_features.parquet
python scripts/build_matr_features.py

# 3. 训练与评估 -> results/metrics + results/figures
python scripts/train_baseline.py

# 4. SOH 轨迹总览图
python scripts/plot_matr_soh_overview.py

# 5. 重新生成演示 notebook（内容修改后）
python scripts/make_baseline_notebook.py --out notebooks/01_matr_a_baseline.ipynb
```

## 技术路线

1. **A 基线（完成）**：逐点回归估计当前 SOH，建立指标管道与对照基准；
2. **B 方案（进行中）**：在线滚动前瞻预测未来 SOH——先移位标签基线，再退化趋势外推，再序列模型，最后不确定性量化；
3. **数据扩展**：MATR 其余批次（扩大 LFP 样本量）、Stanford 动态循环数据集（NCA，动态工况鲁棒性验证，报告中注明化学体系差异）；
4. 最终以技术报告 + 答辩 PPT 交付。

## 关键节点

- 2026-09-15：作品（技术报告）提交截止
- 2026-09-30：初审，确定终审擂台赛入围名单
- 2026-11：终审擂台赛

## 数据集与引用

- MATR 快速充电数据集（LFP，124 只）：Severson, K.A. et al. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy* (2019). <https://data.matr.io/>
- Stanford 动态循环数据集（NCA，92 只）：Geslin, A., Xu, L., Ganapathi, D. et al. Dynamic cycling enhances battery lifetime. *Nat Energy* (2024). <https://purl.stanford.edu/td676xr4322>
- 免诊断车载健康评估（B 方案对标方法）：Che, Y. et al. Diagnostic-free onboard battery health assessment. *Joule* 9, 102010 (2025).

使用上述数据集发表成果时请按各自要求引用原文。

## 许可

[MIT](LICENSE)
