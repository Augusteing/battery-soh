# battery-soh

**挑战者杯「揭榜挂帅」CP-202612：复杂动态工况下混合动力车辆磷酸铁锂电池 SOC/SOH 高精度估计研究**

本仓库面向比赛交付物（技术报告 + 答辩）组织全部实验代码、数据流水线与结果，遵循可复现的计算机科学实验项目结构（参考 Cookiecutter Data Science 惯例）。

## 项目背景

磷酸铁锂（LFP）电池的开路电压曲线在很宽的 SOC 范围内近乎"平坦"，电压随电量变化极小；混合动力车辆又存在纯电、增程、并联驱动等多种工况的高频切换。两者叠加，使得传统安时积分或简单查表的状态估计算法极易失效。本课题的目标是开发一套兼顾新电池与老化电池、在复杂动态工况下仍能高精度估计 SOC/SOH 的算法方案。

## 当前进展

- 搭建标准可复现实验目录结构（数据 / 代码 / 配置 / 结果分离），配置 GitHub Actions CI；
- 接入 **Stanford 动态循环老化数据集**（Geslin et al., *Nature Energy* 2024，92 只电池）：
  - 支持从 Stanford SDR 在线检索文件清单并断点续传下载；
  - 解析逐循环老化汇总表，自动识别容量/循环序号列，计算每只电池的 SOH 曲线；
  - 一键汇总全部电池为统一的 SOH 标签表（Parquet），供后续建模使用；
- 预留 MATR（Severson et al. 2019）与 NASA Randomized Battery Usage 数据集下载接口；
- 数据加载模块配套 pytest 单元测试（合成数据，不依赖真实下载）。

## 目录结构

```
battery-soh/
├── configs/               # 实验配置文件（YAML），一次实验对应一份配置
├── data/                  # 数据目录（内容不入库，仅保留结构）
│   ├── raw/               # 原始数据（实车/台架采集，只读）
│   ├── interim/           # 清洗、对齐等中间产物
│   ├── processed/         # 可直接用于训练/标定的最终数据
│   └── external/          # 外部公开数据集（Stanford、MATR、NASA 等）
├── docs/                  # 项目文档（比赛方案、调研笔记）
├── models/                # 训练好的模型权重（不入库）
├── notebooks/             # 探索性分析 Jupyter 笔记本
├── references/            # 参考文献与资料
├── reports/               # 技术报告（比赛交付物）及图表素材
├── results/               # 实验输出
│   ├── figures/           # 图表
│   └── metrics/           # 指标记录（JSON/CSV）
├── scripts/               # 可执行脚本（数据流水线、训练、评估入口）
├── src/battery_soh/       # 核心代码包（可编辑安装）
│   ├── data/              # 数据加载与预处理（含数据集下载）
│   ├── features/          # 特征工程
│   ├── models/            # 电池模型（ECM、OCV-SOC 曲线等）
│   ├── estimation/        # SOC/SOH 估计器（EKF/UKF/数据驱动）
│   ├── evaluation/        # 评估指标与验证流程
│   └── visualization/     # 可视化工具
├── tests/                 # pytest 单元测试
└── .github/workflows/     # CI
```

## 快速开始

方式一：conda（推荐）

```powershell
conda env create -f environment.yml
conda activate battery-soh
pip install -e .
pytest -q
```

方式二：venv

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q
```

如需 GPU 深度学习（RTX 3060 + 驱动 546.30 → CUDA 12.1）：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 数据流水线

以 Stanford 动态循环数据集为例：

```powershell
# 查看可用数据集与远程文件清单
python scripts/download_data.py list

# 下载逐循环老化汇总（体积小，SOH 标签来源）
python scripts/download_data.py stanford-summary

# 按需下载指定电池的原始时间序列（电压/电流/温度）
python scripts/download_data.py stanford-raw --cells 3 17 45
python scripts/download_data.py stanford-raw --all          # 约 16 GB

# 汇总生成统一的 SOH 标签表 -> data/processed/stanford_soh_table.parquet
python scripts/build_soh_tables.py
```

其他公开数据集接口：

```powershell
python scripts/download_data.py matr --batches 20170512     # 单 batch 约 3 GB
python scripts/download_data.py nasa-rw                     # 约 1 GB
```

> 数据、模型权重等大文件不入库（见 `.gitignore`），克隆仓库后需按上述脚本重新拉取。

## 实验规范

1. 原始数据放入 `data/raw/` 后不再改动，所有变换通过 `scripts/` 中的流水线脚本完成；
2. 每次实验在 `configs/` 下新建一份配置，运行结果写入 `results/metrics/` 与 `results/figures/`，并在提交信息中注明配置名；
3. 通用逻辑沉淀到 `src/battery_soh/`，notebook 只做探索，不存放最终逻辑；
4. 数据、模型权重等大文件不入库。

## 技术路线

1. **SOH 主线**：基于老化汇总数据建立逐循环 SOH 标签，先从数据驱动方法（容量增量分析 ICA/DVA + 机器学习回归）切入，建立基线；
2. **SOC 主线**：构建等效电路模型（ECM）与 OCV-SOC 曲线，针对 LFP 平台区"电压不敏感"问题引入 EKF/UKF 与数据驱动融合的估计器；
3. **联合估计**：在线辨识随老化衰减的模型参数，实现 SOC/SOH 联合估计，并验证对新电池与老化电池的泛化能力；
4. 使用多种公开数据集交叉验证，最终在复杂动态工况片段上评估鲁棒性。

## 关键节点

- 2026-09-15：作品（技术报告）提交截止
- 2026-09-30：初审，确定终审擂台赛入围名单
- 2026-11：终审擂台赛

## 数据集与引用

- Stanford 动态循环数据集：<https://purl.stanford.edu/td676xr4322>
  Geslin, A., Xu, L., Ganapathi, D. et al. Dynamic cycling enhances battery lifetime. *Nat Energy* (2024). <https://doi.org/10.1038/s41560-024-01675-8>
- MATR 快速充电数据集：Severson, K.A. et al. Data-driven prediction of battery cycle life before capacity degradation. *Nat Energy* (2019). <https://data.matr.io/>
- NASA Randomized Battery Usage 数据集：<https://phm-datasets.s3.amazonaws.com/NASA/11.+Randomized+Battery+Usage+Data+Set.zip>

使用上述数据集发表成果时请按各自要求引用原文。

## 许可

[MIT](LICENSE)
