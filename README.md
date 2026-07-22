# battery-soh

挑战者杯「揭榜挂帅」CP-202612：复杂动态工况下混合动力车辆磷酸铁锂电池 SOC/SOH 高精度估计研究。

本仓库面向比赛交付物（技术报告 + 答辩）组织全部实验代码、数据流水线与结果，遵循可复现的计算机科学实验项目结构（参考 Cookiecutter Data Science 惯例）。

## 目录结构

```
battery-soh/
├── configs/               # 实验配置文件（YAML），一次实验对应一份配置
├── data/                  # 数据目录（内容不入库，仅保留结构）
│   ├── raw/               # 原始数据（实车/台架采集，只读）
│   ├── interim/           # 清洗、对齐等中间产物
│   ├── processed/         # 可直接用于训练/标定的最终数据
│   └── external/          # 外部公开数据集（NASA、CALCE、OXford 等）
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
│   ├── data/              # 数据加载与预处理
│   ├── features/          # 特征工程
│   ├── models/            # 电池模型（ECM、OCV-SOC 曲线等）
│   ├── estimation/        # SOC/SOH 估计器（EKF/UKF/数据驱动）
│   ├── evaluation/        # 评估指标与验证流程
│   └── visualization/     # 可视化工具
├── tests/                 # pytest 单元测试
└── .github/workflows/     # CI
```

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q
```

## 实验规范

1. 原始数据放入 `data/raw/` 后不再改动，所有变换通过 `scripts/` 中的流水线脚本完成；
2. 每次实验在 `configs/` 下新建一份配置，运行结果写入 `results/metrics/` 与 `results/figures/`，并在提交信息中注明配置名；
3. 通用逻辑沉淀到 `src/battery_soh/`，notebook 只做探索，不存放最终逻辑；
4. 数据、模型权重等大文件不入库（见 `.gitignore`）。

## 关键节点

- 2026-09-15：作品（技术报告）提交截止
- 2026-09-30：初审，确定终审擂台赛入围名单
- 2026-11：终审擂台赛

## 许可

[MIT](LICENSE)
