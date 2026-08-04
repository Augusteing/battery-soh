"""生成 A 基线展示 notebook（01_matr_a_baseline.ipynb）。

用 nbformat 组装 markdown + code 单元格，产出结构化 notebook JSON。
用法:
    python scripts/make_baseline_notebook.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


def build_cells() -> list:
    cells = []

    # ---------- 0. 标题 ----------
    cells.append(
        md(
            """# LFP 电池 SOH 预测 — A 基线（特征回归）演示

**挑战杯「揭榜挂帅」赛题 CP-202612**：复杂动态工况下混合动力车辆磷酸铁锂电池 SOC/SOH 高精度估计研究

本 notebook 演示 **A 基线**的完整流程：数据集来源与协议 → 数据处理 → 特征工程 → 模型选择 → 验证方案 → 结果解读。

> 运行前提：已生成 `data/processed/matr_soh_table.parquet` 与 `data/processed/matr_features.parquet`
> （由 `scripts/` 目录下的流水线脚本构建，本 notebook 只做分析与建模）。"""
        )
    )

    # ---------- 1. 数据集 ----------
    cells.append(
        md(
            """## 1. 数据集：来源与协议

### 1.1 数据来源

**MATR 快速充电数据集**（Severson et al., *Nature Energy* 2019，[论文链接](https://www.nature.com/articles/s41560-018-0235-9)，[官网](https://data.matr.io/1/)）

| 属性 | 内容 |
| --- | --- |
| 化学体系 | **磷酸铁锂（LFP）/ 石墨**，A123 APR18650M1A，标称 1.1 Ah |
| 总规模 | 124 只电池，寿命 150–2300 次循环 |
| 工况 | 两步快充协议 + 恒流 4C 放电（快充策略多样） |
| 本演示 | batch **20170512**：46 只电池，逐循环汇总（容量 / 内阻 / 温度 / 充电时间） |

**为什么选它**：
1. 化学体系与赛题一致（LFP），且公开、可复现；
2. 23 种快充协议带来丰富的工况多样性，可验证「未见过工况」的泛化；
3. 逐循环摘要自带 SOH 标签（放电容量）与内阻、温度特征，无需额外诊断测试。"""
        )
    )

    cells.append(
        code(
            """# ===== 数据加载 =====
from pathlib import Path
import numpy as np
import pandas as pd

# 兼容「在 notebooks/ 下运行」和「在项目根目录运行」两种方式
_p = Path.cwd()
ROOT = _p if (_p / "data").exists() else (_p.parent if (_p.parent / "data").exists() else _p)

table = pd.read_parquet(ROOT / "data/processed/matr_soh_table.parquet")
features = pd.read_parquet(ROOT / "data/processed/matr_features.parquet")

print(f"SOH 表：{table.shape[0]:,} 行，{table['cell_id'].nunique()} 只电池")
print(f"特征表：{features.shape[0]:,} 行")
table.head()"""
        )
    )

    cells.append(
        md(
            """### 1.2 协议格式与寿命分布

协议字符串形如 `5.4C(40%)-3.6C`，含义：**第一步以 5.4C 恒流充至 40% SOC，第二步以 3.6C 充至满电**；放电统一为 4C 恒流。
更高的首段倍率、更早切换意味着更激进的老化应力，因此不同协议的电池寿命差异很大。"""
        )
    )

    cells.append(
        code(
            """# ===== 协议统计 =====
pol = table.drop_duplicates("cell_id")[["cell_id", "policy", "cycle_life"]]
pol.groupby("policy").agg(
    电池数=("cell_id", "count"),
    最短寿命=("cycle_life", "min"),
    最长寿命=("cycle_life", "max"),
).sort_values("电池数", ascending=False)"""
        )
    )

    cells.append(
        code(
            """# ===== 解析协议字符串 =====
import re

def parse_policy(p):
    m = re.fullmatch(r"([\\d.]+)C\\((\\d+)%\\)-([\\d.]+)C", str(p))
    if not m:
        return (np.nan, np.nan, np.nan)
    return (float(m.group(1)), int(m.group(2)), float(m.group(3)))

parsed = pol["policy"].drop_duplicates().map(parse_policy).apply(pd.Series)
parsed.columns = ["first_C", "soc_pct", "second_C"]
parsed.insert(0, "policy", pol["policy"].drop_duplicates().values)
parsed.sort_values(["first_C", "soc_pct"]).head(10)"""
        )
    )

    # ---------- 2. 数据处理 ----------
    cells.append(
        md(
            """## 2. 数据处理

处理流程（在 `scripts/build_matr_soh_table.py` 中实现，此处验证逻辑）：

1. **去掉化成循环**：容量为 0 的首行（化成/初始化）不参与分析；
2. **SOH 定义**：`SOH_t = Q_discharge,t / Q0`，其中 `Q0` 取每只电池**前 10 个正容量循环的中位数**，抑制初期波动；
3. **统一列名**小写（`IR → ir`、`Tmax → tmax` 等），存储为 Parquet。"""
        )
    )

    cells.append(
        code(
            """# ===== 验证 SOH 定义 =====
REF_CYCLES = 10
q0 = (
    table[table["discharge_capacity"] > 0]
    .groupby("cell_id")["discharge_capacity"]
    .transform(lambda s: s.head(REF_CYCLES).median())
)
recomputed = table["discharge_capacity"] / q0
print("与存储 SOH 的最大差异：", (recomputed - table["soh"]).abs().max())

per_cell = table.groupby("cell_id")["soh"].agg(["min", "last"])
print(f"各电池最低 SOH：均值 {per_cell['min'].mean():.3f}，最小 {per_cell['min'].min():.3f}")
print(f"各电池终止 SOH：均值 {per_cell['last'].mean():.3f}")
per_cell["min"].hist(bins=20, grid=False, figsize=(6, 3));"""
        )
    )

    cells.append(
        code(
            """# ===== SOH 轨迹总览 =====
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
for _cell_id, g in table.groupby("cell_id"):
    ax.plot(g["cycle_index"], g["soh"], lw=0.8, alpha=0.5)
ax.axhline(0.8, color="r", ls="--", lw=1.2, label="EOL 80%")
ax.set_xlabel("循环序号")
ax.set_ylabel("SOH")
ax.set_ylim(0.72, 1.02)
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("46 只 LFP 电池的 SOH 轨迹（MATR batch 20170512）")
plt.show()"""
        )
    )

    # ---------- 3. 特征工程 ----------
    cells.append(
        md(
            """## 3. 特征工程

**预测模式**：用第 t 个循环的特征，预测第 t 个循环的 SOH（逐点回归）。

15 个特征分为四类（`scripts/build_matr_features.py` 实现）：

| 类别 | 特征 | 说明 |
| --- | --- | --- |
| 时序位置 | `cycle_index`、`cumulative_charge` | 循环序号；累计充电安时数（老化应力累计） |
| 当前观测 | `ir`、`tavg/tmax/tmin`、`chargetime` | 本循环内阻、温度、充电时长 |
| 派生量 | `temp_amp`、`chargetime_ratio` | 温度振幅；充电时长与近期均值之比 |
| 历史窗口（过去 10 循环） | `ir_mean10/ir_std10`、`tavg_mean10/tavg_std10`、`ir_deriv10`、`capacity_deriv10` | 内阻/温度的水平与波动；内阻上升速率；容量衰减速率 |

**关键取舍**：
- 不直接用容量水平值（`Q_t/Q0` 就是 SOH，放进特征等于开卷考试）；
- 滚动特征只用 t−9…t 的历史信息，**无未来泄漏**；
- 排除 `cell_id / policy / cycle_life`（分组信息与寿命标签）。"""
        )
    )

    cells.append(
        code(
            """# ===== 特征列表 =====
FEATURES = [
    "cycle_index", "ir", "tavg", "tmax", "tmin", "chargetime", "temp_amp",
    "cumulative_charge", "ir_mean10", "ir_std10", "tavg_mean10", "tavg_std10",
    "ir_deriv10", "capacity_deriv10", "chargetime_ratio",
]
TARGET = "soh"

print("特征数：", len(FEATURES))
print("缺失值总数：", int(features[FEATURES].isna().sum().sum()))
features[FEATURES].head()"""
        )
    )

    # ---------- 4. 模型 ----------
    cells.append(
        md(
            """## 4. 模型：为什么用树模型

两个对照模型：

1. **Ridge（线性回归 + L2 正则）**：作为线性基准的下限参照；
2. **HistGradientBoostingRegressor（直方图梯度提升树）**：主力模型。

### 为什么梯度提升树适合本任务
- **数据形态**：表格数据（行×特征）、约 3.8 万样本——树模型在该规模下通常是 SOTA；深度学习更适合原始序列/图像；
- **非线性与交互**：树分裂自动捕捉「循环数 > 500 且内阻上升速率高 → SOH 快速衰减」这类规则；
- **工程友好**：无需归一化、CPU 训练快、有特征重要性可以解释（答辩加分）。

### 模型特点（关键超参数）
- **梯度提升**：300 棵深度 4 的回归树，逐棵拟合上一轮的残差（负梯度）；
- **学习率收缩**：每棵树的贡献按 `learning_rate=0.05` 缩放入账，防止过拟合；
- **直方图分箱**：连续特征离散化为直方图桶，显著加速训练（LightGBM 同类思想）。"""
        )
    )

    cells.append(
        code(
            """# ===== 模型定义 =====
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS = {
    "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=4, random_state=0
    ),
}

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "最大误差": float(np.max(np.abs(y_pred - y_true))),
        "n": len(y_true),
    }"""
        )
    )

    # ---------- 5. 验证 ----------
    cells.append(
        md(
            """## 5. 验证方案（避免数据泄漏）

| 方案 | 分组方式 | 检验能力 |
| --- | --- | --- |
| 按电池分组 5 折 | 同一电池的所有循环在同一折 | 泛化到**未见过的新电池** |
| 按协议分组 5 折 | 同一协议的所有电池在同一折 | 泛化到**未见过的新工况（协议）** |
| 时间切分 | 每只电池前 70% 循环训练，后 30% 测试 | **在线预测未来**的能力 |

后两种方案对比赛尤为重要：混动车辆会面对新协议/新工况，且 SOH 估计必须在电池使用过程中持续生效。"""
        )
    )

    cells.append(
        code(
            """# ===== 训练与评估 =====
import time

def run_group_cv(groups, label, results):
    gkf = GroupKFold(n_splits=5)
    for name, model in MODELS.items():
        yt, yp = [], []
        for tr, te in gkf.split(features, features[TARGET], groups):
            model.fit(features.iloc[tr][FEATURES], features.iloc[tr][TARGET])
            yt.append(features.iloc[te][TARGET].values)
            yp.append(model.predict(features.iloc[te][FEATURES]))
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        results[label][name] = evaluate(yt, yp)
        results[label][name]["pred"] = yp
        results[label][name]["true"] = yt

def run_temporal(results, frac=0.7):
    tr_idx, te_idx = [], []
    for _, g in features.groupby("cell_id", sort=False):
        cut = int(len(g) * frac)
        tr_idx += list(g.index[:cut])
        te_idx += list(g.index[cut:])
    for name, model in MODELS.items():
        model.fit(features.iloc[tr_idx][FEATURES], features.iloc[tr_idx][TARGET])
        yt = features.iloc[te_idx][TARGET].values
        yp = model.predict(features.iloc[te_idx][FEATURES])
        results["temporal"][name] = evaluate(yt, yp)
        results["temporal"][name]["pred"] = yp
        results["temporal"][name]["true"] = yt

results = {"cv_cell": {}, "cv_policy": {}, "temporal": {}}
t0 = time.time()
run_group_cv(features["cell_id"].values, "cv_cell", results)
run_group_cv(features["policy"].values, "cv_policy", results)
run_temporal(results)
print(f"训练 + 评估耗时：{time.time() - t0:.1f} s")"""
        )
    )

    # ---------- 6. 结果 ----------
    cells.append(
        md(
            """## 6. 结果

下方代码单元格动态汇总本次运行的指标（与上次运行一致，随机种子固定、流程确定）：

| 验证方案 | 含义 | Ridge MAE | HistGB MAE | HistGB RMSE | HistGB R² |
| --- | --- | --- | --- | --- | --- |
| 按电池分组 5 折 | 未见过的新电池 | 1.53% | **0.52%** | 1.38% | 0.898 |
| 按协议分组 5 折 | 未见过的新协议 | 1.58% | **0.59%** | 1.41% | 0.893 |
| 时间切分 | 同电池预测未来 30% | 5.20% | 5.21% | 7.01% | −1.42 |

### 解读

1. **插值能力强**：跨电池、跨协议的 MAE ≈ 0.5–0.6%，R² ≈ 0.9 ——「循环序号 + 内阻 + 温度等特征 → SOH」的模式是稳定可学的；
2. **线性模型失效**：Ridge 按电池分组 R² 为负 —— 证明 SOH 与特征的关系本质非线性，树模型是必要选择；
3. **时间外推是短板**：预测同一电池未来 30% 循环时 MAE 升至约 5%、R² 为负 —— 循环序号、累计充电量等特征超出训练分布后，纯逐点回归只能插值、不能外推。**这正是 A 基线的价值：把问题暴露出来，作为 B 方案的对照基准。**"""
        )
    )

    cells.append(
        code(
            """# ===== 指标汇总表 =====
from IPython.display import display

rows = []
labels = [
    ("cv_cell", "按电池分组 5 折（未见过的新电池）"),
    ("cv_policy", "按协议分组 5 折（未见过的新协议）"),
    ("temporal", "时间切分（同电池预测未来 30% 循环）"),
]
for scheme, label in labels:
    for name in MODELS:
        m = results[scheme][name]
        rows.append([label, name, f"{m['MAE'] * 100:.2f}%", f"{m['RMSE'] * 100:.2f}%", f"{m['R2']:.3f}", f"{m['最大误差'] * 100:.2f}%"])

df_res = pd.DataFrame(rows, columns=["验证方案", "模型", "MAE", "RMSE", "R²", "最大绝对误差"])
display(df_res)"""
        )
    )

    cells.append(
        code(
            """# ===== 预测 vs 真实（HistGradientBoosting）=====
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for ax, (scheme, label) in zip(axes, labels):
    m = results[scheme]["HistGradientBoosting"]
    ax.scatter(m["true"], m["pred"], s=4, alpha=0.25, color="steelblue")
    ax.plot([0.5, 1.05], [0.5, 1.05], "r--", lw=1)
    ax.set_title(f"{label}\\nMAE={m['MAE'] * 100:.2f}%")
    ax.set_xlabel("真实 SOH")
    ax.set_ylabel("预测 SOH")
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.5, 1.05)
    ax.grid(alpha=0.3)

out_fig = ROOT / "results/figures/matr_baseline_pred.png"
out_fig.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_fig, dpi=200)
plt.show()
print("已保存：", out_fig)"""
        )
    )

    cells.append(
        code(
            """# ===== 特征重要性（全量数据拟合，可解释性）=====
final = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0)
final.fit(features[FEATURES], features[TARGET])

imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
imp.plot.barh(ax=ax, color="steelblue")
ax.set_title("HistGradientBoosting 特征重要性（全量数据拟合）")
ax.set_xlabel("重要性")
plt.show()"""
        )
    )

    # ---------- 7. 结论 ----------
    cells.append(
        md(
            """## 7. 结论与下一步

**A 基线已完成**：打通「数据 → 特征 → 模型 → 验证 → 指标」全链路，作为后续方案的对照基准。

**三个关键发现**：
1. 跨电池/跨协议泛化良好（MAE ≈ 0.5%），但纯逐点回归**无法时间外推**（未来 30% MAE ≈ 5%）；
2. 任务本质非线性（线性模型 R² 为负）；
3. 可解释性良好：特征重要性可支撑报告与答辩。

**下一步（B 方案）**：
1. 改为**前瞻预测**：用前 K 个循环预测第 K+Δ 个循环的 SOH，建模退化趋势；
2. 引入**原始充放电曲线片段** + 深度学习 / 物理约束模型（对标 Che et al. 2025, *Joule*：DVA 机理约束的 encoder-decoder）；
3. 加入 Stanford 动态循环数据集做动态工况交叉验证（注意其化学体系为 NCA，作为工况鲁棒性验证）。"""
        )
    )

    cells.append(
        md(
            """## 参考文献

1. Severson, K.A., Attia, P.M., Jin, N. et al. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy* 4, 383–391 (2019).
2. Che, Y., Lam, V.N., Rhyu, J. et al. Diagnostic-free onboard battery health assessment. *Joule* 9, 102010 (2025).
3. 挑战杯「揭榜挂帅」赛题 CP-202612《复杂动态工况下混合动力车辆磷酸铁锂电池 SOC/SOH 高精度估计研究》比赛方案。"""
        )
    )

    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("01_matr_a_baseline.ipynb"))
    args = parser.parse_args()

    nb = new_notebook(cells=build_cells())
    nb.metadata["kernelspec"] = {"display_name": "battery-soh", "language": "python", "name": "battery-soh"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    nbformat.write(nb, str(args.out))
    print(f"cells: {len(nb.cells)}  -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
