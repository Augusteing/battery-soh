# DataLoader：循环标准化模块（M2 · 第一步）

## 用途

本模块把 MATR 原始数据中**一个循环**的 V/I/T 曲线标准化为固定长度
`(3, Tmax)` 的浮点数组（Tmax = 1000），对齐论文 *World Model for Battery
Degradation Prediction Under Non-Stationary Aging*（arXiv:2603.10527）的输入口径，
供后续窗口构建与模型训练使用。

它**只负责"循环 → (3, 1000)"这一步**；SOH 标签、I_mean（action）、
30 循环窗口构建属于后续模块，不在本文件范围内（单一职责）。

## 背景：论文输入口径与两种变体

论文原文：

> "The model input per cycle consists of raw time-series for voltage, current,
> and temperature, each padded or truncated to Tmax = 1000 timesteps."

论文同时出现 "recorded during discharge" 与 "sampled at approximately 1000
timesteps per cycle" 两处描述，而我们的数据中：

- 完整循环原始序列约 1087 点（充电 + 放电 + 静置）；
- 放电段（I < -3，4C）仅约 330 点。

两者都能解释为"pad/truncate 到 1000"，存在歧义（详见
`docs/m1_data_exploration.md` 第 6 节）。因此本模块提供两种变体，由
`mode` 参数切换：

| mode | 含义 | 贴合论文哪句话 |
| --- | --- | --- |
| `full`（默认） | 完整循环截断/填充到 1000 点 | "approximately 1000 timesteps per cycle" |
| `discharge` | 只取放电段再填充/截断到 1000 点 | "recorded during discharge" |

## 文件

```
DataLoader/
├── README.md               本文件
├── standardize_cycle.py    循环标准化模块（含演示）
├── imean.py                 逐循环平均充电电流（动作向量）
├── labels.py                SOH 标签（Q(2) 口径）
├── windows.py               窗口索引表（30 输入 + 80 未来）
├── data_quality.py          数据排除规则（论文口径 + 坏循环审计）
├── splits.py                数据划分（按电池 / 按协议，防泄漏）
└── build_dataset.py         M2 统一构建流水线（一键串联全部阶段）
```

## API

### load_raw_cycle(mat_path, cell, cycle) -> dict

读取一个循环的原始曲线（IO 职责）。返回：

- `V / I / T / t`：等长 float64 数组（t 单位=分钟，I 单位=C-rate，M1 已验证）；
- `n_cycles`：该电池总循环数；
- `policy`：协议名（用于元数据/图标题）。

### CycleStandardizer(tmax=1000, mode="full", discharge_threshold=-3.0, pad_mode="edge")

变换器（变换职责）。调用 `std(raw)` 返回 `(3, Tmax)` float32 数组，
行顺序 `[V, I, T]`。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `tmax` | 1000 | 目标序列长度 |
| `mode` | `"full"` | `"full"` 或 `"discharge"` |
| `discharge_threshold` | -3.0 | 放电段判定阈值（C-rate） |
| `pad_mode` | `"edge"` | 不足 Tmax 时的填充方式：`"edge"`（末尾值）或 `"zero"` |

边界行为：

- 超长：保留开头 Tmax 个点（截断）；
- 不足：末尾填充（默认用最后观测值，可选填 0）；
- 越界循环号、同族长度不一致、非法参数：直接抛异常（fail fast）。

## 用法

```powershell
python "src/world_model/DataLoader/standardize_cycle.py"
```

演示输出：电池 0 第 100 个循环在两种变体下的 `(3, 1000)` 数组与数值范围。

在其他脚本中复用：

```python
import sys
sys.path.insert(0, r"src/world_model/DataLoader")   # 目录含空格，无法作为包 import
from standardize_cycle import load_raw_cycle, CycleStandardizer

raw = load_raw_cycle("data/external/matr/MATR_batch_20170512.mat", cell=0, cycle=100)
X = CycleStandardizer(mode="full")(raw)   # (3, 1000) float32
```

## 注意事项

1. **目录名含空格**：`world_model` 不能被 Python 作为包 `import`。
   若后续需要 `from ... import`，建议将目录改名为 `world_model`；
2. **单位**：`t` 为分钟、`I` 为 C-rate（1C ≈ 1.1A），由 M1 电量守恒验证；
3. **截断方向**：超长时保留开头 Tmax 点，这是时间序列的常见约定；
   如后续实验需要保留结尾（放电段），可扩展 `_fix_length`。

## imean.py：平均充电电流（M2 · 第二步）

论文原文：The charging current I_mean is computed per cycle and used as the
action vector u(k) for the dynamics transition.

注意：不是全循环平均电流（充放电相消会接近 0），而是充电段（I > 0）
的平均电流，单位与原始 I 一致（C-rate）。它编码了该循环的充电倍率，
是跨电池/跨协议间唯一变化的运行条件（放电恒为 4C）。

API: mean_charging_current(I: np.ndarray) -> float

## labels.py：逐循环 SOH 标签（M2 · 第三步）

论文口径：SOH(k) = Q_discharge(k) / Q_ref，Q_ref = Q_discharge(cycle 2)。

- 只产出"逐循环 SOH 数组"；样本标签（当前 s(k) + 未来 80 个 s(k+1..k+80)）
  由 windows 模块用索引切出；
- 保留原表 soh 列（前 10 循环中位数口径），新增 soh_q2 列，便于两口径对比；
- 输入：data/processed/matr_soh_table.parquet
- 输出：data/processed/matr_soh_labels.parquet

API: add_soh_q2(table: pd.DataFrame) -> pd.DataFrame

## windows.py：窗口索引表（M2 · 第四步）

论文口径：窗口 = 30 个连续输入循环，输出当前 SOH + 未来 80 循环轨迹，
窗口只在本电池内部滑动，绝不跨电池。

- **索引式设计**：只记录 (cell_id, pos, start, k)，不物化曲线
  （全部窗口物化约 36GB，索引表仅几十万行）；
- 合法性：k 前至少有 29 个循环，且 k + 80 <= 电池总循环数；
- 坏循环清洗：触碰坏循环（data_quality.BAD_CYCLES）的窗口整窗丢弃，
  不插值、不整删电池（详见 data_quality.py 一节）；
- 附带验证：未来 80 循环内 SOH 跌破 0.95 的窗口比例（对照论文 10.9%）。

API:
- build_window_table(labels, W=30, H=80, stride=1) -> DataFrame
- window_crossing_rate(table, labels, H=80, threshold=0.95) -> float
- count_bad_windows(table, labels, W=30, H=80) -> int

输出：data/processed/matr_windows.parquet

### 窗口表字段字典（matr_windows.parquet）

每行 = 一个窗口（某电池的某个 k 位置）；该表**只存索引，不存曲线和标签**。

| 列 | 类型 | 含义 | 取值 / 示例 |
| --- | --- | --- | --- |
| `cell_id` | str | 电池唯一标识（批次_编号） | `2017-06-30_c000`（138 个）|
| `pos` | int | 窗口末尾循环在该电池内的 0-based 位置，用于索引曲线数组 | 29 ~ 2155 |
| `start` | float | 输入窗口起始循环的 cycle_index（= k − 29）| 2 ~ 2128 |
| `k` | float | 输入窗口末尾循环的 cycle_index（当前循环，标签 s(k) 所在）| 31 ~ 2157 |
| `batch` | str | 制造批次 | `20170512` / `2017-06-30` / `2018-04-12` |
| `policy` | str | 充电协议名（快充协议） | `1C(4%)-6C`（72 个）|
| `stage` | str | 该窗口当前 SOH s(k) 的老化阶段（s1/s2/s3，s4 无窗口）| `s1_healthy` 等 |

对应关系（已在代码中验证）：
- `k` = 该电池排除 cycle 1 后的循环列表在 `pos` 处的 cycle_index；
- `start` = 同列表在 `pos − 29` 处的 cycle_index；
- 输入窗口 = 循环 `start .. k`（30 个连续循环）；输出 = 当前 SOH s(k) + 未来 80 个 s(k+1..k+80)。

## 数据不平衡处理（论文 Imbalance handling，M3 训练侧）

论文观测：窗口的 SOH 分布严重右偏——83% 样本在 0.95 以上、11% 在
0.90–0.95、6% 在 0.85–0.90。若直接按均匀采样训练，模型会退化为
"总是预测健康"的平凡解。

论文对策：按四档老化阶段（s1_healthy / s2_mild / s3_aged / s4_heavy）
做**逆频率采样**（inverse-frequency sampling），保证每个 batch 内
各老化阶段均衡出现。

- 窗口表已带 `stage` 列（按当前 SOH s(k) 划分，见 windows.py）；
- 逆频率采样权重属于训练数据加载器（M3），不在本阶段实现；
- 本数据集窗口分布实测：83.4% / 14.1% / 2.5% / 0.0%
  （s1 与论文 83% 吻合；低档差异源于电池构成与 k+80 约束，
  详见 docs/m1_data_exploration.md 复现记录）。

注意：由于窗口要求 k + 80 <= 总循环数，每只电池**最后 80 个循环
不会产生窗口**，这正是 s4（<0.85）占比为 0 的原因——也是逆频率
采样要解决的偏置来源之一。

## data_quality.py：数据排除规则（M2 · 第五步）

论文口径：
- 排除 batch 1 的 cell 0 与 cell 18（设备故障）；
- 排除每只电池的 cycle 1（数据质量，Severson 约定）；
- 得到 138 只可用电池（44 + 48 + 46）。

我们的批次构成：b1 排除后 44 只、b2 48 只、b4 46 只，共 138 只；
与论文的 b1/b2 数量一致，第三批次不同（我们无 2017-07-25，用 2018-04-12
替代），报告中需注明。

两级排除策略（2026-08-12 全量审计后定稿）：
- **整电池排除**（EXCLUDED_CELLS）：维持论文原判，只有 c000 / c018 两只；
- **循环级排除**（BAD_CYCLES）：审计发现 batch 2（2017-06-30）c012/c044
  有同型容量越界尖峰（1.489 / 1.545 Ah > 1.1 Ah 物理上限），以及 10 只
  batch 2 电池在 cycle 247-259 的放电中断、c005@909、c037 的 593-600
  噪声段——均为单循环或短段事件，不整删电池，改为窗口级清洗
  （windows.py 自动避开触碰坏循环的窗口）。

- windows.py 默认应用该规则，`--keep-all` 可关闭（消融用）；
- 参考循环仍为 cycle 2（cycle 1 排除不影响 SOH 口径）。

## splits.py：数据划分（M2 · 第六步）

方案口径（docs/b_plan_design.md §1.3）：
- **划分单元 = 电池**：同一电池的全部窗口必须属于同一集合（防泄漏）；
- **主线 split_by_cell**：按电池随机 70/15/15，并在每个 batch 内部分层，
  检验"未见过的新电池"；
- **扩展 split_by_policy**：以充放电协议为划分单元（同协议跨批次也同组），
  测试协议在训练中完全不可见，检验"未见工况"。

当前结果（seed=42）：
- 按电池：train/val/test = 97/21/20 只电池，68,984 / 13,750 / 13,398 窗口；
- 按协议：train/val/test = 96/22/20 只电池，64,994 / 15,001 / 16,137 窗口；
- 防泄漏自检：同一电池不跨集合（脚本内断言）。

输出：data/processed/splits.parquet（cell_id -> split_by_cell / split_by_policy）

## build_dataset.py：统一构建流水线（M2 · 收尾）

一键执行完整 M2 数据构建：

    python "src/world_model/DataLoader/build_dataset.py"

四阶段依赖自动补齐（soh_table -> labels -> windows -> splits），
可只跑子集（`--stages windows,splits`），支持 `--keep-all`（消融）、`--seed`。
任一阶段失败即中断，不留半成品；构建后打印全部产物汇总。

## 下一步（M3 起点）

1. 数据集加载器：按窗口索引读取 V/I/T 曲线与标签、
   四档老化阶段逆频率采样、标准化统计量只在训练集拟合；
2. 坏循环审计记录（data/processed/_audit_all_cells_qd.csv）择期并入正式文档。
