# DataLoader

这一层负责从 Severson LFP 原始数据中生成“部分充电片段”训练样本。

## 模块划分

每个文件只做一件事：

| 文件 | 职责 |
| --- | --- |
| `mat_io.py` | 扫描 .mat 文件；读取一个循环的原始 V/I/T/t/Qc/Qd |
| `charge.py` | 从完整循环中提取 I > 0 的充电阶段 |
| `segments.py` | 按论文的 20% 容量窗口、0%–50% 起点、1% 步长生成片段索引，并做插值 |
| `quality.py` | 按 Severson 124 / Scientific Reports 123 口径排除电池、cycle 1，并标记坏循环 |
| `labels.py` | 生成两种 SOH 标签：`soh_nominal` 与 `soh_q2` |
| `splits.py` | 按电池生成 train/val/test 划分，防止片段泄漏 |
| `build_dataset.py` | 编排以上模块，输出片段索引表 |

## 已实现用法

### 1. 读取原始 MAT

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/DataLoader/mat_io.py"
```

### 2. 提取充电阶段

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/DataLoader/charge.py"
```

### 3. 生成部分充电片段预览

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/DataLoader/segments.py"
```

默认对第一只电池的前 3 个循环生成片段索引，输出到：

`data/processed/partial_segments_preview.parquet`

### 4. 生成标签

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/DataLoader/labels.py"
```

输出：`data/processed/partial_soh_labels.parquet`

### 5. 生成电池级划分

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/DataLoader/splits.py"
```

默认使用论文的 99/24 口径（train/test）。该口径要求先过滤到 123 只电池：

- 140 channels → 124 cells：应用 Severson 2019 的 16 只排除规则；
- 124 cells → 123 cells：再排除 1 只异常短寿命电池 `b2c1`。

123 只电池 = 41（batch 1）+ 42（batch 2）+ 40（batch 3）。

开发阶段如不想用论文口径，可用 `--strategy ratio` 切到 70/15/15。

### 6. 生成片段索引表

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/DataLoader/build_dataset.py"
```

默认只处理 3 只电池、每只 3 个循环，输出：

`data/processed/partial_segments_index.parquet`

## 论文切片口径

- 额定容量：1.1 Ah；
- 观测窗口：20% 额定容量 = 0.22 Ah；
- 起点：0%–50% 额定容量，每次移动 1%，共 51 个起点；
- 每 1% 容量区间插值为 5 步；
- 20% 窗口因此产生 100 个间隔、101 个网格点；
- 电压预训练预测窗口：7% 额定容量 = 0.077 Ah。

## 必须遵守的原则

- 先按电池划分训练/测试，再切片段，防止同电池泄漏；
- 片段起点、长度、电流协议等条件信息作为显式字段保存；
- 标签为片段所属循环的当前 SOH；
- 曲线数据暂时不物化到 parquet，先存索引，后续按需读取，避免磁盘爆炸。

## 124-cell 口径来源

本地三个 .mat 文件共 140 个 channel。Severson 2019 原始代码
`generate_voltage_arrays.m` 把 140 过滤成 124（三类规则）：

| 批次 | 文件内电池数 | 排除数 | 保留数 | 排除原因 |
| --- | --- | --- | --- | --- |
| 2017-05-12 (batch 1) | 46 | 5 | 41 | 实验未跑到寿命终点 |
| 2017-06-30 (batch 2) | 48 | 5 | 43 | 从 batch 1 续测，避免重复计数 |
| 2018-04-12 (batch 3) | 46 | 6 | 40 | channel 46 + 2 只未衰减到 EOL + 3 只噪声 |
| 合计 | 140 | 16 | 124 | |

Scientific Reports 2026 再从 124 里排除 1 只异常短寿命电池 `b2c1`
（cycle_life 约 148，是唯一低于约 300 的离群值），得到 123 只。
