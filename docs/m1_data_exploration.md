# M1 数据探索总结（存档）

> 日期：2026-08-12 ｜ 状态：M1 完成，进入 M2 前的事实基础
> 关联：`scripts/explore_matr_curves.py`、`docs/report_data_processing.md`

## 1. 文件格式

- MATR `.mat` 为 MATLAB v7.3 格式，本质是 HDF5，用 `h5py` 读取；
- `batch` 下所有字段第一维 = 电池数（46），**先选电池 i，再进循环 j**。

## 2. 数据规模

- 3 个批次：2017-05-12（46 只）、2017-06-30（48 只）、2018-04-12（46 只），共 140 只电池、72 种协议；
- 循环数 170–2237；SOH 最低 0.761；电池 0（b1）有 1189 个循环，cycle_life = 1190。

## 3. 字段结构

| 视图 | 字段 | 内容 |
| --- | --- | --- |
| 汇总 | summary | 每循环一行标量：cycle, QCharge, QDischarge, IR, Tavg/Tmax/Tmin, chargetime |
| 曲线 | cycles | 每循环一条曲线：V, I, T, t, Qc, Qd, Qdlin, Tdlin, discharge_dQdV |
| 元数据 | barcode / channel_id / policy / policy_readable / cycle_life | 电池标识、协议（ASCII/UTF-16）、寿命 |
| 特殊 | Vdlin（batch 级） | 每只电池仅一条 (1, 1000) 放电电压曲线，不逐循环 |

## 4. 单位与窗口（已破解）

- `t` 单位 = 分钟（一个循环 ≈ 54 分钟）；
- `I` 单位 = C-rate（1C ≈ 1.1A；放电恒为 4C，I.min() ≈ -4.001）；
- 验证：`trapz(I[I>0], t[I>0]) = 58.386 C·min`，`58.386 × 1.1 / 60 = 1.0705 Ah ≈ Qc 终点 1.071` ✅；
- 1087 点窗口 = 完整循环：充电 2.0→3.5V（含 LFP 平台区）→ 放电 3.5→2.0V；
- 原始放电段（I < -3）：约 330 点、14.4 分钟、V 从 3.5 到 2.0；
- 采样不均匀（开头密、中间疏）。

## 5. dlin 族（1000 点）

- dlin = discharge linearized：放电段按容量等距 1000 点；
- `Qdlin` = 容量坐标（0→1.05Ah），`Tdlin` = 放电温度，均在 cycles 内逐循环存在；
- `discharge_dQdV` = 增量容量曲线（1000 点，逐循环）；
- **逐循环的线性化电压无现成版本**（batch 级 Vdlin 每电池仅一条）。

## 6. 论文（World Model, arXiv 2603.10527）输入组织

- 输入 = 每循环原始 V/I/T 时间序列，**pad/truncate 到 Tmax=1000**；
- 明确排除放电容量 Qd 与内阻 IR（论文称需实验室级设备）；
- 标签：SOH = Q(t)/Q(2)（第 2 循环参考，注意与我们的前 10 循环中位数不同）；
- 窗口 W=30 循环；输出当前 SOH + 未来 H=80 循环轨迹；
- **修正记录**：早期判断"使用 dlin"被原文否决——原文措辞为 "raw time-series ... padded or truncated to Tmax=1000"；
- **剩余歧义**：完整循环截断到 1000 点 vs 放电段填充到 1000 点（原文 "recorded during discharge" 与 "approximately 1000 timesteps per cycle" 两处说法冲突）；论文无公开代码；
- 对策：M2 做双变体消融，用论文核心结论（迭代 rollout 相对直接回归未来轨迹误差减半）作判别。

## 7. M2 待办

1. 构建逐循环曲线数据集：每循环 (V, I, T) → 1000 点，双变体（完整循环截断 / 放电段填充）；
2. SOH 标签按 Q(2) 口径重算；
3. 确认 batch 级 Vdlin 是否可用（或自行插值逐循环放电电压）；
4. 按电池 70/15/15 划分，防泄漏。

## 8. 方法论收获

- h5py 读取 + 解引用（deref）模式；
- 电量守恒验证单位：`trapz(I, t)` 对照 Qc；
- 调试纪律：print shapes、assert 长度一致、读完整 traceback；
- 引文纪律：原文引用与个人解读必须分开标注。