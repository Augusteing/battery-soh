# DataLoader

本层负责把 LFP 数据集（MATR / SIT，以及后续 HUST / SNL）统一成
“部分充电片段”训练样本，供 Trainer 使用。

## 模块划分（每个文件只做一件事）

| 文件 | 职责 |
| --- | --- |
| `registry.py` | 数据集注册表：路径、温度、电池文件发现、统一加载、温度推断 |
| `mat_io.py` | 读取 Severson .mat（MATR）→ 统一循环结构（含单位换算） |
| `sit_io.py` | 读取 SIT（新加坡理工）xlsx+CSV → 统一循环结构（20 只 50Ah LFP，环境温/40°C） |
| `segments.py` | 20% 容量窗口、0–50% 起点、1% 步长的片段切分（4 通道 I/V/Q/T） |
| `labels.py` | 生成 SOH 标签（统一以 cycle 2 放电容量为参考） |
| `splits.py` | 按电池 / 按温度生成 train/val/test 划分，防止片段泄漏 |
| `build_index.py` | 编排以上模块，输出统一片段索引表（parquet） |

## 状态

- [x] `registry.py`：三个数据集的登记、扫描、pkl 统一加载、温度推断
- [x] `mat_io.py`：MATR .mat -> 统一循环结构（C-rate->A、分钟->秒）
- [x] `sit_io.py`：SIT xlsx -> 统一循环结构（Cycle_Summary 映射 + 内嵌温度）
- [x] `segments.py`：4 通道片段索引 + 插值（冒烟：SNL/MATR 均输出 51 起点）
- [ ] `labels.py` / `splits.py` / `build_index.py`

## SIT 数据（跨温度 LFP）

SIT 数据集提供 20 只 50 Ah 方形 LFP 电芯的完整老化轨迹：

  - Repower_001（8 只）与 Chroma_101（2 只）：环境温度；
  - Repower_002（6 只）与 Repower_003（4 只）：40°C 恒温箱。

这是目前唯一包含**温度变化**的 LFP 公开老化数据，将用于
温度嵌入模块的"变温微调"与跨温测试。注意电芯规格与 Severson
（1.1 Ah 18650）不同，跨电芯训练需把输入归一化为 SOC / C-rate。

## 运行

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/temperature_soh/DataLoader/registry.py"
```

输出三个数据集的电池文件统计与温度分布。
