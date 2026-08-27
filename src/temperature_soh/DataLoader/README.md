# DataLoader

本层负责把三个 LFP 数据集（MATR / HUST / SNL）统一成
“部分充电片段”训练样本，供 Trainer 使用。

## 模块划分（每个文件只做一件事）

| 文件 | 职责 |
| --- | --- |
| `registry.py` | 数据集注册表：路径、温度、电池文件发现、统一加载、温度推断 |
| `pkl_io.py` | 读取 BatteryLife 标准化 pkl（SNL / HUST）→ 统一循环结构 |
| `mat_io.py` | 读取 Severson .mat（MATR）→ 统一循环结构（对齐 partial_soh 口径） |
| `segments.py` | 20% 容量窗口、0–50% 起点、1% 步长的片段切分（含温度通道） |
| `labels.py` | 生成 SOH 标签（统一以 cycle 2 放电容量为参考） |
| `splits.py` | 按电池 / 按温度生成 train/val/test 划分，防止片段泄漏 |
| `build_index.py` | 编排以上模块，输出统一片段索引表（parquet） |

## 状态

- [x] `registry.py`：三个数据集的登记、扫描、pkl 统一加载、温度推断
- [ ] `pkl_io.py` / `mat_io.py`：统一循环结构（下一步）
- [ ] `segments.py` / `labels.py` / `splits.py` / `build_index.py`

## 运行

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/temperature_soh/DataLoader/registry.py"
```

输出三个数据集的电池文件统计与温度分布。
