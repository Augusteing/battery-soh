# Trainer

本目录实现 Scientific Reports 2026 的迁移学习训练流程：

1. 电压预测预训练；
2. SOH 回归微调；
3. 与直接训练 LSTM、CNN、Random Forest 等基线做内部比较（基线尚未实现）。

## 文件

| 文件 | 职责 |
| --- | --- |
| `dataset.py` | 惰性加载的 PyTorch Dataset：索引 -> MAT -> 插值 -> `(101, 3)` 张量 |
| `model.py` | 共享编码器（嵌入 + LSTM）+ 电压头 / SOH 头 |
| `trainer.py` | 预训练 + 微调 + 评估的入口 |

## 模型输入 / 输出

- 输入：一个部分充电片段，重采样为 `(101, 3)`，三通道依次为
  电流 `I`、电压 `V`、容量坐标 `Q`（步长 `d=0.0022` Ah）。
- 预训练输出：每个时间步的“下一步电压”，形状 `(101,)`，目标为 `V[1:101]`。
- 微调输出：标量 SOH，形状 `()`，标签为 `soh_nominal = Q_charge / 1.1`。

## 运行

冒烟测试（小样本，快速验证）：

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/trainer.py" `
  --max-samples 2000 --batch-size 256 --pretrain-epochs 1 --finetune-epochs 1
```

全量训练（预加载曲线到内存，避免反复读 MAT）：

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/trainer.py" `
  --preload --pretrain-epochs 50 --finetune-epochs 50
```

`--preload` 会把约 8 万条充电曲线读进内存（约 1 GB），耗时约 8 分钟，
之后训练不再读 MAT。

## 当前简化（后续补齐）

- 预训练目标当前是“观测窗内的下一步电压预测”。论文还提到 7% 容量的
  未来预测窗口，尚未接入；
- 输入暂未做 z-score 标准化，V/I/Q 使用原始单位；
- 梯度裁剪默认 1.0。论文 Table 1 写的是 0.0005，量级可疑，待对照补充材料
  后再决定是否严格采用；
- 划分沿用 99 train / 24 test 的随机划分，尚未按充电协议分层。
