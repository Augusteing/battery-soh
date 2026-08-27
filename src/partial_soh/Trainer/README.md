# Trainer

本目录实现 Scientific Reports 2026 的迁移学习训练流程：

1. 电压预测预训练；
2. SOH 回归微调；
3. 与直接训练 LSTM 等基线做内部比较（已实现，见下方“当前基线结果”）。

## 文件

| 文件 | 职责 |
| --- | --- |
| `dataset.py` | 惰性加载的 PyTorch Dataset：索引 -> MAT -> 插值 -> `(101, 3)` 张量 |
| `model.py` | 共享编码器（嵌入 + LSTM）+ 电压头 / SOH 头([h;c]) / 重建头，含未来窗 rollout |
| `trainer.py` | 预训练 + 微调 + 评估的入口 |
| `consistency.py` | 创新 1：同循环一致性约束（分组采样器 + 一致性损失） |
| `ssl_tasks.py` | 创新 2：扩展自监督（掩码电压重建） |
| `build_cache.py` | 把全部片段一次性物化到磁盘 memmap（训练提速） |
| `run_ablation.py` | 一键顺序跑 5 个消融配置并汇总指标/画图 |

## 模型输入 / 输出

- 输入：一个部分充电片段，重采样为 `(101, 3)`，三通道依次为
  电流 `I`、电压 `V`、容量坐标 `Q`（步长 `d=0.0022` Ah）。
- 预训练输出：观测窗内“下一步电压”（目标 `V[1:101]`）+ 未来 7% 容量窗的
  自回归电压 rollout（目标为该窗真实电压，形状 `(36,)`）。
- 微调输出：标量 SOH，形状 `()`，标签为 `soh_nominal = Q_charge / 1.1`。

## 当前结果（测试集，10 预训练 + 10 微调 epoch，seed=42）

| 方法 | test MAE | test RMSE |
| --- | --- | --- |
| 基线（论文复现） | 2.2651% | 2.7706% |
| 只改采样（对照，λ=0） | 2.2045% | 2.6693% |
| 同循环一致性（创新 1） | 2.1118% | 2.5841% |
| 掩码电压重建（创新 2） | 2.6694% | 3.1530% |
| 完整方案（一致+重建） | **2.0439%** | **2.5188%** |
| 论文报告（迁移，50+50） | 0.91% | 1.30% |

## 最终结果（50 预训练 + 50 微调 epoch，seed=42，batch 4096）

| 配置 | Test MAE | Test RMSE |
| --- | --- | --- |
| **纯基线（新架构，50+50）** | **1.2466%** | **1.8760%** |
| 完整方案（一致+重建，新架构，50+50） | 1.7971% | 2.5405% |

说明（严格同预算消融，seed=42，batch 4096）：
- 50+50 纯基线取得全场最佳 **1.2466%**，已显著优于历史单次 1.80%，
  并逼近论文 0.91%（注意：划分与论文不同，数字仅作语境对照）；
- 一致性 + 重建在短预算（10+10，旧架构）有帮助（2.27%→2.04%），
  但在 50+50 长训练下反而限制拟合（完整方案训练 MAE 1.36% vs
  基线 0.73%），这是消融给出的明确结论，报告需如实呈现。

说明：早期记录中的“迁移学习 LSTM 1.80% / 直接 LSTM 1.87%”是用**未提交
git 的旧版训练代码**跑出的单次结果（seed 未记录、代码已丢失），当前
可复现代码无法复现，仅作历史参考；正式对比以本表的 seed=42 协议为准，
并建议后续用 3 个 seed 报均值±标准差。

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

## 创新特性（本分支）

在论文复现基线之上，本项目新增两个创新点，都通过命令行开关控制：

### 创新 1：同循环一致性约束（`--consistency`）

同一个循环可以切出最多 51 个部分充电片段，它们的 SOH 标签完全相同。
一致性约束要求：同一循环的所有片段，模型输出必须彼此接近。

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/trainer.py" `
  --preload --consistency --pretrain-epochs 50 --finetune-epochs 50
```

可选参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--group-size` | 4 | 每个循环抽取 K 个片段组成一组 |
| `--batch-groups` | 512 | 每批包含多少个循环（有效 batch = 512×4 = 2048） |
| `--consist-lambda` | 1.0 | 一致性损失权重；设为 0 等于“只改采样、不加约束” |

分组模式下每个 epoch 的更新次数会与普通模式对齐（按样本数折算），
保证消融实验里不同配置“看到的数据量”相同，只有采样方式 / 损失项不同。

### 创新 2：扩展自监督（掩码电压重建，`--recon-loss`）

在原版“下一步电压预测”之外，随机遮掉 30% 的电压点，让模型用上下文
把它们重建出来，迫使编码器理解电压曲线的平滑结构与 LFP 电压平台。

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/trainer.py" `
  --preload --recon-loss --pretrain-epochs 50 --finetune-epochs 50
```

可选参数：`--mask-ratio`（默认 0.3）、`--recon-lambda`（默认 1.0）。

### 消融实验设计

| 配置 | 采样方式 | 损失项 | 说明 |
| --- | --- | --- | --- |
| 基线（论文复现） | 普通 shuffle | 数据损失 | 不加任何创新 |
| 只改采样（对照） | 循环分组 | 数据损失 | `--consistency --consist-lambda 0` |
| + 同循环一致性 | 循环分组 | 数据 + 一致性 | `--consistency` |
| + 扩展自监督 | 普通 shuffle | 数据 + 重建 | `--recon-loss` |
| 完整方案 | 循环分组 | 数据 + 一致性 + 重建 | 两个开关都开 |

冒烟验证已通过；全量消融（10+10 epoch）进行中，
结果写入 `results/metrics/ablation_consistency_ssl.json`。

建议先用小规模冒烟测试确认数值正常，再做全量消融：

```powershell
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/trainer.py" `
  --max-samples 3000 --batch-size 256 --pretrain-epochs 1 --finetune-epochs 1 `
  --consistency --group-size 2 --batch-groups 16 --recon-loss
```

## 磁盘缓存（推荐，训练提速）

训练时每个 batch 都要在 CPU 上把几千个片段插值到 101 点容量网格，
CPU 成为瓶颈（实测约 0.25s/step，一个 epoch 超过 5 分钟）。
片段是静态数据，可以一次性物化到磁盘：

```powershell
# 先构建 train 与 test 的缓存（各约 15 / 3 分钟，共约 6 GB）
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/build_cache.py" --split train
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/build_cache.py" --split test

# 之后训练加 --cache-dir 即可直接读缓存（跳过 MAT 读取与插值）
& "E:\conda\envs\battery-soh\python.exe" "src/partial_soh/Trainer/trainer.py" `
  --cache-dir "data/processed/segments_cache" `
  --consistency --recon-loss --pretrain-epochs 10 --finetune-epochs 10
```

`run_ablation.py` 默认就会带上 `--cache-dir`，只要先构建过缓存即可。

## 当前简化（后续补齐）

- 输入通道目前是 [I, V, Q]，尚未加入温度 T；等补充带温度变化的数据集
  后，会把 T 作为第 4 通道接入，并用“跨温度验证”检验温度泛化能力；
- 输入暂未做 z-score 标准化，V/I/Q 使用原始单位；
- 梯度裁剪默认 1.0。论文 Table 1 写的是 0.0005，量级可疑，待对照补充材料
  后再决定是否严格采用；batch 用 4096 + 梯度累积（`--accum-steps 5`）模拟
  论文的 batch 20,000；
- 论文的 99/24 划分未公开具体名单，当前用 seed=42 的随机划分，仅能
  “按协议组成尽量接近”，严格数字对比需自行复跑统一协议。
- 划分沿用 99 train / 24 test 的随机划分，尚未按充电协议分层。
