# Trainer：M3 训练侧模块

## 用途

M2（`DataLoader/`）产出了窗口索引表与划分；M3 负责把索引变成可训练的
数据流并完成模型训练与评估：

1. ~~数据集加载器~~（已完成 dataset.py）
2. ~~标准化~~（已完成 normalize.py，参数 data/processed/normalizer.json）
3. **逆频率采样**：四档老化阶段按 batch 内逆频率加权（论文 Imbalance
   handling），训练器里用 WeightedRandomSampler 实现；
4. **训练目标**：L = L_data + 0.1·L_phys（§3.3–3.4），EWC 接口保留（§3.5）。
5. **训练与评估**：训练循环（trainer.py）+ 测试集报告（evaluate.py，
   已完成）：整体 / 老化阶段 / 批次 / 逐电池 MAE + 三张诊断图。

## 规划文件（按实现顺序）

```
Trainer/
├── README.md      本文件
├── dataset.py     窗口数据集：曲线 + 标签 + 阶段权重（已完成）
├── normalize.py   标准化：训练集拟合，验证/测试集复用（已完成）
├── model.py       World Model 架构（四个部件 + 完整组装已完成）
├── loss.py        训练目标：数据损失 + 物理约束 + EWC 接口（已完成）
├── sampler.py     逆频率采样器（可选：集成进 dataset 也可）
├── trainer.py     训练循环：论文 §4 配置（已完成）
└── evaluate.py    测试集评估：多粒度 MAE + 诊断图（已完成）
```

### 已完成模块速览

**dataset.py** — `WindowDataset` 按窗口索引惰性读取曲线（LRU 缓存电池级数据），
每个样本返回 X=(30,3,1000)、y_cur、y_fut=(80,)、stage；`make_stage_weights`
按 (batch, stage) 计算逆频率采样权重。

**normalize.py** — `ChannelNormalizer` 对 V/I/T 逐通道 z-score：
`fit`（训练集流式拟合）→ `save`（JSON）→ `transform`（验证/测试复用参数），
并提供 `inverse_transform` 还原量纲。拟合命令：

    python "src/world_model/Trainer/normalize.py"

拟合结果（训练集 97 只电池）：V=3.05±0.57，I=0.28±3.21，T=34.3±2.83

**model.py（CycleEncoder + PatchTSTEncoder）** —
第一部分（CycleEncoder）：论文原文的共享 1D CNN：
Conv1d(3→32,k7,s2)→BN→ReLU→Conv1d(32→64,k5,s2)→…→Conv1d(64→128,k3,s2)
→AdaptiveAvgPool→Linear(128→64)。输入 (…, 3, Tmax) 输出 (…, 64)，
支持 (B,W,3,T) 批量处理（前导维自动折叠，共享权重逐循环编码）。
参数量 44,416。
第二部分（PatchTSTEncoder）：e(k) 序列 (B,30,64) 按 P=6/S=3 切出 9 个
patch → Linear(384→64) 投影 → 正弦位置编码 → nn.TransformerEncoder
（L=3、4 头、d_ff=256）→ AdaptiveAvgPool → z(k) (B,64)。
Transformer 与池化复用 PyTorch 现成组件；patch 切分与正弦编码为手写。
参数量 174,592。
第三部分（DynamicsTransition）：z(k+1) = z(k) + MLP([z(k) ∥ u(k)])，
两层 MLP（65→128→64）+ 残差连接；`rollout(z, u, H=80)` 迭代 H 步生成
未来潜在状态序列。残差分支**零初始化**（训练起点 = 状态不变，只学有界
增量，防止随机初始化下 80 步滚动指数爆炸）。参数量 16,704。
数据集已提供 action u(k)=I_mean（充电段平均电流，C-rate）。
第四部分（SOHHead）：共享两层 MLP（64→64→1），当前状态与 rollout 每个
未来状态共用同一个头。参数量 4,225。
**WorldModel 组装**：X (B,30,3,1000) + u (B,) -> (ŝ(k) (B,), ŝ(k+1..k+80) (B,80))。
总参数量 239,937；已用真实数据验证前向形状与反向梯度（loss 能一路传回
卷积层，无参数缺失梯度）。

**loss.py** — 训练目标（论文 §3.3–3.5）：
- `data_loss`：式 7，当前 SOH + 未来 H 步轨迹的 MSE；
- `monotonicity_loss`：式 8，含容差 eps=0.005 的单调下降损失；
- `resistance_soh_loss`：式 10-11，内阻隐含 SOH 一致性，gamma=0.75；
  IR=0 时自动跳过无效样本；
- `voltage_consistency_loss`：论文只有文字描述（"relative check"），
  暂无公式，默认关闭（use_voltage=False）；
- `WorldModelLoss`：按式 6 组合，LAMBDA_PHYS=0.1，同时返回每个分量便于
  消融 / 调试；
- `EWC` / `ewc_loss`：式 13-14 的实现和防泄漏（Fisher 只能在模型未收敛时
  计算）。主配置不使用（论文 §4：all data simultaneous），留给分阶段部署实验。

演示命令：

    python "src/world_model/Trainer/loss.py"

**trainer.py** — 训练循环（论文 §4 配置）：
- 优化器 Adam（lr=1e-3，weight decay=1e-4），梯度裁剪 L2=1.0；
- 早停：验证 MAE 连续 15 轮无提升即停，最多 100 轮；
- 逆频率采样（WeightedRandomSampler）+ 每 epoch 样本量上限
  `--max-samples`（0=全量，论文口径；CPU 快速迭代可设 3000）；
- 训练前把全部电池曲线预载进内存（约 2GB，避免随机采样反复读盘），
  可用 `--no-preload --cache-size N` 关闭以省内存；
- 输出到 `results/runs/<run_name>/`：最优权重 checkpoint.pt、
  每 epoch 指标 metrics.json、学习曲线 learning_curve.png。

运行示例（快速出第一批结果）：

    python "src/world_model/Trainer/trainer.py" --max-samples 3000 --epochs 10

**evaluate.py** — 测试集正式评估（训练阶段不碰测试集，防信息泄漏）：
- 加载 checkpoint，在测试集（默认 20 只电池、全部窗口）上做预测；
- 输出整体 / 老化阶段（s1/s2/s3）/ 制造批次 / 逐电池的 MAE，
  以及窗口级 vs 电池级两种口径；
- 三张诊断图：误差随预测距离曲线、逐电池误差条形图、
  深度老化电池的真实 vs 预测轨迹；
- 结果存为 `results/runs/<run_name>/test_report.json` 与 `figures/`。

运行示例：

    python "src/world_model/Trainer/evaluate.py" \
        --checkpoint "results/runs/run1_baseline/checkpoint.pt"

首个基线（run1_baseline，10 epoch × 3000 窗口/epoch）测试集结果：
MAE 当前 SOH 0.0096，未来轨迹 0.0119（窗口级）/ 0.0179（电池级），
h=80 0.0163；按阶段 s1 0.0109 → s2 0.0185 → s3 0.0194，
深度老化电池仍是主要短板。

## 设计约束（与 M2 一致的软件工程原则）

- **防泄漏**：同一电池的全部窗口只出现在一个集合；标准化统计量、采样权重
  一律只用训练集信息；
- **索引优先**：窗口表不物化曲线，训练时按需读取，避免 36GB 内存峰值；
- **对齐口径**：pos 是排除 cycle 1 之后的 0-based 位置，读取曲线时必须用
  与 windows.py 相同的排除规则对齐循环列表；
- **fail fast**：索引越界、标签缺失、参数非法立即报错。

## 输入 / 输出

- 输入：`data/processed/matr_windows.parquet`、`matr_soh_labels.parquet`、
  `splits.parquet`、`data/external/matr/*.mat`
- 输出：`results/`（模型权重、指标 JSON、预测 parquet、图表）
