"""World Model 模型架构（M3 · 第三步：完整模型）。

论文口径（World Model, arXiv:2603.10527）：
  "Each cycle's raw time-series [V(t), I(t), T(t)] ∈ R^{3×Tmax} is processed
   independently by a shared 1-D convolutional neural network. The encoder
   consists of three Conv1d layers (channels 32→64→128, kernels 7/5/3,
   stride 2) with batch normalisation and ReLU activation, followed by
   adaptive average pooling and a linear projection to a d-dimensional
   embedding e(k) ∈ R^d, where d = 64."

已完成：
  - CycleEncoder（第一部分：单个循环曲线 -> 嵌入 e(k)）
  - PatchTSTEncoder（第二部分：e(k) 序列 -> 潜在退化状态 z(k)）
  - DynamicsTransition（第三部分：带残差的 action 条件状态转移 + 滚动预测）
  - SOHHead（第四部分：共享两层 MLP 解码头）
  - WorldModel（完整组装：输入窗口 -> 当前 SOH + 未来轨迹）

论文原文（Output Head）:
  "A single shared head (two-layer MLP with ReLU) maps any latent vector to
   a scalar SOH estimate. The future trajectory is obtained by applying this
   head to each step of the dynamics rollout."

可复用性说明（针对"PyTorch 有没有现成的"）:
  - Transformer 编码器直接用 nn.TransformerEncoder（现成）；
  - 自适应池化用 nn.AdaptiveAvgPool1d（现成）；
  - Patch 切分与投影、正弦位置编码是 PatchTST 特有逻辑，需手写
    （代码量各只有几行，见下方实现）。

设计说明（软件工程）
---------------------
- 单一职责：CycleEncoder 只负责"一个循环 -> 一个嵌入"，不做窗口/序列逻辑；
- 任意前导维度：forward 接受 (B, 3, T) 或 (B, W, 3, T)，
  自动把前导维折叠成 batch 编码后再还原形状（共享权重，逐循环独立处理）；
- 忠实论文：卷积核/通道/步长按原文实现，不加 padding（尾部靠
  AdaptiveAvgPool1d 收成单点，长度变化不影响输出维度）；
- 可复现：演示入口固定随机种子。

用法（演示）:
    python "src/world_model/Trainer/model.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/world_model/Trainer"))

from dataset import WindowDataset                                # noqa: E402
from normalize import ChannelNormalizer                          # noqa: E402


class CycleEncoder(nn.Module):
    """共享 1D CNN：单个循环曲线 (3, Tmax) -> 嵌入 e(k) ∈ R^d。

    结构（按论文原文）:
        Conv1d(3→32, k=7, s=2) -> BN -> ReLU
        Conv1d(32→64, k=5, s=2) -> BN -> ReLU
        Conv1d(64→128, k=3, s=2) -> BN -> ReLU
        AdaptiveAvgPool1d(1) -> flatten
        Linear(128 -> d)

    说明:
    - 三层卷积逐步"看"越来越大的感受野：第一层看曲线局部形状，
      最后一层看整体结构；stride=2 让序列长度逐层减半（1000 -> 497
      -> 247 -> 123），自适应池化把任意剩余长度收成 1 个点；
    - BatchNorm 在训练时用当前 batch 统计量、评估时用历史运行统计量，
      所以 eval() 前后输出会略有不同（这是 BN 的正常行为）；
    - 输入应为 z-score 标准化后的曲线（见 normalize.py）。
    """

    def __init__(self, in_channels: int = 3, d: int = 64):
        super().__init__()
        self.d = d

        # 三个卷积块：通道 3->32->64->128，核 7/5/3，步长 2
        conv_specs = ((in_channels, 32, 7), (32, 64, 5), (64, 128, 3))
        blocks = []
        for cin, cout, kernel in conv_specs:
            blocks += [
                nn.Conv1d(cin, cout, kernel_size=kernel, stride=2),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)     # 任意剩余长度 -> 1 个点
        self.proj = nn.Linear(128, d)           # 128 维特征 -> d 维嵌入

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：x (..., 3, T) -> e (..., d)。

        实现技巧：把前导维度（如 batch、window 位置）全部折叠成一维批量，
        卷积逐个循环独立处理（参数共享），最后再还原形状。
        """
        lead = x.shape[:-2]                     # 前导维度，如 (B, W)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])   # (-1, 3, T)

        h = self.net(x)                         # (-1, 128, L')
        h = self.pool(h).squeeze(-1)            # (-1, 128)
        e = self.proj(h)                        # (-1, d)
        return e.reshape(*lead, self.d)         # (..., d)


def sinusoidal_positional_encoding(n_tokens: int, d: int) -> torch.Tensor:
    """生成 Transformer 论文（Attention Is All You Need）的正弦位置编码。

    公式（i 为维度下标，成对出现）:
        PE[pos, 2i]   = sin(pos / 10000^(2i/d))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
    返回形状 (n_tokens, d) 的 float32 张量。
    """
    pos = torch.arange(n_tokens, dtype=torch.float32).unsqueeze(1)   # (T, 1)
    i = torch.arange(d // 2, dtype=torch.float32).unsqueeze(0)       # (1, d/2)
    angle = pos / torch.pow(10000.0, 2 * i / d)                      # (T, d/2)
    pe = torch.zeros(n_tokens, d)
    pe[:, 0::2] = torch.sin(angle)      # 偶数下标
    pe[:, 1::2] = torch.cos(angle)      # 奇数下标
    return pe


class PatchTSTEncoder(nn.Module):
    """PatchTST 编码器：循环嵌入序列 e -> 潜在退化状态 z(k) ∈ R^d。

    按论文原文:
      - 把 W=30 个循环嵌入切成 patch，长度 P=6、步长 S=3
        -> 共 floor((30-6)/3)+1 = 9 个 patch token；
      - 每个 patch 展平后线性投影回 d 维 token；
      - 加正弦位置编码；
      - 过 L=3 层、N_h=4 头、d_ff=256 的 Transformer 编码器；
      - 自适应平均池化 -> z(k)。

    复用现成组件：nn.TransformerEncoderLayer / nn.TransformerEncoder、
    nn.AdaptiveAvgPool1d；patch 切分与位置编码为手写。
    """

    def __init__(self, d: int = 64, P: int = 6, S: int = 3,
                 L: int = 3, n_head: int = 4, d_ff: int = 256,
                 dropout: float = 0.1, max_tokens: int = 64):
        super().__init__()
        self.P, self.S = P, S

        # 每个 patch 展平成 P*d 维，再投影回 d 维 token
        self.patch_proj = nn.Linear(P * d, d)

        # 正弦位置编码作为"不可训练缓冲区"存进模块（随模型保存/迁移）
        self.register_buffer("pe", sinusoidal_positional_encoding(max_tokens, d))

        # 论文参数：L=3 层、4 头、前馈 256；batch_first 让输入是 (B, T, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=L)

        # 池化在 token 维上：9 个 token -> 1 个向量
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """前向：e (B, W, d) -> z (B, d)。"""
        B, W, d = e.shape

        # 1) patch 切分：unfold 沿第 1 维（循环维）滑窗
        #    (B, W, d) -> (B, n_patches, d, P)
        patches = e.unfold(1, self.P, self.S)
        # 2) 展平每个 patch：d×P -> P*d（unfold 的维度顺序是 d 在前）
        patches = patches.permute(0, 1, 3, 2).reshape(B, -1, self.P * d)
        # 3) 线性投影 -> token，加位置编码（只取实际 token 数）
        tokens = self.patch_proj(patches)                       # (B, n, d)
        tokens = tokens + self.pe[: tokens.shape[1]]
        # 4) Transformer 编码器（自注意力建模 patch 间的时序关系）
        out = self.transformer(tokens)                          # (B, n, d)
        # 5) 池化到单个向量 z(k)
        z = self.pool(out.transpose(1, 2)).squeeze(-1)          # (B, d)
        return z


class DynamicsTransition(nn.Module):
    """动力学转移：z(k+1) = z(k) + MLP([z(k) ∥ u(k)])。

    论文原文：
      "A two-layer MLP with residual connection computes the next latent
       state... The residual connection enforces that z(k+1) departs from
       z(k) by a learned increment, reflecting the physical intuition that
       degradation is a continuous process with bounded per-cycle change.
       Iterating Equation (3) for H steps produces the rollout sequence
       {z(k+1), ..., z(k+H)}."

    实现要点:
      - 输入拼接 [z ∥ u]：z 是 64 维潜在状态，u 是标量 I_mean（C-rate），
        拼成 65 维后过两层 MLP（64+1 -> hidden -> 64）；
      - 残差连接 z + Δz：让模型只学"单步增量"，而不是直接学绝对状态，
        这是"退化是连续过程"这一物理先验的编码方式；
      - rollout：把转移反复迭代 H 次，得到未来潜在状态序列，用于预测
        未来 SOH 轨迹（u 在整个滚动中保持不变，因为快充协议固定）。
    """

    def __init__(self, d: int = 64, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d),
        )
        # 残差分支零初始化：训练开始时 Δz ≈ 0，模型从"状态不变"出发学习
        # 有界增量。若不做这一步，随机初始化的 MLP 在 80 步滚动中会指数爆炸
        # （实测单步增量可达百万级），违背"退化是慢过程"的物理先验。
        # 注意：初始时最后一层权重为 0，导致第一层（mlp.0）暂时收不到梯度
        # （∂Δz/∂W1 = 0），这是零初始化的预期行为；优化开始后自动解除。
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """单步转移：z (B, d) + u (B,) -> z_next (B, d)。"""
        u = u.to(dtype=z.dtype).reshape(z.shape[0], 1)  # 对齐类型并 (B,) -> (B, 1)
        delta = self.mlp(torch.cat([z, u], dim=-1)) # Δz = MLP([z ∥ u])
        return z + delta                            # 残差：z(k+1) = z(k) + Δz

    def rollout(self, z: torch.Tensor, u: torch.Tensor,
                H: int = 80) -> torch.Tensor:
        """滚动 H 步：从 z(k) 出发，迭代产出 {z(k+1), ..., z(k+H)}。

        返回 (B, H, d)；每步都用同一个 action u（快充协议固定不变）。
        """
        states = []
        for _ in range(H):
            z = self.forward(z, u)
            states.append(z)
        return torch.stack(states, dim=1)


class SOHHead(nn.Module):
    """共享 SOH 解码头：z (B, d) -> ŝ (B, 1)，两层 MLP + ReLU。

    论文要求"单一共享头"：当前状态和 rollout 的每个未来状态都用同一个头
    解码，保证所有时间步的 SOH 映射一致（公平地比较不同时刻的状态）。
    """

    def __init__(self, d: int = 64, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (B, d) 或 (B, H, d) -> ŝ (B, 1) 或 (B, H, 1)。"""
        return self.mlp(z)


class WorldModel(nn.Module):
    """完整 World Model：把四个部件组装成一个可训练模型。

    前向（一次调用得到论文的两个输出）:
      X    : (B, W, 3, Tmax)  输入窗口的标准化曲线
      u    : (B,)             最后观测循环的 I_mean（action）
      -> ŝ_cur : (B,)         当前 SOH s(k)
      -> ŝ_fut : (B, H)       未来轨迹 s(k+1)..s(k+H)

    数据流:
      X --CycleEncoder--> e(k) 序列 (B, W, d)
        --PatchTSTEncoder--> z(k) (B, d)  --SOHHead--> ŝ(k)
        --DynamicsTransition.rollout--> z(k+1..k+H) (B, H, d)
                                   --SOHHead（同一个）--> ŝ(k+1..k+H)
    """

    def __init__(self, d: int = 64, P: int = 6, S: int = 3,
                 L: int = 3, n_head: int = 4, d_ff: int = 256,
                 dyn_hidden: int = 128, head_hidden: int = 64,
                 H: int = 80):
        super().__init__()
        self.H = H
        self.cycle_encoder = CycleEncoder(d=d)
        self.patch_encoder = PatchTSTEncoder(d=d, P=P, S=S, L=L,
                                             n_head=n_head, d_ff=d_ff)
        self.transition = DynamicsTransition(d=d, hidden=dyn_hidden)
        self.head = SOHHead(d=d, hidden=head_hidden)

    def forward(self, X: torch.Tensor, u: torch.Tensor,
                H: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """X (B, W, 3, T) + u (B,) -> (ŝ_cur (B,), ŝ_fut (B, H))。"""
        H = self.H if H is None else H

        e = self.cycle_encoder(X)               # (B, W, d)
        z_k = self.patch_encoder(e)             # (B, d)
        s_cur = self.head(z_k).squeeze(-1)      # (B, 1) -> (B,)

        z_roll = self.transition.rollout(z_k, u, H=H)   # (B, H, d)
        s_fut = self.head(z_roll).squeeze(-1)   # (B, H, 1) -> (B, H)
        return s_cur, s_fut


def main() -> None:
    """演示：完整 WorldModel 前向，检查两个输出的形状与数值范围。"""
    import pandas as pd

    torch.manual_seed(0)                        # 固定种子，保证可复现

    # 取一小批训练窗口（含标准化），喂给编码器
    splits = pd.read_parquet(ROOT / "data/processed/splits.parquet")
    train_cells = set(splits.loc[splits["split_by_cell"] == "train", "cell_id"])
    windows = pd.read_parquet(ROOT / "data/processed/matr_windows.parquet")
    labels = pd.read_parquet(ROOT / "data/processed/matr_soh_labels.parquet")
    # 挑两只"充电协议不同"的训练电池，验证 action u 对轨迹的影响
    cells_policy = (windows[windows["cell_id"].isin(train_cells)]
                    [["cell_id", "policy"]].drop_duplicates())
    chosen: list[str] = []
    seen_policies: set[str] = set()
    for cid, pol in cells_policy.values:
        if pol not in seen_policies:
            chosen.append(cid)
            seen_policies.add(pol)
        if len(chosen) == 2:
            break
    sub = (windows[windows["cell_id"].isin(chosen)]
           .groupby("cell_id").head(1))            # 每只电池取 1 个窗口

    normalizer = ChannelNormalizer.load(ROOT / "data/processed/normalizer.json")
    ds = WindowDataset(sub, labels, ROOT / "data/external/matr", normalizer=normalizer)
    s0, s1 = ds[0], ds[1]
    X = torch.stack([torch.from_numpy(s0["X"]), torch.from_numpy(s1["X"])])
    u = torch.tensor([s0["u"], s1["u"]], dtype=torch.float32)

    print(f"输入 X: {tuple(X.shape)}  ({X.shape[1]} 个循环 x {X.shape[2]} 通道 x {X.shape[3]} 点)")
    print(f"action u: {u.tolist()}  C-rate（两只电池充电倍率应不同）")

    model = WorldModel()
    for name, sub in model.named_children():
        print(f"  {name}: {sum(p.numel() for p in sub.parameters()):,} 参数")
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")

    s_cur, s_fut = model(X, u)                  # (B,) + (B, H)
    print(f"当前 SOH s(k): {tuple(s_cur.shape)}  值={s_cur.detach().numpy().round(3)}")
    print(f"未来轨迹 s(k+1..k+80): {tuple(s_fut.shape)}")
    print(f"轨迹范围: [{s_fut.detach().numpy().min():.3f}, "
          f"{s_fut.detach().numpy().max():.3f}]（随机初始化，范围仅供参考）")

    # 合理性检查：初始化时 rollout 恒等（残差零初始化），未来轨迹应平坦
    flat = (s_fut[:, 1:] - s_fut[:, :-1]).abs().max().item()
    print(f"未来轨迹步间变化: {flat:.2e}（应≈0：初始化即恒等滚动，训练后变有界轨迹）")


if __name__ == "__main__":
    main()
