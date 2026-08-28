"""temperature_soh 的 LSTM 模型（支持循环级温度嵌入）。

与 partial_soh 的 PartialSohLSTM 结构一致，在此基础上扩展了
**循环级温度嵌入**（默认关闭，`use_temp_embed=True` 时启用）：

    - 输入通道可选 3 或 4：`(I, V, Q)` 或 `(I, V, Q, T')`；
    - 温度嵌入对齐 arXiv 2504.00393（钠离子电池多温度老化）：
      循环级温度标量 T 归一化到 [0,1] 后，走两条路——
        EDD 离散化查表：Embedding(floor(T' * N_T))；
        FFN 连续变换：  小前馈网络；
      拼接成 T_num = concat(EDD(T), FFN(T))，再拼到 SOH 头输入里。

设计原则：编码器（嵌入 + LSTM）只学电学曲线形态；温度是"决策层
条件"，只在回归 SOH 时参与，因此预训练（电压预测）完全不受影响。

结构（对齐 Scientific Reports 2026 论文 Table 1，仅输入通道扩展）：

    输入通道        3/4 (电流 I、电压 V、容量坐标 Q，可选 T')
    嵌入 W1        Linear(4, 128)
    嵌入 W2        Linear(128, 32)
    LSTM           input=32, hidden=64, cell=64
    电压预测头      Linear(64, 128) -> ReLU -> Linear(128, 1)
    SOH 估计头      Linear(128(+温度嵌入), 128) -> ReLU -> Linear(128, 1)
    重建头（扩展）  Linear(64, 128) -> ReLU -> Linear(128, 1)
    未来窗头（扩展）Linear(128, 128) -> ReLU -> Linear(128, 36)  # 未来 7% 容量窗电压

关键设计（与 partial_soh 相同）：编码器（嵌入 + LSTM）共享，
两个任务只换输出头。

    - 预训练：电压头在每一步预测“下一步电压”（密集监督）
              + 未来窗头从最终状态预测 36 点电压曲线；
    - 微调：  把输出头换成 SOH 头，用 [h; c] 拼接回归标量 SOH。
"""

from __future__ import annotations

import torch
from torch import nn


class TemperatureEmbedding(nn.Module):
    """循环级温度标量嵌入（对齐 arXiv 2504.00393 的 EDD + FFN 双通道做法）。

    论文原式：

        T'   = (T - T_min) / (T_max - T_min)      # 归一化到 [0,1]
        EDD  = Embedding(floor(T' * N_T))          # 离散档位查表
        T_num = concat(EDD(T); FFN(T))             # 与连续路径拼接

    为什么双通道：
      - 离散查表给模型"档位式"温度记忆（35°C 就是一个明确的类别），
        适合表达"不同温度区间对应不同老化速率"这种分段规律；
      - 连续 FFN 提供插值能力，33.7°C 也能平滑推断，不会因为
        落在两个档位之间就失去精度。

    输入：循环级温度标量（摄氏度，float），形状 (B,) 或标量；
    输出：温度嵌入向量，形状 (B, emb_dim * 2)。
    """

    def __init__(
        self,
        emb_dim: int = 16,
        n_bins: int = 16,
        t_min: float = 0.0,
        t_max: float = 45.0,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.n_bins = n_bins
        self.t_min = t_min
        self.t_max = t_max

        # EDD：离散温度档位 -> 可学习向量。索引范围 [0, n_bins)。
        self.edd = nn.Embedding(n_bins, emb_dim)

        # FFN：归一化温度标量 -> 同维度向量（与 EDD 输出可拼接）。
        self.ffn = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    @property
    def out_dim(self) -> int:
        """温度嵌入输出维度 = EDD 与 FFN 两路拼接。"""
        return self.emb_dim * 2

    def forward(self, t_celsius: torch.Tensor) -> torch.Tensor:
        """把循环级温度标量编码成 (..., emb_dim*2) 向量。"""
        t_norm = ((t_celsius - self.t_min) / (self.t_max - self.t_min)).clamp(0.0, 1.0)
        # 离散档位索引：T'=1 时落到最后一个 bin（n_bins-1），不越界。
        bin_idx = torch.floor(t_norm * self.n_bins).long().clamp(0, self.n_bins - 1)
        e_edd = self.edd(bin_idx)              # (..., emb_dim)
        e_ffn = self.ffn(t_norm.unsqueeze(-1))  # (..., emb_dim)
        return torch.cat([e_edd, e_ffn], dim=-1)


class TemperatureSohLSTM(nn.Module):
    """共享编码器 + 输出头（电压预测 / SOH 回归），支持温度嵌入。

    参数
    ----
    input_dim      : 输入通道数（3=I,V,Q；4=再加归一化温度 T'）。
    use_temp_embed : 是否启用循环级温度嵌入（默认关闭，保持旧基线结构）。
    temp_emb_dim   : 温度嵌入单路维度（EDD/FFN 各 temp_emb_dim，拼接后 2 倍）。
    temp_bins      : EDD 离散档位数 N_T。
    temp_range     : 温度归一化范围 (T_min, T_max)，默认 (0, 45)，对齐论文。
    """

    def __init__(
        self,
        input_dim: int = 3,
        emb_hidden: int = 128,
        emb_out: int = 32,
        hidden: int = 64,
        head_hidden: int = 128,
        use_temp_embed: bool = False,
        temp_emb_dim: int = 16,
        temp_bins: int = 16,
        temp_range: tuple[float, float] = (0.0, 45.0),
    ) -> None:
        super().__init__()
        self.use_temp_embed = use_temp_embed

        # 输入嵌入：input_dim（I, V, Q[, T']）-> 128 -> 32。
        self.embed = nn.Sequential(
            nn.Linear(input_dim, emb_hidden),
            nn.ReLU(),
            nn.Linear(emb_hidden, emb_out),
        )

        # LSTM：输入 32 维嵌入，隐藏状态 64 维。
        self.lstm = nn.LSTM(
            input_size=emb_out,
            hidden_size=hidden,
            batch_first=True,
        )

        # 电压头：每个时间步的隐藏状态 h_t -> 下一步电压 V_{t+1}。
        self.voltage_head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # SOH 头输入 = [h; c] 拼接（128 维），可选再拼温度嵌入向量。
        soh_in_dim = hidden * 2
        if use_temp_embed:
            self.temperature_embed = TemperatureEmbedding(
                emb_dim=temp_emb_dim,
                n_bins=temp_bins,
                t_min=temp_range[0],
                t_max=temp_range[1],
            )
            soh_in_dim += self.temperature_embed.out_dim
        self.soh_head = nn.Sequential(
            nn.Linear(soh_in_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # 重建头：h_t -> V_t（与输入对齐，用于掩码电压重建扩展任务）。
        self.recon_head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # 未来窗头：从最终状态 [h; c] 一次预测未来 7% 容量窗的 36 点电压。
        self.future_head = nn.Sequential(
            nn.Linear(hidden * 2, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 36),
        )

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把 (B, T, 4) 输入编码成 LSTM 输出和最终状态。

        返回:
            lstm_out : (B, T, hidden)，每个时间步的隐藏状态；
            h_n      : (1, B, hidden)，最后一个时间步的隐藏状态；
            c_n      : (1, B, hidden)，最后一个时间步的细胞状态。
        """
        emb = self.embed(x)  # (B, T, 32)
        lstm_out, (h_n, c_n) = self.lstm(emb)
        return lstm_out, h_n, c_n

    def voltage_predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测每个时间步的下一步电压，返回 (B, T)。"""
        lstm_out, _, _ = self.encode(x)
        return self.voltage_head(lstm_out).squeeze(-1)

    def soh_predict(
        self, x: torch.Tensor, temp_celsius: torch.Tensor | None = None
    ) -> torch.Tensor:
        """用 [h; c]（可选 + 温度嵌入）回归标量 SOH。

        参数
        ----
        x            : (B, T, input_dim) 电学曲线输入。
        temp_celsius : (B,) 循环级温度标量（摄氏度）。仅
                       use_temp_embed=True 时必需，否则会被忽略。
        """
        _, h_n, c_n = self.encode(x)
        state = torch.cat([h_n[-1], c_n[-1]], dim=-1)  # (B, 128)
        if self.use_temp_embed:
            if temp_celsius is None:
                raise ValueError("use_temp_embed=True 时必须传入 temp_celsius")
            t_emb = self.temperature_embed(temp_celsius)  # (B, 32)
            state = torch.cat([state, t_emb], dim=-1)
        return self.soh_head(state).squeeze(-1)

    def future_predict(self, x: torch.Tensor) -> torch.Tensor:
        """从最终状态 [h; c] 一次预测未来 7% 容量窗的电压曲线 (B, 36)。"""
        _, h_n, c_n = self.encode(x)
        state = torch.cat([h_n[-1], c_n[-1]], dim=-1)  # (B, 128)
        return self.future_head(state)

    def voltage_and_future(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """一次编码同时输出“下一步电压”和“未来窗电压”（省一次前向）。"""
        lstm_out, h_n, c_n = self.encode(x)
        voltage_pred = self.voltage_head(lstm_out).squeeze(-1)  # (B, T)
        state = torch.cat([h_n[-1], c_n[-1]], dim=-1)  # (B, 128)
        future_pred = self.future_head(state)  # (B, 36)
        return voltage_pred, future_pred

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """重建每个时间步的电压，返回 (B, T)。用于掩码电压重建。"""
        lstm_out, _, _ = self.encode(x)
        return self.recon_head(lstm_out).squeeze(-1)


if __name__ == "__main__":
    """冒烟测试：随机输入，检查两个头的前向形状。"""
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(0)
    model = TemperatureSohLSTM()
    x = torch.randn(4, 101, 3)  # batch=4, seq=101, channels=3 (I,V,Q)

    v = model.voltage_predict(x)
    s = model.soh_predict(x)
    r = model.reconstruct(x)
    f = model.future_predict(x)

    print(f"输入 x.shape       : {tuple(x.shape)}")
    print(f"电压预测 v.shape   : {tuple(v.shape)}")
    print(f"SOH 预测 s.shape   : {tuple(s.shape)}")
    print(f"电压重建 r.shape   : {tuple(r.shape)}")
    print(f"未来窗 f.shape     : {tuple(f.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量           : {n_params:,}")

    # 温度嵌入冒烟测试：3 通道编码 + 循环级温度标量。
    model_t = TemperatureSohLSTM(input_dim=3, use_temp_embed=True)
    temp = torch.tensor([30.0, 35.0, 20.0, 45.0])
    s_t = model_t.soh_predict(x, temp)
    print(f"温度嵌入版 SOH     : {tuple(s_t.shape)}")
    print(f"温度嵌入参数量     : {sum(p.numel() for p in model_t.temperature_embed.parameters()):,}")
    print(f"温度嵌入版总参数量 : {sum(p.numel() for p in model_t.parameters()):,}")
