"""temperature_soh 的 LSTM 模型（4 通道版）。

结构与 partial_soh 的 PartialSohLSTM **完全一致**，唯一区别是
输入通道从 3 变成 4：`(I, V, Q, T)`，其中 T 是归一化温度
`T' = (T - 25) / 10`（归一化在 Dataset 阶段完成，模型不管单位）。

结构（对齐 Scientific Reports 2026 论文 Table 1，仅输入通道扩展）：

    输入通道        4  (电流 I、电压 V、容量坐标 Q、归一化温度 T')
    嵌入 W1        Linear(4, 128)
    嵌入 W2        Linear(128, 32)
    LSTM           input=32, hidden=64, cell=64
    电压预测头      Linear(64, 128) -> ReLU -> Linear(128, 1)
    SOH 估计头      Linear(128, 128) -> ReLU -> Linear(128, 1)  # 输入 = [h; c] 拼接
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


class TemperatureSohLSTM(nn.Module):
    """共享编码器 + 两个输出头（电压预测 / SOH 回归），4 通道输入。"""

    def __init__(
        self,
        input_dim: int = 4,
        emb_hidden: int = 128,
        emb_out: int = 32,
        hidden: int = 64,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()

        # 输入嵌入：4（I, V, Q, T'）-> 128 -> 32。
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

        # SOH 头：输入 [h; c] 拼接（128 维），回归标量 SOH。
        self.soh_head = nn.Sequential(
            nn.Linear(hidden * 2, head_hidden),
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

    def soh_predict(self, x: torch.Tensor) -> torch.Tensor:
        """用最后一个隐藏状态 h 与细胞状态 c 的拼接回归标量 SOH。"""
        _, h_n, c_n = self.encode(x)
        state = torch.cat([h_n[-1], c_n[-1]], dim=-1)  # (B, 128)
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
    x = torch.randn(4, 101, 4)  # batch=4, seq=101, channels=4 (I,V,Q,T')

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
