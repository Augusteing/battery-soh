"""partial_soh 的 LSTM 模型。

结构对齐 Scientific Reports 2026 论文 Table 1：

    输入通道        3  (电流 I、电压 V、容量坐标 Q)
    嵌入 W1        Linear(3, 128)
    嵌入 W2        Linear(128, 32)
    LSTM           input=32, hidden=64, cell=64
    电压预测头      Linear(64, 128) -> ReLU -> Linear(128, 1)
    SOH 估计头      Linear(64, 128) -> ReLU -> Linear(128, 1)
    重建头（新增）  Linear(64, 128) -> ReLU -> Linear(128, 1)

关键设计：编码器（嵌入 + LSTM）是共享的，两个任务只换输出头。

    - 预训练：电压头在每一步预测“下一步电压”，得到密集监督；
    - 微调：  把电压头替换成 SOH 头，用最后一个隐藏状态回归标量 SOH。
    - 重建头：用于扩展自监督（掩码电压重建），在预训练阶段使用。
"""

from __future__ import annotations

import torch
from torch import nn


class PartialSohLSTM(nn.Module):
    """共享编码器 + 两个输出头（电压预测 / SOH 回归）。"""

    def __init__(
        self,
        input_dim: int = 3,
        emb_hidden: int = 128,
        emb_out: int = 32,
        hidden: int = 64,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()

        # 输入嵌入：3 -> 128 -> 32。
        # 论文里写的是 W1(128x3) 和 W2(32x128)，这里用 nn.Linear 表达：
        # Linear(in, out) 就是 out x in 的矩阵。
        self.embed = nn.Sequential(
            nn.Linear(input_dim, emb_hidden),
            nn.ReLU(),
            nn.Linear(emb_hidden, emb_out),
        )

        # LSTM：输入是嵌入后的 32 维，隐藏状态 64 维。
        # batch_first=True 表示输入形状为 (batch, seq_len, features)。
        self.lstm = nn.LSTM(
            input_size=emb_out,
            hidden_size=hidden,
            batch_first=True,
        )

        # 两个输出头，结构相同：64 -> 128 -> 1。
        self.voltage_head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        self.soh_head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # 重建头：把“当前时间步的隐藏状态”映射回“当前时间步的电压”。
        # 与电压预测头的区别：
        #   电压头 : 隐藏状态 h_t -> V_{t+1}（预测下一步，预训练主任务）；
        #   重建头 : 隐藏状态 h_t -> V_t    （与输入对齐，用于掩码重建）。
        self.recon_head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把 (B, T, 3) 输入编码成 LSTM 输出和最终状态。

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
        # lstm_out: (B, T, 64) -> 电压头 -> (B, T, 1) -> 去掉最后一维。
        return self.voltage_head(lstm_out).squeeze(-1)

    def soh_predict(self, x: torch.Tensor) -> torch.Tensor:
        """用最后一个隐藏状态回归标量 SOH，返回 (B,)。"""
        _, h_n, _ = self.encode(x)
        # h_n 形状 (1, B, 64)，取最后一行得到 (B, 64)。
        return self.soh_head(h_n[-1]).squeeze(-1)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """重建每个时间步的电压，返回 (B, T)。

        用于掩码电压重建：输入 x 的电压通道被部分遮住，重建头从
        LSTM 每个时间步的隐藏状态里“补出”该时刻的电压。
        """
        lstm_out, _, _ = self.encode(x)
        # lstm_out: (B, T, 64) -> 重建头 -> (B, T, 1) -> 去掉最后一维。
        return self.recon_head(lstm_out).squeeze(-1)


if __name__ == "__main__":
    """冒烟测试：随机输入，检查两个头的前向形状。"""
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(0)
    model = PartialSohLSTM()
    x = torch.randn(4, 101, 3)  # batch=4, seq=101, channels=3

    v = model.voltage_predict(x)
    s = model.soh_predict(x)
    r = model.reconstruct(x)

    print(f"输入 x.shape       : {tuple(x.shape)}")
    print(f"电压预测 v.shape   : {tuple(v.shape)}")
    print(f"SOH 预测 s.shape   : {tuple(s.shape)}")
    print(f"电压重建 r.shape   : {tuple(r.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量           : {n_params:,}")
