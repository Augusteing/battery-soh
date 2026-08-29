"""temperature_soh 的 LSTM 模型（支持循环级温度嵌入）。

与 partial_soh 的 PartialSohLSTM 结构一致，在此基础上扩展了
**循环级温度嵌入**（默认关闭，`use_temp_embed=True` 时启用）：

    - 输入通道可选 3 或 4：`(I, V, Q)` 或 `(I, V, Q, T')`；
    - 温度嵌入升级为**曲线形状特征向量**（见 DataLoader/temp_features.py）：
      SIT 温度传感器量化到 0.1°C，均值标量会丢掉"温度怎么变化"的信息，
      因此把 101 点温度曲线压缩成 12 维形状特征（T_mean / ΔT / dT/dSOC
      峰谷 / 峰值位置等），再走两条路（对齐 arXiv 2504.00393 的
      EDD + FFN 双通道思想）——
        EDD 离散化查表：用 T_mean 归一化后离散成温度档位，表达
                        "环境 25°C / 恒温箱 40°C"这类分段规律；
        FFN 连续变换：  吃完整 12 维特征向量，保留形状细节；
      拼接成 T_num = concat(EDD(T_mean), FFN(features))，再拼到 SOH 头。

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
    """温度曲线形状特征嵌入（EDD + FFN 双通道，对齐 arXiv 2504.00393 思想）。

    设计背景：
      旧版输入是"循环级温度标量"（均值），在变温数据上丢失了形状信息。
      新版输入是 DataLoader/temp_features.py 提取的 12 维形状特征向量：
        T_mean / T_start / T_end / T_max / T_min / T_range / ΔT /
        slope_T_vs_soc / dTdSOC_max / dTdSOC_min / pos_dTdSOC_max / pos_T_max。

    为什么双通道：
      - EDD 离散查表：只用 T_mean（特征 0）归一化后离散成 N_T 个温度档位，
        表达"环境温度 ~25-35°C / 恒温箱 40°C"这种分段式环境标签；
      - FFN 连续路径：吃完整 12 维标准化特征，保留温差、温升率、峰值位置
        等形状细节，提供连续插值能力。
      拼接后同时具备"档位记忆"与"形状感知"。

    输入：特征向量（物理单位，°C / °C/SOC / 位置），形状 (..., F)；
    输出：温度嵌入向量，形状 (..., emb_dim * 2)。
    """

    def __init__(
        self,
        emb_dim: int = 16,
        n_bins: int = 16,
        t_min: float = 0.0,
        t_max: float = 55.0,
        feature_dim: int = 12,
        feature_center: tuple[float, ...] | None = None,
        feature_scale: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.n_bins = n_bins
        self.t_min = t_min
        self.t_max = t_max

        # 每特征的归一化常数（物理定标，非统计量纲），与 temp_features.py
        # 的 FEATURE_CENTER / FEATURE_SCALE 一致。调用方可显式传入。
        if feature_center is None:
            feature_center = (25.0,) * 5 + (0.0,) * 5 + (0.5, 0.5)
        if feature_scale is None:
            feature_scale = (10.0,) * 5 + (3.0, 3.0) + (8.0,) * 3 + (0.25, 0.25)
        if len(feature_center) != feature_dim or len(feature_scale) != feature_dim:
            raise ValueError(
                f"feature_center/scale 长度必须等于 feature_dim={feature_dim}，"
                f"实际 {len(feature_center)} / {len(feature_scale)}"
            )
        self.register_buffer(
            "feature_center", torch.tensor(feature_center, dtype=torch.float32)
        )
        self.register_buffer(
            "feature_scale", torch.tensor(feature_scale, dtype=torch.float32)
        )

        # EDD：温度水平档位（用特征 0 = T_mean）-> 可学习向量。
        self.edd = nn.Embedding(n_bins, emb_dim)

        # FFN：完整特征向量（标准化后）-> 同维度向量（与 EDD 输出可拼接）。
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    @property
    def out_dim(self) -> int:
        """温度嵌入输出维度 = EDD 与 FFN 两路拼接。"""
        return self.emb_dim * 2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """把 (..., F) 温度形状特征编码成 (..., emb_dim*2) 向量。"""
        t_mean = features[..., 0]  # 特征 0 = T_mean（°C）
        t_norm = ((t_mean - self.t_min) / (self.t_max - self.t_min)).clamp(0.0, 1.0)
        # 离散档位索引：T'=1 时落到最后一个 bin（n_bins-1），不越界。
        bin_idx = torch.floor(t_norm * self.n_bins).long().clamp(0, self.n_bins - 1)
        e_edd = self.edd(bin_idx)  # (..., emb_dim)

        # 标准化：减去物理零点、除以物理尺度，让 FFN 输入各维同量级。
        z = (features - self.feature_center) / self.feature_scale
        e_ffn = self.ffn(z)  # (..., emb_dim)
        return torch.cat([e_edd, e_ffn], dim=-1)


class TemperatureSohLSTM(nn.Module):
    """共享编码器 + 输出头（电压预测 / SOH 回归），支持温度嵌入。

    参数
    ----
    input_dim          : 输入通道数（3=I,V,Q；4=再加归一化温度 T'）。
    use_temp_embed     : 是否启用温度形状特征嵌入（默认关闭，保持旧基线结构）。
    temp_emb_dim       : 温度嵌入单路维度（EDD/FFN 各 temp_emb_dim，拼接后 2 倍）。
    temp_bins          : EDD 离散档位数 N_T（T_mean 分档）。
    temp_range         : T_mean 的离散化范围 (T_min, T_max)，°C。
    temp_feature_dim   : 温度形状特征维度（temp_features.py 的 N_FEATURES=12）。
    temp_feature_center/scale: 每特征的物理定标常数（与 temp_features.py 一致）。
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
        temp_range: tuple[float, float] = (0.0, 55.0),
        temp_feature_dim: int = 12,
        temp_feature_center: tuple[float, ...] | None = None,
        temp_feature_scale: tuple[float, ...] | None = None,
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
                feature_dim=temp_feature_dim,
                feature_center=temp_feature_center,
                feature_scale=temp_feature_scale,
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
        self, x: torch.Tensor, temp_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """用 [h; c]（可选 + 温度嵌入）回归标量 SOH。

        参数
        ----
        x             : (B, T, input_dim) 电学曲线输入。
        temp_features : (B, F) 温度形状特征（物理单位），仅
                        use_temp_embed=True 时必需，否则会被忽略。
        """
        _, h_n, c_n = self.encode(x)
        state = torch.cat([h_n[-1], c_n[-1]], dim=-1)  # (B, 128)
        if self.use_temp_embed:
            if temp_features is None:
                raise ValueError("use_temp_embed=True 时必须传入 temp_features")
            t_emb = self.temperature_embed(temp_features)  # (B, 32)
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

    # 温度嵌入冒烟测试：3 通道编码 + 12 维温度形状特征。
    model_t = TemperatureSohLSTM(input_dim=3, use_temp_embed=True)
    temp_feats = torch.randn(4, 12)
    s_t = model_t.soh_predict(x, temp_feats)
    print(f"温度嵌入版 SOH     : {tuple(s_t.shape)}")
    print(f"温度嵌入参数量     : {sum(p.numel() for p in model_t.temperature_embed.parameters()):,}")
    print(f"温度嵌入版总参数量 : {sum(p.numel() for p in model_t.parameters()):,}")
