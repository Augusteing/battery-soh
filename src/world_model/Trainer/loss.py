"""训练目标模块（M3 · 第五步）：论文 §3.3–3.5 的损失函数。

论文口径：
  L = L_data + lambda_phys * L_phys + lambda_EWC * L_EWC      （式 6）
  L_phys = L_mono + L_ir + L_voltage                          （式 12）

本模块只做"损失怎么算"，不负责：
  - 反向传播 / 优化器（交给 trainer.py）；
  - 数据读取（交给 dataset.py）；
  - 模型结构（交给 model.py）。

三个损失的作用（为什么要有它们）：
  1) L_data   —— 监督信号：当前 SOH 和未来 H 步轨迹都要逼近标签；
  2) L_phys   —— 物理约束：轨迹必须单调不增（式 8）、当前 SOH 要和内阻
                 标度律一致（式 9-11）。没有它，模型可能学出"先降后升"
                 这种物理上不可能的退化轨迹；
  3) L_EWC    —— 弹性权重巩固：部署场景下数据分批次到达、旧数据被丢弃，
                 防止模型在学新批次时"灾难性遗忘"旧批次（式 13-14）。
                 论文 §4 明确：主配置（全量数据同时可用）不用 EWC。

设计说明（软件工程）：
  - 每个损失一个纯函数，输入输出都是张量，便于单独调试和消融；
  - WorldModelLoss 负责按论文权重组合并返回明细 dict，方便打印每个
    分量看谁在主导训练；
  - L_ir 有 IR 防护：标签表存在少量 IR=0 的循环，除零必须被挡住；
  - L_voltage 论文只给了文字描述、没有公式，故默认关闭并留占位接口。

用法（演示）：
    python "src/world_model/Trainer/loss.py"
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# 论文给出的常数
# ---------------------------------------------------------------------------
GAMMA = 0.75        # 式 9：R/R0 ~ (1/SOH)^gamma，LFP/A123 的经验指数
EPS_MONO = 0.005    # 式 8：单调性容差（容忍测量噪声）
LAMBDA_PHYS = 0.1   # 式 6：物理损失权重
LAMBDA_EWC = 0.4    # 式 6：EWC 权重（主配置不用，留给分阶段部署实验）


# ---------------------------------------------------------------------------
# 三个独立损失（每个都可以单独调用 / 单独做消融）
# ---------------------------------------------------------------------------
def data_loss(s_cur: torch.Tensor, s_fut: torch.Tensor,
              y_cur: torch.Tensor, y_fut: torch.Tensor) -> torch.Tensor:
    """式 7：数据损失 = 当前 SOH 的 MSE + 未来轨迹的 MSE。

    形状约定（与 model.py 输出一致）：
      s_cur : (B,)          当前 SOH 预测 s_hat(k)
      s_fut : (B, H)        未来轨迹预测 s_hat(k+1..k+H)
      y_cur : (B,)          当前 SOH 标签 s(k)
      y_fut : (B, H)        未来轨迹标签 s(k+1..k+H)
    """
    return torch.nn.functional.mse_loss(s_cur, y_cur) \
        + torch.nn.functional.mse_loss(s_fut, y_fut)


def monotonicity_loss(s_fut: torch.Tensor,
                      eps: float = EPS_MONO) -> torch.Tensor:
    """式 8：单调性损失 —— 预测的未来轨迹不允许回升。

    退化在正常循环下不可逆，因此 s(k+h+1) 应 <= s(k+h)。
    对相邻两步求差：diff = s_{h+1} - s_h；若 diff 超过容差 eps 就罚：
      L_mono = 1/(H-1) * sum_h max(0, diff_h + eps)^2
    用 clamp_min 实现 max(0, .)：负值截断为 0，正值保留。
    """
    diff = s_fut[:, 1:] - s_fut[:, :-1]              # (B, H-1)
    violation = torch.clamp_min(diff + eps, 0.0)     # 只有"回升超容差"才非零
    return (violation ** 2).mean()                   # 对 (B, H-1) 全平均


def resistance_soh_loss(s_cur: torch.Tensor, ir_0: torch.Tensor,
                        ir_k: torch.Tensor,
                        gamma: float = GAMMA) -> torch.Tensor:
    """式 10-11：内阻 -> SOH 一致性损失。

    单粒子模型（SPM）给出内阻与容量的标度律：
      R/R0 ~ (1/SOH)^gamma
    反解出"内阻隐含的 SOH"：
      s_IR = (R0 / R_last)^(1/gamma)
    然后惩罚当前 SOH 预测偏离 s_IR：L_ir = MSE(s_hat(k), s_IR)。

    物理直觉：内阻和容量都反映电极活性物质损失，二者应互相印证。
    只做正则化（权重 0.1），不让它喧宾夺主，因为内阻测量本身有噪声。

    IR 防护：标签表存在少量 IR=0 的循环（测量故障），此时 s_IR 会除零。
    做法：对 ir_0/ir_k 都 > 0 且有限的有效样本才计算损失；若整批都无效
    则返回 0（该 batch 不提供内阻约束）。
    """
    valid = (ir_0 > 0) & (ir_k > 0) \
        & torch.isfinite(ir_0) & torch.isfinite(ir_k)  # (B,) 布尔掩码
    if not valid.any():
        return torch.tensor(0.0, dtype=s_cur.dtype, device=s_cur.device)

    ratio = ir_0[valid] / ir_k[valid]                 # R0/R_last，正常 < 1
    s_ir = torch.pow(ratio, 1.0 / gamma)              # (V,) 内阻隐含 SOH
    return torch.nn.functional.mse_loss(s_cur[valid], s_ir)


def voltage_consistency_loss(s_cur: torch.Tensor) -> torch.Tensor:
    """式 12 中的 L_voltage 占位实现。

    论文原文只有一句："A relative check on terminal voltage consistency
    with observed current and resistance, serving as a structural
    regulariser." —— 没有给出公式。要落地它需要从潜变量重建电压曲线并
    与观测电压比对，属于后续工作，当前返回 0 并默认关闭
    （WorldModelLoss(use_voltage=False)）。
    """
    del s_cur  # 占位：暂不使用输入
    return torch.tensor(0.0)  # 占位；实际使用时会按输入设备创建


# ---------------------------------------------------------------------------
# 组合损失
# ---------------------------------------------------------------------------
class WorldModelLoss:
    """按式 6 组合 L_data + lambda_phys * L_phys。

    不继承 nn.Module 是因为它没有可训练参数；把它设计成普通类，
    好处是 forward 可以返回 dict，一次调用同时拿到总损失和每个分量。
    """

    def __init__(self, lambda_phys: float = LAMBDA_PHYS,
                 lambda_ewc: float = LAMBDA_EWC,
                 use_voltage: bool = False,
                 eps: float = EPS_MONO, gamma: float = GAMMA):
        self.lambda_phys = lambda_phys
        self.lambda_ewc = lambda_ewc     # EWC 由 trainer 按需叠加，这里仅记录
        self.use_voltage = use_voltage
        self.eps = eps
        self.gamma = gamma

    def __call__(self, s_cur: torch.Tensor, s_fut: torch.Tensor,
                 y_cur: torch.Tensor, y_fut: torch.Tensor,
                 ir_0: torch.Tensor, ir_k: torch.Tensor) -> dict:
        """返回 {'total', 'data', 'mono', 'ir', 'voltage', 'phys'}。"""
        l_data = data_loss(s_cur, s_fut, y_cur, y_fut)
        l_mono = monotonicity_loss(s_fut, eps=self.eps)
        l_ir = resistance_soh_loss(s_cur, ir_0, ir_k, gamma=self.gamma)
        l_voltage = voltage_consistency_loss(s_cur) if self.use_voltage \
            else s_cur.new_tensor(0.0)          # 与 s_cur 同设备同 dtype
        l_phys = l_mono + l_ir + l_voltage
        total = l_data + self.lambda_phys * l_phys

        return {
            "total": total,
            "data": l_data,
            "mono": l_mono,
            "ir": l_ir,
            "voltage": l_voltage,
            "phys": l_phys,
        }


# ---------------------------------------------------------------------------
# EWC（弹性权重巩固）—— 主配置不使用，为分阶段部署实验保留接口
# ---------------------------------------------------------------------------
class EWC:
    """式 13-14 的实现。

    场景：训练数据按批次先后到达（部署中不可能一次拿到全部电池）。
    学完第 t 批后：
      1) 估算每个权重对旧任务的重要度 F_i（Fisher 信息对角阵）；
      2) 保存当时的权重 theta*_i；
      3) 学第 t+1 批时，对重要权重的偏离加罚：
         L_EWC = sum_i F_i * (theta_i - theta*_i)^2

    Fisher 时机（论文强调的非标准点）：必须在模型未收敛时算，
    否则梯度趋近 0，F_i ≈ 0，罚项形同虚设。论文在每阶段第 10 个 epoch
    计算 Fisher。
    """

    def __init__(self, model: torch.nn.Module,
                 dataloader, loss_fn: WorldModelLoss,
                 device: torch.device):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        # theta*：Fisher 估算时的权重快照
        self.prev_params = {n: p.detach().clone()
                            for n, p in model.named_parameters()}
        self.fisher: dict[str, torch.Tensor] = {}

    def compute_fisher(self) -> None:
        """式 13：F_i ~= 1/N * sum_n (dL_data(n)/d theta_i)^2。

        遍历若干个 mini-batch，累加 L_data 对每个权重的梯度平方并取平均。
        注意这里只对 L_data 求梯度（论文口径），不含物理项。
        """
        self.model.train()
        acc = {n: torch.zeros_like(p)
               for n, p in self.model.named_parameters()}
        n_batches = 0

        for batch in self.dataloader:
            X = batch["X"].to(self.device)
            u = batch["u"].to(self.device)
            y_cur = batch["y_cur"].to(self.device)
            y_fut = batch["y_fut"].to(self.device)
            ir_0 = batch["ir_0"].to(self.device)
            ir_k = batch["ir_k"].to(self.device)

            self.model.zero_grad()
            s_cur, s_fut = self.model(X, u)
            losses = self.loss_fn(s_cur, s_fut, y_cur, y_fut, ir_0, ir_k)
            losses["data"].backward()          # 只回传数据损失

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    acc[n] += p.grad.detach() ** 2
            n_batches += 1

        for n in acc:
            self.fisher[n] = acc[n] / n_batches

    def penalty(self) -> torch.Tensor:
        """式 14：L_EWC = sum_i F_i * (theta_i - theta*_i)^2。"""
        total = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                total = total + (self.fisher[n]
                                 * (p - self.prev_params[n]) ** 2).sum()
        return total


def ewc_loss(model: torch.nn.Module, fisher: dict,
             prev_params: dict) -> torch.Tensor:
    """函数式 EWC 罚项（不依赖 EWC 类的场景）。"""
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if n in fisher:
            total = total + (fisher[n] * (p - prev_params[n]) ** 2).sum()
    return total


def main() -> None:
    """演示：随机数据走一遍全部损失，检查形状、数值和梯度可达性。"""
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT / "src/world_model/Trainer"))
    from model import WorldModel

    torch.manual_seed(0)
    B, W, T = 4, 30, 1000

    # 随机"假数据"，形状与 dataset.py 的输出一致
    X = torch.randn(B, W, 3, T)
    u = torch.full((B,), 2.5)
    y_cur = torch.tensor([0.99, 0.95, 0.90, 0.85])
    y_fut = torch.linspace(1.0, 0.80, 80).expand(B, -1).clone()
    ir_0 = torch.full((B,), 0.017)
    ir_k = torch.tensor([0.0170, 0.0185, 0.0200, 0.0220])   # 内阻递增 -> 更老

    model = WorldModel()
    s_cur, s_fut = model(X, u)
    print(f"模型输出：s_cur {tuple(s_cur.shape)}，s_fut {tuple(s_fut.shape)}")

    criterion = WorldModelLoss()
    losses = criterion(s_cur, s_fut, y_cur, y_fut, ir_0, ir_k)
    print("损失分量：")
    for k, v in losses.items():
        print(f"  {k:8s} = {float(v.detach()):.4f}")   # detach 后再转标量

    # 梯度可达性：每个子模块至少有一个参数收到非零梯度
    losses["total"].backward()
    for name, sub in model.named_children():
        grads = [p.grad.abs().sum().item() for p in sub.parameters()
                 if p.grad is not None]
        print(f"  {name}: {len(grads)}/{sum(p.numel() for p in sub.parameters())} "
              f"参数量级={max(grads):.3e}" if grads else f"  {name}: 无梯度!")

    # 单调性损失的单元测试：给一个"回升"的轨迹，损失应大于纯下降轨迹
    rising = torch.tensor([[1.0, 0.9, 0.95, 0.88]])          # 中间回升
    falling = torch.tensor([[1.0, 0.9, 0.85, 0.80]])         # 单调下降
    print(f"回升轨迹 L_mono={monotonicity_loss(rising):.4f}  "
          f"下降轨迹 L_mono={monotonicity_loss(falling):.4f}"
          f"（前者应明显更大）")

    # L_ir 防护测试：IR=0 的样本不应报错
    ir_bad = torch.tensor([0.0, 0.017, 0.018, 0.019])
    l_ir_safe = resistance_soh_loss(s_cur, ir_0, ir_bad)
    print(f"IR 含 0 时的 L_ir = {float(l_ir_safe):.4f}（应只对有效样本计算）")


if __name__ == "__main__":
    main()
