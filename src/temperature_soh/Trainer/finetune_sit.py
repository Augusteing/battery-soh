"""SIT few-shot 微调：把 LFP 通用表征适配到目标 LFP 电芯。

背景
----
模型先在 Severson（1.1 Ah 18650 LFP）上预训练 + 微调，学到 LFP 的
通用充电曲线表征。SIT（50 Ah 方形 LFP，环境温 / 40°C 恒温箱）是
"另一种 LFP 电芯"：容量规格、内部材料、工况温度都不同。本脚本用
SIT 的**少量电池**（few-shot）微调模型，再在其余电池上评估，
验证"一个 LFP 模型微调到任意 LFP 电芯"的可行性。

通用微调策略（本脚本实现两种）：
  1. 冻结编码器 + 只训 SOH 头（--freeze-encoder，默认）：目标域
     数据少时最稳，防止过拟合，保留源域通用表征；
  2. 全量低学习率微调（--unfreeze-encoder）：编码器 1e-4、头部 1e-3，
     数据稍多时的常用做法。

SIT 标签口径：SOH(k) = Qc(k) / Qc_max（该电池全寿命最大充电容量）。
理由：SIT 出厂容量普遍高于标称 50 Ah，用 Qc/50 会出现 >1 的标签；
用最大容量作基准则全部 ≤1，且与 Severson 的 Qc/1.1 数值区间兼容。

输入归一化：I -> C-rate（÷50），Q -> SOC（÷50），与 Severson
归一化基线（normalized_3ch.pt）一致。

用法
----
```powershell
# 默认：3 只电池微调（冻结编码器），其余电池测试
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/finetune_sit.py

# 全量低学习率微调对照
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/finetune_sit.py --unfreeze-encoder

# 自定义电池
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/finetune_sit.py --train-cells 001-1,001-2,002-1
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
TR_DIR = Path(__file__).resolve().parent
DL_DIR = ROOT / "src" / "temperature_soh" / "DataLoader"
for d in (TR_DIR, DL_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from dataset import TEMP_CENTER_C, TEMP_SCALE_C  # noqa: E402
from dvdq_features import (  # noqa: E402
    DVDQ_MOMENT_CENTER,
    DVDQ_MOMENT_SCALE,
    N_DVDQ_MOMENTS,
    compute_dvdq_moments,
)
from model import TemperatureSohLSTM  # noqa: E402
from physics_loss import cycle_physics_losses  # noqa: E402
from segments import (  # noqa: E402
    OBSERVED_CAPACITY_PCT,
    PREDICTION_CAPACITY_PCT,
    build_segment_index_for_cycle,
    extract_charge_curve,
    interpolate_segment,
)
from sit_io import (  # noqa: E402
    DEFAULT_SIT_DIR,
    SIT_NOMINAL_CAPACITY_AH,
    cell_temperature_c,
    discover_sit_cells,
    load_cycle_summary,
    read_charge_cycle,
)
from temp_features import (  # noqa: E402
    FEATURE_CENTER,
    FEATURE_SCALE,
    N_FEATURES,
    extract_temp_shape_features,
    neutralize_absolute_features,
)

DEFAULT_PRETRAINED = ROOT / "models" / "temperature_soh" / "normalized_3ch.pt"
DEFAULT_OUT = ROOT / "models" / "temperature_soh" / "sit_fewshot.pt"
DEFAULT_CACHE_DIR = ROOT / "data" / "processed" / "sit_cache"
DEFAULT_MIN_SOH = 0.75

# 老化阶段分桶边界（左闭右开，与 5.3 误差分解/报告口径一致）。
# 逆频率加权：每个桶的权重与该桶样本数成反比，让样本少的
# 健康段（0.95~1.00）和深老化段（0.75~0.80）在损失里不被
# 中间段（0.90~0.95，样本最多）淹没。
BUCKET_EDGES = (0.75, 0.80, 0.85, 0.90, 0.95, 1.001)


# ---------------------------------------------------------------------------
# SIT 片段构建（电池级，预加载到内存）
# ---------------------------------------------------------------------------

def _charge_capacity_series(cell_id: str, data_dir: Path) -> np.ndarray:
    """返回该电池全部循环的充电容量序列（Cycle 升序）。"""
    summary = load_cycle_summary(cell_id, data_dir)
    ch = summary[summary["Type"] == "charge"].sort_values("Cycle")
    return ch["Capacity_Ah"].to_numpy(float)


def build_cell_samples_full(
    cell_id: str, data_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建一只 SIT 电池的全部片段样本，返回 (x, soh, temp_features, cycle_ids)。

    x             : (N, 101, 3) float32，归一化 [I(C-rate), V, Q(SOC)]；
    soh           : (N,) float32，标签 = Qc(k) / Qc_max；
    temp_features : (N, 12) float32，温度曲线形状特征（temp_features.py）；
    cycle_ids     : (N,) int64，每片段所属循环号（物理约束按循环聚合需要）。

    该电池的所有循环、所有合法片段都保留（约 4 万片段/电池），
    由调用方决定哪些电池进训练、哪些进测试。
    """
    summary = load_cycle_summary(cell_id, data_dir)
    cycles = sorted(summary["Cycle"].unique().tolist())
    qc = _charge_capacity_series(cell_id, data_dir)
    qc_max = float(qc.max())
    temperature_c = cell_temperature_c(cell_id, data_dir=data_dir)

    x_list: list[np.ndarray] = []
    soh_list: list[np.ndarray] = []
    temp_list: list[np.ndarray] = []
    cyc_list: list[int] = []
    for cycle_number in cycles:
        cycle = read_charge_cycle(cell_id, cycle_number, data_dir)
        index = build_segment_index_for_cycle(
            cycle,
            cell_id=cell_id,
            cycle_index=int(cycle_number),
            temperature_c=temperature_c,
            nominal_capacity=SIT_NOMINAL_CAPACITY_AH,
            observed_capacity_pct=OBSERVED_CAPACITY_PCT,
            prediction_capacity_pct=PREDICTION_CAPACITY_PCT,
        )
        charge = extract_charge_curve(cycle)
        soh = float(qc[cycles.index(cycle_number)]) / qc_max

        for row in index.itertuples(index=False):
            if not row.is_valid_soh:
                continue
            seg = interpolate_segment(
                charge,
                start_ah=float(row.start_ah),
                end_ah=float(row.end_ah),
                nominal_capacity=SIT_NOMINAL_CAPACITY_AH,
            )
            x = np.stack(
                [
                    seg["I"] / SIT_NOMINAL_CAPACITY_AH,   # C-rate
                    seg["V"],                              # V
                    seg["capacity"] / SIT_NOMINAL_CAPACITY_AH,  # SOC
                ],
                axis=1,
            ).astype(np.float32)
            temp_feat = extract_temp_shape_features(
                seg,
                nominal_capacity=SIT_NOMINAL_CAPACITY_AH,
                fallback_temp_c=temperature_c,
            )
            x_list.append(x)
            soh_list.append(soh)
            temp_list.append(temp_feat)
            cyc_list.append(int(cycle_number))

    if not x_list:
        raise ValueError(f"{cell_id} 没有生成任何片段")
    return (
        np.stack(x_list),
        np.asarray(soh_list, dtype=np.float32),
        np.stack(temp_list).astype(np.float32),
        np.asarray(cyc_list, dtype=np.int64),
    )


def build_cell_samples(
    cell_id: str, data_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """兼容旧接口：只返回 (x, soh)，温度特征/循环号丢弃。"""
    x, y, _, _ = build_cell_samples_full(cell_id, data_dir)
    return x, y


def load_cell_from_cache(
    cell_id: str, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """从 sit_cache 读取一只电池的片段（秒级），无缓存则报错。"""
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缓存不存在: {cache_dir}（先运行 sit_cache.py）")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if cell_id not in set(meta.get("cells", [])):
        raise KeyError(f"{cell_id} 不在缓存中（已缓存: {meta.get('cells')}）")

    cells = np.load(cache_dir / "cell_ids.npy", allow_pickle=True)
    rows = np.flatnonzero(cells == cell_id)
    x = np.load(cache_dir / "X.npy", mmap_mode="r")[rows]
    y = np.load(cache_dir / "y.npy")[rows]
    return np.asarray(x), np.asarray(y)


def load_cell_from_cache_full(
    cell_id: str, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从扩展缓存读取 (x, soh, temp_features, cycle_ids)；缺新文件则报错。"""
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缓存不存在: {cache_dir}（先运行 sit_cache.py）")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if cell_id not in set(meta.get("cells", [])):
        raise KeyError(f"{cell_id} 不在缓存中（已缓存: {meta.get('cells')}）")
    for name in ("temp_features.npy", "cycle_ids.npy"):
        if not (cache_dir / name).exists():
            raise FileNotFoundError(
                f"{cache_dir / name} 不存在，请用新版 sit_cache.py 重建缓存"
            )

    cells = np.load(cache_dir / "cell_ids.npy", allow_pickle=True)
    rows = np.flatnonzero(cells == cell_id)
    x = np.load(cache_dir / "X.npy", mmap_mode="r")[rows]
    y = np.load(cache_dir / "y.npy")[rows]
    temp = np.load(cache_dir / "temp_features.npy", mmap_mode="r")[rows]
    cyc = np.load(cache_dir / "cycle_ids.npy")[rows]
    return np.asarray(x), np.asarray(y), np.asarray(temp), np.asarray(cyc)


def get_cell_samples(
    cell_id: str, data_dir: Path, cache_dir: Path | None
) -> tuple[np.ndarray, np.ndarray]:
    """优先读缓存（秒级），否则从 xlsx 构建（慢）。"""
    if cache_dir is not None:
        try:
            return load_cell_from_cache(cell_id, cache_dir)
        except (FileNotFoundError, KeyError):
            print(f"  [提示] {cell_id} 不在缓存，回退 xlsx 构建（较慢）", flush=True)
    return build_cell_samples(cell_id, data_dir)


def get_cell_samples_full(
    cell_id: str, data_dir: Path, cache_dir: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """完整版：优先扩展缓存，否则回退 xlsx 构建（较慢）。"""
    if cache_dir is not None:
        try:
            return load_cell_from_cache_full(cell_id, cache_dir)
        except (FileNotFoundError, KeyError) as exc:
            print(f"  [提示] {cell_id} 扩展缓存不可用（{exc}），回退 xlsx", flush=True)
    return build_cell_samples_full(cell_id, data_dir)


def filter_by_soh(
    x: np.ndarray, y: np.ndarray, min_soh: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """只保留 SOH >= min_soh 的片段（与赛事/BMS 工作区间对齐）。

    真实电车在 SOH 低于约 0.75~0.8 时就会报警更换，深衰减片段
    既不符合比赛场景，也不在 Severson 训练分布内。默认过滤到
    SOH > 0.75，让训练与评估都在常规老化区进行。
    """
    if min_soh is None:
        return x, y
    mask = y >= min_soh
    return x[mask], y[mask]


def filter_by_soh_full(
    x: np.ndarray,
    y: np.ndarray,
    temp: np.ndarray,
    cyc: np.ndarray,
    min_soh: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """完整版 SOH 过滤：x / y / 温度特征 / 循环号同步裁剪。"""
    if min_soh is None:
        return x, y, temp, cyc
    mask = y >= min_soh
    return x[mask], y[mask], temp[mask], cyc[mask]


def inverse_frequency_weights(
    y: torch.Tensor, edges: tuple[float, ...] = BUCKET_EDGES
) -> torch.Tensor:
    """按老化阶段计算逆频率损失权重（与样本数成反比，平均权重 = 1）。

    动机：SOH 退化数据是长尾不平衡的（健康段/深老化段样本少），
    纯 MSE 会被样本最多的中间段主导，导致模型"向均值回归"——
    健康段低估、深老化段高估（见 5.3 误差分解）。给样本少的桶
    更高的权重，让各老化阶段在损失函数里同等重要。

    权重公式（每个桶权重与其样本数成反比）：
        w_i = n / (K_eff * n_bucket(i))
    其中 n 为总样本数，K_eff 为非空桶数，n_bucket(i) 为样本 i
    所在桶的样本数。可以验证：所有权重之和 = n，即平均权重为 1，
    损失量级与不加权时可比（物理约束权重无需调整）。

    对齐文献：
      - arXiv 2603.10527（我们复现的论文）的 inverse-frequency
        sampling across aging stages（采样版，这里是损失加权版）；
      - Delving into Deep Imbalanced Regression (ICML 2021) 的
        naive inverse (INV) 加权。

    参数
    ----
    y     : (n,) 真实 SOH 标签。
    edges : 分桶边界（左闭右开），默认 BUCKET_EDGES。

    返回
    ----
    (n,) float32 权重向量，与 y 同 device。
    """
    edges_t = torch.tensor(edges, dtype=y.dtype, device=y.device)
    # right=True：左闭右开 [edges[i-1], edges[i])，与误差分解口径一致。
    # bucketize 返回"插入位置"（落在第 i 段返回 i+1），桶号 = 返回值-1。
    # 真实标签恒在 [edges[0], edges[-1]) 内，返回值 ∈ 1..len-1。
    bins = torch.bucketize(y, edges_t, right=True) - 1  # 0..K-2
    counts = torch.bincount(bins, minlength=len(edges) - 1).float()
    n = y.numel()
    k_eff = int((counts > 0).sum().item())
    # 样本只会落在非空桶，counts[bins] >= 1，不会除零。
    w = n / (k_eff * counts[bins])
    return w.float()


def group_samples_by_cycle(
    cyc_key: torch.Tensor,
) -> dict[int, list[int]]:
    """按复合循环键分组，返回 {循环键: [样本索引列表]}。

    同循环（同电池、同循环号）的所有片段——不同容量起点的窗口——
    共享同一个退化状态与同一个 SOH 标签。对比学习用它们构造
    "正样本对"（同一循环的两个片段应映射到相近的表征）。
    """
    ck = cyc_key.cpu().numpy()
    order = np.argsort(ck, kind="stable")          # 按键排序
    sorted_ck = ck[order]
    split = np.flatnonzero(np.diff(sorted_ck) != 0) + 1  # 组边界
    groups: dict[int, list[int]] = {}
    start = 0
    for end in np.concatenate([split, [len(ck)]]):
        groups[int(sorted_ck[start])] = order[start:end].tolist()
        start = int(end)
    return groups


def contrastive_loss(
    z: torch.Tensor, tau: float = 0.1
) -> torch.Tensor:
    """同循环片段对比损失（InfoNCE / NT-Xent 形式）。

    输入 z 形状 (B, d)，约定样本按"正样本对相邻"排列：
    (2i, 2i+1) 是同一循环的两个片段（正对），其余是负样本。

    对每个样本 i：
      L_i = -log( exp(sim(z_i, z_p)/τ) / Σ_{j≠i} exp(sim(z_i, z_j)/τ) )
    其中 sim 是余弦相似度，τ 是温度系数。
    含义：把"与正样本相似"的概率最大化，等价于把同循环片段的
    表征拉近、不同循环片段推远。τ 越小，对相似度差异越敏感。

    直观理解：50 个片段像 50 个证人描述同一事故，对比损失强迫
    模型把这些"零散描述"映射到表征空间里的同一点附近。
    """
    z = F.normalize(z, dim=-1)          # 余弦相似度前的 L2 归一化
    sim = z @ z.T / tau                 # (B, B) 相似度矩阵
    b = z.size(0)
    if b % 2 != 0:
        raise ValueError("对比损失要求 batch 内样本数为偶数（正对相邻）")

    # 正样本索引：样本 2i 与 2i+1 互为对方。
    pos_idx = torch.arange(0, b, 2, device=z.device)
    pairs = torch.stack([pos_idx, pos_idx + 1], dim=1)  # (B/2, 2)

    # 分母：对所有 j≠i 求 log-sum-exp（排除自己，sim[i,i]=1 不能算进去）。
    mask = ~torch.eye(b, dtype=torch.bool, device=z.device)
    log_denom = torch.logsumexp(
        sim.masked_fill(~mask, -float("inf")), dim=1
    )

    # 分子：正样本相似度的对数（双向各算一次再平均）。
    log_num = sim[pairs[:, 0], pairs[:, 1]]
    l_01 = -(log_num - log_denom[pairs[:, 0]])
    l_10 = -(log_num - log_denom[pairs[:, 1]])
    return (l_01 + l_10).mean()


# ---------------------------------------------------------------------------
# 微调
# ---------------------------------------------------------------------------

def finetune(
    model: TemperatureSohLSTM,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    freeze_encoder: bool,
    device: torch.device,
    seed: int = 42,
    temp_train: torch.Tensor | None = None,
    cyc_key_train: torch.Tensor | None = None,
    use_temp_embed: bool = False,
    phys_lambda: float = 0.0,
    batch_size: int = 4096,
    bucket_weight: bool = False,
    contrastive_lambda: float = 0.0,
    contrastive_tau: float = 0.1,
    contrastive_batch_size: int = 512,
) -> None:
    """在 SIT 片段上微调模型（全量数据已在内存，一个 epoch 一次前向）。

    bucket_weight=True 时，数据损失用老化阶段逆频率加权
    （inverse_frequency_weights），物理约束损失不受影响。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 冻结编码器：embed + lstm 不更新，只训 SOH 头。
    if freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith(("embed", "lstm")):
                param.requires_grad_(False)

    # 分层学习率：编码器 1e-4、头部 1e-3（unfreeze 模式）。
    head_params = [p for n, p in model.named_parameters() if not n.startswith(("embed", "lstm"))]
    enc_params = [p for n, p in model.named_parameters() if n.startswith(("embed", "lstm")) and p.requires_grad]
    param_groups = [{"params": head_params, "lr": lr}]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": lr * 0.1})
    optimizer = torch.optim.Adam(param_groups)
    loss_fn = nn.MSELoss()

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    if temp_train is not None:
        temp_train = temp_train.to(device)
    if cyc_key_train is not None:
        cyc_key_train = cyc_key_train.to(device)
    n = len(x_train)
    groups: dict[int, list[int]] = {}
    if contrastive_lambda > 0:
        if cyc_key_train is None:
            raise ValueError("contrastive_lambda>0 时必须提供 cyc_key_train")
        groups = group_samples_by_cycle(cyc_key_train)
        print(f"对比学习: {len(groups):,} 个循环可作正样本组，"
              f"λ={contrastive_lambda} τ={contrastive_tau}", flush=True)
    w_train = inverse_frequency_weights(y_train) if bucket_weight else None
    if w_train is not None:
        edges_t = torch.tensor(BUCKET_EDGES, device=y_train.device)
        counts = torch.bincount(
            torch.bucketize(y_train, edges_t, right=True) - 1,
            minlength=len(BUCKET_EDGES) - 1,
        )
        print(f"老化分桶加权: 各桶样本数 {counts.tolist()}  "
              f"边界 {list(BUCKET_EDGES)}", flush=True)
    print(f"微调样本数: {n:,}  冻结编码器: {freeze_encoder}  "
          f"lr(头)={lr}  温度嵌入: {use_temp_embed}  "
          f"物理约束λ: {phys_lambda}  分桶加权: {bucket_weight}  "
          f"对比λ: {contrastive_lambda}", flush=True)

    for epoch in range(1, epochs + 1):
        # SOH 主任务：全量打乱后线性切 batch（与无对比时完全一致，
        # 保证 30 个 epoch 遍历 30 遍数据，消融对比公平）。
        perm = torch.randperm(n)
        batch_idx_list = [
            perm[s : s + batch_size] for s in range(0, n, batch_size)
        ]
        # 对比学习：为每个 SOH batch 预生成一个"对比子集"（每循环随机
        # 取 2 个片段组成正对）。对比子集只参与对比损失，不参与 SOH。
        cl_idx_list: list[torch.Tensor] = []
        if contrastive_lambda > 0:
            cyc_keys_all = list(groups.keys())
            for _ in batch_idx_list:
                idxs: list[int] = []
                while len(idxs) < contrastive_batch_size:
                    ck = cyc_keys_all[int(torch.randint(len(cyc_keys_all), (1,)))]
                    members = groups[ck]
                    if len(members) >= 2:
                        idxs.extend(
                            np.random.choice(members, 2, replace=False).tolist()
                        )
                cl_idx_list.append(torch.tensor(idxs))
        total, n_batches = 0.0, 0
        total_cl = 0.0
        model.train()
        for k, idx in enumerate(batch_idx_list):
            xb, yb = x_train[idx], y_train[idx]
            tb = temp_train[idx] if use_temp_embed else None
            pred = model.soh_predict(xb, tb)
            if w_train is not None:
                # 加权 MSE：每片段误差平方乘上所属老化阶段的权重。
                loss = torch.mean(w_train[idx] * (pred - yb) ** 2)
            else:
                loss = loss_fn(pred, yb)
            if contrastive_lambda > 0:
                # 对比损失：对比子集里（2i, 2i+1）是同循环正对。
                x_cl = x_train[cl_idx_list[k]]
                _, h_n, c_n = model.encode(x_cl)
                z = torch.cat([h_n[-1], c_n[-1]], dim=-1)  # (B, 128)
                cl_loss = contrastive_loss(z, contrastive_tau)
                loss = loss + contrastive_lambda * cl_loss
                total_cl += float(cl_loss.item()) * len(x_cl)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(idx)
            n_batches += 1

        # 物理约束阶段：按 (电池, 循环) 聚合后计算单调性 + 有界性 +
        # 同循环一致性，再做一次反向更新。
        # 注意：编码器可训练（从头训练）时，全量前向会同时保留整个
        # 训练集的 LSTM 反向图（pred 全部 cat 后统一 backward），6GB
        # 显存放不下，因此只在一个随机子集上计算（minibatch 正则化）。
        phys_info = ""
        if phys_lambda > 0:
            if cyc_key_train is None:
                raise ValueError("phys_lambda>0 时必须提供 cyc_key_train")
            model.train()
            if not freeze_encoder and n > 8192:
                phys_idx = torch.randperm(n, device=device)[:8192]
                x_phys = x_train[phys_idx]
                t_phys = temp_train[phys_idx] if use_temp_embed else None
                c_phys = cyc_key_train[phys_idx]
                phys_n = 8192
            else:
                x_phys = x_train
                t_phys = temp_train
                c_phys = cyc_key_train
                phys_n = n
            preds: list[torch.Tensor] = []
            for start in range(0, phys_n, batch_size):
                xb = x_phys[start : start + batch_size]
                tb = t_phys[start : start + batch_size] if use_temp_embed else None
                preds.append(model.soh_predict(xb, tb))
            pred_all = torch.cat(preds)
            losses = cycle_physics_losses(pred_all, c_phys)
            phys_loss = (
                losses["mono"] + losses["bounds"] + losses["consistency"]
            ) * phys_lambda
            optimizer.zero_grad()
            phys_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            phys_info = (f"  phys mono={losses['mono'].item():.2e} "
                         f"bounds={losses['bounds'].item():.2e} "
                         f"cons={losses['consistency'].item():.2e}")

        print(
            f"  [finetune-sit] epoch {epoch:3d}/{epochs}  "
            f"loss={total / n:.6f}"
            + (f"  cl={total_cl / n_batches:.4f}" if contrastive_lambda > 0 else "")
            + phys_info,
            flush=True,
        )


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: TemperatureSohLSTM,
    cell_ids: list[str],
    data_dir: Path,
    device: torch.device,
    cache_dir: Path | None = None,
    min_soh: float | None = DEFAULT_MIN_SOH,
    use_temp_embed: bool = False,
    relative_only: bool = False,
    dvdq_moments: bool = False,
    save_preds_path: Path | None = None,
) -> dict[str, dict]:
    """在指定电池上评估，返回每只电池 + 汇总统计。

    use_temp_embed=True 时加载温度形状特征并传给模型；
    relative_only=True 时把绝对温度水平维（T_mean/T_start/T_end/T_max/T_min）
        抹成固定中性值（诊断"绝对温度 = 电池身份捷径"的消融实验）；
    dvdq_moments=True 时把 tanh(dV/dQ) 的均值/方差/偏度拼到特征向量尾部；
    save_preds_path 不为空时，额外保存逐片段预测 parquet
    （cell_id, cycle_index, soh_true, soh_pred），供 5.3 图表使用。
    """
    model.eval()
    rows: dict[str, dict] = {}
    all_pred: list[np.ndarray] = []
    all_label: list[np.ndarray] = []
    pred_records: list[dict] = []

    cells = discover_sit_cells(data_dir)
    temp_of = dict(zip(cells["cell_id"], cells["temp_group"]))

    for cell_id in cell_ids:
        if use_temp_embed or save_preds_path is not None:
            x, y, temp, cyc = get_cell_samples_full(cell_id, data_dir, cache_dir)
            x, y, temp, cyc = filter_by_soh_full(x, y, temp, cyc, min_soh)
            if relative_only:
                temp = neutralize_absolute_features(temp)
            if dvdq_moments:
                moments = compute_dvdq_moments(
                    x[:, :, 1], x[:, :, 2]  # V、Q(SOC) 通道
                )
                temp = np.concatenate([temp, moments], axis=-1)
        else:
            x, y = get_cell_samples(cell_id, data_dir, cache_dir)
            x, y = filter_by_soh(x, y, min_soh)
            temp = cyc = None
        if len(y) == 0:
            rows[cell_id] = {
                "temp_group": temp_of.get(cell_id, "?"),
                "n": 0,
                "mae": float("nan"),
                "bias": float("nan"),
                "mae_debiased": float("nan"),
            }
            continue
        # 分批前向，避免整只电池（4 万+ 片段）一次性上 GPU 导致 OOM。
        pred_parts: list[np.ndarray] = []
        for start in range(0, len(x), 4096):
            xb = torch.from_numpy(x[start : start + 4096]).to(device)
            tb = (
                torch.from_numpy(temp[start : start + 4096]).to(device)
                if use_temp_embed else None
            )
            pred_parts.append(model.soh_predict(xb, tb).cpu().numpy())
        pred = np.concatenate(pred_parts)
        err = pred - y
        if save_preds_path is not None:
            for j in range(len(pred)):
                pred_records.append(
                    {
                        "cell_id": cell_id,
                        "cycle_index": int(cyc[j]),
                        "soh_true": float(y[j]),
                        "soh_pred": float(pred[j]),
                    }
                )
        rows[cell_id] = {
            "temp_group": temp_of.get(cell_id, "?"),
            "n": len(y),
            "mae": float(np.abs(err).mean()),
            "bias": float(err.mean()),
            "mae_debiased": float(np.abs(err - err.mean()).mean()),
        }
        all_pred.append(pred)
        all_label.append(y)

    pred = np.concatenate(all_pred)
    label = np.concatenate(all_label)
    err = pred - label
    rows["_all"] = {
        "n": len(label),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "mae_debiased": float(np.abs(err - err.mean()).mean()),
        "pearson": float(np.corrcoef(pred, label)[0, 1]),
    }
    if save_preds_path is not None:
        import pandas as pd

        save_preds_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pred_records).to_parquet(save_preds_path, index=False)
        print(f"逐片段预测已保存: {save_preds_path} "
              f"({len(pred_records):,} 条)", flush=True)
    return rows


def _print_report(rows: dict[str, dict]) -> None:
    """打印逐电池 + 汇总 + 分温度组报告。"""
    print("\n===== 逐电池评估 =====")
    print(f"{'电池':<8}{'温度组':<10}{'n':>8}{'MAE':>10}{'bias':>10}{'去偏MAE':>10}")
    for cell, r in rows.items():
        if cell.startswith("_"):
            continue
        print(
            f"{cell:<8}{r['temp_group']:<10}{r['n']:>8,}"
            f"{r['mae'] * 100:>9.2f}%{r['bias'] * 100:>9.2f}%"
            f"{r['mae_debiased'] * 100:>9.2f}%"
        )
    a = rows["_all"]
    print(f"\n汇总: MAE={a['mae'] * 100:.2f}%  bias={a['bias'] * 100:.2f}%  "
          f"去偏MAE={a['mae_debiased'] * 100:.2f}%  Pearson r={a['pearson']:.4f}  "
          f"n={a['n']:,}")

    # 分温度组
    groups: dict[str, list[float]] = {}
    for cell, r in rows.items():
        if cell.startswith("_"):
            continue
        groups.setdefault(r["temp_group"], []).append(r["mae"])
    if groups:
        print("\n分温度组 MAE（电池均值）:")
        for g, maes in groups.items():
            print(f"  {g:<10}: {np.mean(maes) * 100:.2f}%  ({len(maes)} 只电池)")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--init", choices=("pretrained", "random"), default="pretrained",
                        help="pretrained=从 Severson 预训练权重出发（迁移）；"
                             "random=随机初始化（SIT-only 从头训练对照）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SIT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="SIT 片段缓存目录（先跑 sit_cache.py；存在则秒级读取）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--train-cells",
        default="001-1,001-2,002-1",
        help="few-shot 微调电池（逗号分隔）",
    )
    parser.add_argument("--max-test-cells", type=int, default=None,
                        help="只评估前 N 只测试电池（冒烟测试用）")
    parser.add_argument("--test-cells", default=None,
                        help="显式指定评估电池（逗号分隔，优先于自动选择）")
    parser.add_argument("--min-soh", type=float, default=DEFAULT_MIN_SOH,
                        help="只保留 SOH >= 该值的片段（默认 0.75，对齐赛事/BMS 区间）")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="冻结编码器只训 SOH 头（默认关闭 = 全量低学习率）")
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="全量低学习率微调（编码器 lr = 头的 1/10）")
    parser.add_argument("--use-temp-embed", action="store_true",
                        help="启用温度形状特征嵌入（12 维特征，EDD+FFN）")
    parser.add_argument("--temp-mode", choices=("concat", "film"), default="concat",
                        help="温度嵌入方式：concat=拼接（默认，旧行为）；"
                             "film=条件调制（FiLM，温度生成 γ、β 调制表征，"
                             "推荐配 --relative-only）")
    parser.add_argument("--relative-only", action="store_true",
                        help="温度消融：把绝对温度水平维（T_mean/T_start/T_end/"
                             "T_max/T_min）抹成固定中性值，只保留相对形状特征"
                             "（温差、温升率、位置）。需配合 --use-temp-embed")
    parser.add_argument("--dvdq-moments", action="store_true",
                        help="启用 dV/dQ 矩特征：对片段窗口内 tanh(dV/dQ) 计算"
                             "均值/方差/偏度（3 维），拼到温度形状特征后一起"
                             "送入条件调制特征层。需配合 --use-temp-embed。"
                             "对应导师建议的'相变峰'预处理（S-G deriv=1）")
    parser.add_argument("--discard-head", action="store_true",
                        help="预训练初始化时丢弃 SOH 头（随机头诊断实验用："
                             "隔离'温度特征'与'SOH头初始化'两个因素）")
    parser.add_argument("--phys-lambda", type=float, default=0.0,
                        help="物理约束权重（0 = 关闭；0.1 = 与车辆微调一致）")
    parser.add_argument("--save-preds", type=Path, default=None,
                        help="保存测试集逐片段预测 parquet 的路径")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="训练/物理损失批大小（从头训练需调小避免显存不足）")
    parser.add_argument("--bucket-weight", action="store_true",
                        help="老化阶段逆频率加权：按真实 SOH 分桶，样本少的"
                             "桶（健康段/深老化段）在损失里权重更高，缓解"
                             "'向均值回归'（对齐 2603.10527 的 inverse-frequency）")
    parser.add_argument("--contrastive-lambda", type=float, default=0.0,
                        help="同循环片段对比学习损失权重（0 = 关闭）。"
                             "同一循环不同起点片段共享同一退化状态，"
                             "作为正样本对拉近表征，异循环片段推开")
    parser.add_argument("--contrastive-tau", type=float, default=0.1,
                        help="对比损失温度系数（越小对相似度差异越敏感）")
    parser.add_argument("--contrastive-batch-size", type=int, default=512,
                        help="每个 SOH batch 附加的对比子集大小（偶数，"
                             "每循环 2 个片段为一对）")
    args = parser.parse_args()

    if args.freeze_encoder and args.unfreeze_encoder:
        raise ValueError("--freeze-encoder 与 --unfreeze-encoder 互斥")
    if args.relative_only and not args.use_temp_embed:
        raise ValueError("--relative-only 是温度嵌入的消融，必须配合 "
                         "--use-temp-embed 使用")
    if args.dvdq_moments and not args.use_temp_embed:
        raise ValueError("--dvdq-moments 需要启用 --use-temp-embed"
                         "（矩特征经 FiLM/特征层注入）")
    freeze = args.freeze_encoder or not args.unfreeze_encoder  # 默认冻结
    if args.init == "random" and freeze:
        print("提示: 随机初始化不能冻结编码器，自动切换为全量训练")
        freeze = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 条件特征维度：12 维温度形状特征，可选追加 3 维 dV/dQ 矩。
    feature_dim = N_FEATURES
    feature_center: tuple[float, ...] = FEATURE_CENTER
    feature_scale: tuple[float, ...] = FEATURE_SCALE
    if args.dvdq_moments:
        feature_dim += N_DVDQ_MOMENTS
        feature_center = tuple(FEATURE_CENTER) + DVDQ_MOMENT_CENTER
        feature_scale = tuple(FEATURE_SCALE) + DVDQ_MOMENT_SCALE
    model = TemperatureSohLSTM(
        input_dim=3,
        use_temp_embed=args.use_temp_embed,
        temp_mode=args.temp_mode,
        temp_range=(0.0, 55.0),
        temp_feature_dim=feature_dim,
        temp_feature_center=feature_center,
        temp_feature_scale=feature_scale,
    )
    if args.init == "pretrained":
        if not args.pretrained.exists():
            raise FileNotFoundError(f"找不到预训练模型: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        # 拼接版 SOH 头是 160 维（128+32 温度嵌入），装不下预训练的 128 维
        # 头，必须丢弃重新学；--discard-head 是在无温度下故意丢弃预训练头，
        # 用于诊断"温度模块 vs SOH 头初始化"谁在起作用。
        # FiLM 版调制不改变维度（仍 128 维），SOH 头与预训练完全匹配，
        # 可以直接继承完整预训练权重（含 SOH 头），起点更好。
        drop_head = args.discard_head or (
            args.use_temp_embed and args.temp_mode == "concat"
        )
        if drop_head:
            state = {k: v for k, v in ckpt["model"].items()
                     if not k.startswith("soh_head")}
            missing, unexpected = model.load_state_dict(state, strict=False)
            reason = "温度版头维度不匹配" if args.use_temp_embed else "诊断实验"
            print(f"初始化: 预训练编码器 {args.pretrained} "
                  f"(丢弃 SOH 头[{reason}]，missing={len(missing)})")
        else:
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            print(f"初始化: 预训练权重 {args.pretrained} "
                  f"(温度模块新参数随机初始化，missing={len(missing)})")
    else:
        print("初始化: 随机权重（SIT-only 从头训练对照）")
    model.to(device)
    print(f"设备: {device}  温度嵌入: {args.use_temp_embed}  "
          f"方式: {args.temp_mode}  相对特征: {args.relative_only}  "
          f"dV/dQ矩: {args.dvdq_moments}  物理约束λ: {args.phys_lambda}")

    train_cells = [c.strip() for c in args.train_cells.split(",") if c.strip()]
    all_cells = discover_sit_cells(args.data_dir)["cell_id"].tolist()
    test_cells = [c for c in all_cells if c not in train_cells]
    if args.test_cells is not None:
        test_cells = [c.strip() for c in args.test_cells.split(",") if c.strip()]
    if args.max_test_cells is not None:
        test_cells = test_cells[: args.max_test_cells]
    print(f"微调电池: {train_cells}  测试电池数: {len(test_cells)}")
    if args.cache_dir is not None:
        print(f"缓存目录: {args.cache_dir}")

    # 预加载微调数据（温度/物理需要完整版：含温度特征与循环号）。
    # epochs=0 且无训练电池 = 纯零样本评估，跳过数据构建。
    x_train = y_train = temp_train = cyc_key_train = None
    if args.epochs > 0 or train_cells:
        print("构建 few-shot 微调片段 ...", flush=True)
        t0 = time.perf_counter()
        xs, ys = [], []
        temps, cyc_keys = [], []
        for cell_id in train_cells:
            if args.use_temp_embed or args.phys_lambda > 0 or args.contrastive_lambda > 0:
                x, y, temp, cyc = get_cell_samples_full(
                    cell_id, args.data_dir, args.cache_dir
                )
                x, y, temp, cyc = filter_by_soh_full(
                    x, y, temp, cyc, args.min_soh
                )
                if args.relative_only:
                    temp = neutralize_absolute_features(temp)
                if args.dvdq_moments:
                    moments = compute_dvdq_moments(
                        x[:, :, 1], x[:, :, 2]  # V、Q(SOC) 通道
                    )
                    temp = np.concatenate([temp, moments], axis=-1)
                temps.append(temp)
                # 复合循环键：电池序号 * 1e6 + 循环号，避免跨电池循环号冲突。
                cyc_keys.append(
                    np.full(len(cyc), len(xs), dtype=np.int64) * 1_000_000 + cyc
                )
            else:
                x, y = get_cell_samples(cell_id, args.data_dir, args.cache_dir)
                x, y = filter_by_soh(x, y, args.min_soh)
            xs.append(x)
            ys.append(y)
            print(
                f"  {cell_id}: {len(x):,} 片段 "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        if xs:
            x_train = torch.from_numpy(np.concatenate(xs))
            y_train = torch.from_numpy(np.concatenate(ys))
            temp_train = (
                torch.from_numpy(np.concatenate(temps)) if temps else None
            )
            cyc_key_train = (
                torch.from_numpy(np.concatenate(cyc_keys)) if cyc_keys else None
            )

    if args.epochs > 0 and x_train is not None:
        finetune(
            model, x_train, y_train, args.epochs, args.lr, freeze, device,
            temp_train=temp_train,
            cyc_key_train=cyc_key_train,
            use_temp_embed=args.use_temp_embed,
            phys_lambda=args.phys_lambda,
            batch_size=args.batch_size,
            bucket_weight=args.bucket_weight,
            contrastive_lambda=args.contrastive_lambda,
            contrastive_tau=args.contrastive_tau,
            contrastive_batch_size=args.contrastive_batch_size,
        )
    else:
        print("epochs=0：跳过微调，直接评估预训练模型（零样本对照）")

    rows = evaluate(
        model, test_cells, args.data_dir, device, args.cache_dir,
        min_soh=args.min_soh,
        use_temp_embed=args.use_temp_embed,
        relative_only=args.relative_only,
        dvdq_moments=args.dvdq_moments,
        save_preds_path=args.save_preds,
    )
    _print_report(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.out)
    print(f"\n模型已保存: {args.out}")


if __name__ == "__main__":
    main()
