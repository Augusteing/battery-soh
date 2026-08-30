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

ROOT = Path(__file__).resolve().parents[3]
TR_DIR = Path(__file__).resolve().parent
DL_DIR = ROOT / "src" / "temperature_soh" / "DataLoader"
for d in (TR_DIR, DL_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from dataset import TEMP_CENTER_C, TEMP_SCALE_C  # noqa: E402
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
    extract_temp_shape_features,
)

DEFAULT_PRETRAINED = ROOT / "models" / "temperature_soh" / "normalized_3ch.pt"
DEFAULT_OUT = ROOT / "models" / "temperature_soh" / "sit_fewshot.pt"
DEFAULT_CACHE_DIR = ROOT / "data" / "processed" / "sit_cache"
DEFAULT_MIN_SOH = 0.75


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
) -> None:
    """在 SIT 片段上微调模型（全量数据已在内存，一个 epoch 一次前向）。"""
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
    print(f"微调样本数: {n:,}  冻结编码器: {freeze_encoder}  "
          f"lr(头)={lr}  温度嵌入: {use_temp_embed}  "
          f"物理约束λ: {phys_lambda}", flush=True)

    for epoch in range(1, epochs + 1):
        # 每个 epoch 打乱顺序，按 batch 切分。
        perm = torch.randperm(n)
        total, n_batches = 0.0, 0
        model.train()
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            tb = temp_train[idx] if use_temp_embed else None
            pred = model.soh_predict(xb, tb)
            loss = loss_fn(pred, yb)
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
            f"loss={total / n:.6f}{phys_info}",
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
    save_preds_path: Path | None = None,
) -> dict[str, dict]:
    """在指定电池上评估，返回每只电池 + 汇总统计。

    use_temp_embed=True 时加载温度形状特征并传给模型；
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
    parser.add_argument("--phys-lambda", type=float, default=0.0,
                        help="物理约束权重（0 = 关闭；0.1 = 与车辆微调一致）")
    parser.add_argument("--save-preds", type=Path, default=None,
                        help="保存测试集逐片段预测 parquet 的路径")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="训练/物理损失批大小（从头训练需调小避免显存不足）")
    args = parser.parse_args()

    if args.freeze_encoder and args.unfreeze_encoder:
        raise ValueError("--freeze-encoder 与 --unfreeze-encoder 互斥")
    freeze = args.freeze_encoder or not args.unfreeze_encoder  # 默认冻结
    if args.init == "random" and freeze:
        print("提示: 随机初始化不能冻结编码器，自动切换为全量训练")
        freeze = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemperatureSohLSTM(
        input_dim=3,
        use_temp_embed=args.use_temp_embed,
        temp_range=(0.0, 55.0),
        temp_feature_center=FEATURE_CENTER,
        temp_feature_scale=FEATURE_SCALE,
    )
    if args.init == "pretrained":
        if not args.pretrained.exists():
            raise FileNotFoundError(f"找不到预训练模型: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        if args.use_temp_embed:
            # 预训练模型是 3ch 无温度版（SOH 头 128 维）；温度版 SOH 头
            # 160 维。只加载编码器，SOH 头与温度嵌入随机初始化。
            state = {k: v for k, v in ckpt["model"].items()
                     if not k.startswith("soh_head")}
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"初始化: 预训练编码器 {args.pretrained} "
                  f"(跳过 SOH 头，missing={len(missing)})")
        else:
            model.load_state_dict(ckpt["model"])
            print(f"初始化: 预训练权重 {args.pretrained}")
    else:
        print("初始化: 随机权重（SIT-only 从头训练对照）")
    model.to(device)
    print(f"设备: {device}  温度嵌入: {args.use_temp_embed}  "
          f"物理约束λ: {args.phys_lambda}")

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
            if args.use_temp_embed or args.phys_lambda > 0:
                x, y, temp, cyc = get_cell_samples_full(
                    cell_id, args.data_dir, args.cache_dir
                )
                x, y, temp, cyc = filter_by_soh_full(
                    x, y, temp, cyc, args.min_soh
                )
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
        )
    else:
        print("epochs=0：跳过微调，直接评估预训练模型（零样本对照）")

    rows = evaluate(
        model, test_cells, args.data_dir, device, args.cache_dir,
        min_soh=args.min_soh,
        use_temp_embed=args.use_temp_embed,
        save_preds_path=args.save_preds,
    )
    _print_report(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.out)
    print(f"\n模型已保存: {args.out}")


if __name__ == "__main__":
    main()
