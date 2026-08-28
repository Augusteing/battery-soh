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

DEFAULT_PRETRAINED = ROOT / "models" / "temperature_soh" / "normalized_3ch.pt"
DEFAULT_OUT = ROOT / "models" / "temperature_soh" / "sit_fewshot.pt"
DEFAULT_CACHE_DIR = ROOT / "data" / "processed" / "sit_cache"


# ---------------------------------------------------------------------------
# SIT 片段构建（电池级，预加载到内存）
# ---------------------------------------------------------------------------

def _charge_capacity_series(cell_id: str, data_dir: Path) -> np.ndarray:
    """返回该电池全部循环的充电容量序列（Cycle 升序）。"""
    summary = load_cycle_summary(cell_id, data_dir)
    ch = summary[summary["Type"] == "charge"].sort_values("Cycle")
    return ch["Capacity_Ah"].to_numpy(float)


def build_cell_samples(
    cell_id: str, data_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """构建一只 SIT 电池的全部片段样本，返回 (x, soh)。

    x   : (N, 101, 3) float32，归一化 [I(C-rate), V, Q(SOC)]；
    soh : (N,) float32，标签 = Qc(k) / Qc_max。

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
            x_list.append(x)
            soh_list.append(soh)

    if not x_list:
        raise ValueError(f"{cell_id} 没有生成任何片段")
    return np.stack(x_list), np.asarray(soh_list, dtype=np.float32)


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
    n = len(x_train)
    print(f"微调样本数: {n:,}  冻结编码器: {freeze_encoder}  lr(头)={lr}", flush=True)

    for epoch in range(1, epochs + 1):
        # 每个 epoch 打乱顺序，按 batch 切分。
        perm = torch.randperm(n)
        total, n_batches = 0.0, 0
        model.train()
        for start in range(0, n, 4096):
            idx = perm[start : start + 4096]
            xb, yb = x_train[idx], y_train[idx]
            pred = model.soh_predict(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(idx)
            n_batches += 1
        print(
            f"  [finetune-sit] epoch {epoch:3d}/{epochs}  "
            f"loss={total / n:.6f}",
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
) -> dict[str, dict]:
    """在指定电池上评估，返回每只电池 + 汇总统计。"""
    model.eval()
    rows: dict[str, dict] = {}
    all_pred: list[np.ndarray] = []
    all_label: list[np.ndarray] = []

    cells = discover_sit_cells(data_dir)
    temp_of = dict(zip(cells["cell_id"], cells["temp_group"]))

    for cell_id in cell_ids:
        x, y = get_cell_samples(cell_id, data_dir, cache_dir)
        # 分批前向，避免整只电池（4 万+ 片段）一次性上 GPU 导致 OOM。
        pred_parts: list[np.ndarray] = []
        for start in range(0, len(x), 4096):
            xb = torch.from_numpy(x[start : start + 4096]).to(device)
            with torch.no_grad():
                pred_parts.append(model.soh_predict(xb).cpu().numpy())
        pred = np.concatenate(pred_parts)
        err = pred - y
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="冻结编码器只训 SOH 头（默认关闭 = 全量低学习率）")
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="全量低学习率微调（编码器 lr = 头的 1/10）")
    args = parser.parse_args()

    if args.freeze_encoder and args.unfreeze_encoder:
        raise ValueError("--freeze-encoder 与 --unfreeze-encoder 互斥")
    freeze = args.freeze_encoder or not args.unfreeze_encoder  # 默认冻结

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.pretrained.exists():
        raise FileNotFoundError(f"找不到预训练模型: {args.pretrained}")
    model = TemperatureSohLSTM(input_dim=3, use_temp_embed=False)
    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"预训练模型: {args.pretrained}  设备: {device}")

    train_cells = [c.strip() for c in args.train_cells.split(",") if c.strip()]
    all_cells = discover_sit_cells(args.data_dir)["cell_id"].tolist()
    test_cells = [c for c in all_cells if c not in train_cells]
    if args.max_test_cells is not None:
        test_cells = test_cells[: args.max_test_cells]
    print(f"微调电池: {train_cells}  测试电池数: {len(test_cells)}")
    if args.cache_dir is not None:
        print(f"缓存目录: {args.cache_dir}")

    # 预加载微调数据
    print("构建 few-shot 微调片段 ...", flush=True)
    t0 = time.perf_counter()
    xs, ys = [], []
    for cell_id in train_cells:
        x, y = get_cell_samples(cell_id, args.data_dir, args.cache_dir)
        xs.append(x)
        ys.append(y)
        print(f"  {cell_id}: {len(x):,} 片段 ({time.perf_counter() - t0:.0f}s)", flush=True)
    x_train = torch.from_numpy(np.concatenate(xs))
    y_train = torch.from_numpy(np.concatenate(ys))

    finetune(model, x_train, y_train, args.epochs, args.lr, freeze, device)

    rows = evaluate(model, test_cells, args.data_dir, device, args.cache_dir)
    _print_report(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, args.out)
    print(f"\n模型已保存: {args.out}")


if __name__ == "__main__":
    main()
