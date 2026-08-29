"""车辆级（单车时间序列）微调实验。

场景：模型部署到一辆具体车辆后，用这辆车**自己的历史充电片段**
（前 N 个循环）微调，得到"车辆专属"健康模型，再预测这辆车
未来（N 循环之后）的 SOH。对比零样本（不微调）验证车辆个性化
是否带来增益。

数据组织：单车按循环切分——
    train: cycle_index <= split_cycle 的片段（车辆历史）
    test : cycle_index >  split_cycle 的片段（车辆未来）

用法：
```powershell
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/vehicle_finetune.py --cell 001-2 --split-cycle 300
```
"""

from __future__ import annotations

import argparse
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
    load_cycle_summary,
    read_charge_cycle,
)
from temp_features import (  # noqa: E402
    FEATURE_CENTER,
    FEATURE_SCALE,
    extract_temp_shape_features,
)

DEFAULT_PRETRAINED = ROOT / "models" / "temperature_soh" / "normalized_3ch.pt"


def build_cell_with_cycles(
    cell_id: str, data_dir: Path, min_soh: float | None = 0.75
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取单车全部片段，返回 (x, soh, cycle_ids, temp_features)。

    与 finetune_sit.build_cell_samples 一致，但额外记录每个片段的
    循环号（车辆时间序列切分需要）和**片段级温度形状特征**（12 维，
    见 temp_features.extract_temp_shape_features）。
    温度曲线全 NaN 时用电池级代表温度兜底（整段常数温度）。
    """
    summary = load_cycle_summary(cell_id, data_dir)
    cycles = sorted(summary["Cycle"].unique().tolist())
    qc = summary[summary["Type"] == "charge"].sort_values("Cycle")["Capacity_Ah"].to_numpy(float)
    qc_max = float(qc.max())
    temperature_c = cell_temperature_c(cell_id, data_dir=data_dir)

    x_list, y_list, cyc_list, temp_list = [], [], [], []
    for cycle_number in cycles:
        cycle = read_charge_cycle(cell_id, cycle_number, data_dir)
        index = build_segment_index_for_cycle(
            cycle, cell_id=cell_id, cycle_index=int(cycle_number),
            temperature_c=temperature_c,
            nominal_capacity=SIT_NOMINAL_CAPACITY_AH,
            observed_capacity_pct=OBSERVED_CAPACITY_PCT,
            prediction_capacity_pct=PREDICTION_CAPACITY_PCT,
        )
        charge = extract_charge_curve(cycle)
        soh = float(qc[cycles.index(cycle_number)]) / qc_max
        if min_soh is not None and soh < min_soh:
            continue
        for row in index.itertuples(index=False):
            if not row.is_valid_soh:
                continue
            seg = interpolate_segment(
                charge, start_ah=float(row.start_ah), end_ah=float(row.end_ah),
                nominal_capacity=SIT_NOMINAL_CAPACITY_AH,
            )
            x = np.stack(
                [
                    seg["I"] / SIT_NOMINAL_CAPACITY_AH,
                    seg["V"],
                    seg["capacity"] / SIT_NOMINAL_CAPACITY_AH,
                ],
                axis=1,
            ).astype(np.float32)
            x_list.append(x)
            y_list.append(soh)
            cyc_list.append(int(cycle_number))
            # 片段级温度形状特征：同一循环的不同片段（起点不同）温度
            # 曲线不同，因此特征逐片段提取，而不是循环级标量。
            temp_list.append(
                extract_temp_shape_features(
                    seg,
                    nominal_capacity=SIT_NOMINAL_CAPACITY_AH,
                    fallback_temp_c=temperature_c,
                )
            )
    return (
        np.stack(x_list),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(cyc_list, dtype=np.int64),
        np.stack(temp_list).astype(np.float32),
    )


@torch.no_grad()
def evaluate(
    model: TemperatureSohLSTM,
    x: torch.Tensor,
    y: torch.Tensor,
    temp: torch.Tensor | None,
    device: torch.device,
    use_temp_embed: bool,
) -> dict[str, float]:
    """分批评估，返回 MAE/bias/去偏。"""
    preds = []
    for start in range(0, len(x), 4096):
        xb = x[start : start + 4096].to(device)
        tb = temp[start : start + 4096].to(device) if use_temp_embed else None
        preds.append(model.soh_predict(xb, tb).cpu().numpy())
    pred = np.concatenate(preds)
    err = pred - y.numpy()
    return {
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "mae_debiased": float(np.abs(err - err.mean()).mean()),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, help="目标电池（如 001-2）")
    parser.add_argument("--split-cycle", type=int, default=300,
                        help="前 N 个循环做车辆历史（微调），之后做未来（测试）")
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SIT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-soh", type=float, default=0.75)
    parser.add_argument("--use-temp-embed", action="store_true",
                        help="启用温度形状特征嵌入（EDD+FFN，12 维曲线特征）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemperatureSohLSTM(
        input_dim=3,
        use_temp_embed=args.use_temp_embed,
        temp_range=(0.0, 55.0),
        temp_feature_center=FEATURE_CENTER,
        temp_feature_scale=FEATURE_SCALE,
    )
    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    if args.use_temp_embed:
        # 预训练模型是 3ch 无温度版（SOH 头 128 维输入）；温度版 SOH 头
        # 是 160 维输入。只加载编码器（embed + lstm），SOH 头随机初始化。
        state = {k: v for k, v in ckpt["model"].items()
                 if not k.startswith("soh_head")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"温度嵌入版：加载编码器权重（跳过 SOH 头），"
              f"missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"电池 {args.cell}  切分循环 {args.split_cycle}  "
          f"温度嵌入 {args.use_temp_embed}  设备 {device}", flush=True)

    t0 = time.perf_counter()
    x, y, cyc, temp = build_cell_with_cycles(args.cell, args.data_dir, args.min_soh)
    print(f"读取 {len(x):,} 片段 ({time.perf_counter() - t0:.0f}s)", flush=True)
    if args.use_temp_embed:
        print(f"温度形状特征 ({temp.shape[1]} 维): "
              f"T_mean {temp[:, 0].min():.1f}~{temp[:, 0].max():.1f}°C, "
              f"ΔT {temp[:, 6].min():.1f}~{temp[:, 6].max():.1f}°C, "
              f"dT/dSOC_max {temp[:, 8].min():.1f}~{temp[:, 8].max():.1f}")

    train_mask = cyc <= args.split_cycle
    test_mask = cyc > args.split_cycle
    x_train, y_train = torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask])
    x_test, y_test = torch.from_numpy(x[test_mask]), torch.from_numpy(y[test_mask])
    temp_train = torch.from_numpy(temp[train_mask]) if args.use_temp_embed else None
    temp_test = torch.from_numpy(temp[test_mask]) if args.use_temp_embed else None
    print(f"车辆历史（cycle ≤ {args.split_cycle}）: {len(x_train):,} 片段, "
          f"SOH {y_train.min():.3f}~{y_train.max():.3f}")
    print(f"车辆未来（cycle > {args.split_cycle}）: {len(x_test):,} 片段, "
          f"SOH {y_test.min():.3f}~{y_test.max():.3f}")

    # 1) 零样本（不微调）在车辆未来上的表现。
    zero = evaluate(model, x_test, y_test, temp_test, device, args.use_temp_embed)
    print(f"\n零样本（Severson 预训练，无车辆数据）: MAE={zero['mae'] * 100:.2f}%  "
          f"bias={zero['bias'] * 100:.2f}%")

    # 2) 车辆微调：用该车历史片段全量微调。
    torch.manual_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    n = len(x_train)
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        model.train()
        total = 0.0
        for start in range(0, n, 4096):
            idx = perm[start : start + 4096]
            xb = x_train[idx].to(device)
            tb = temp_train[idx].to(device) if args.use_temp_embed else None
            pred = model.soh_predict(xb, tb)
            loss = loss_fn(pred, y_train[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(idx)
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  [vehicle-finetune] epoch {epoch:3d}/{args.epochs}  "
                  f"loss={total / n:.6f}", flush=True)

    finetuned = evaluate(model, x_test, y_test, temp_test, device, args.use_temp_embed)
    print(f"车辆微调后（用该车前 {args.split_cycle} 循环）: "
          f"MAE={finetuned['mae'] * 100:.2f}%  bias={finetuned['bias'] * 100:.2f}%  "
          f"去偏={finetuned['mae_debiased'] * 100:.2f}%")
    print(f"提升: MAE {zero['mae'] * 100:.2f}% → {finetuned['mae'] * 100:.2f}% "
          f"（{- (finetuned['mae'] - zero['mae']) * 100:+.2f}pp）")


if __name__ == "__main__":
    main()
