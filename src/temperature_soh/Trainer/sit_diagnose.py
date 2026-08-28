"""SIT 零样本诊断：用 Severson 训练的模型直接评估 SIT 电池。

目的（跨电芯泛化诊断）：
    模型只在 Severson（1.1 Ah 18650，恒温 30°C）上训练，从未见过
    SIT（50 Ah 方形 LFP，环境温 / 40°C 恒温箱）。本脚本用归一化
    输入（I -> C-rate、Q -> SOC）把 SIT 片段喂给模型，看零样本
    跨规格泛化误差有多大。

    这是"跨数据集泛化"卖点的证据：如果 MAE 可控（例如 <3%），
    说明 LFP 归一化曲线表征可跨电芯迁移；若误差很大，则说明
    规格差异不可忽略，需要 SIT 变温微调。

标签口径（与 Severson 训练标签一致）：
    SOH = 该循环充电容量 ÷ 标称容量（Severson: Qc/1.1；SIT: Qc/50）。
    SIT 的充电容量来自 Cycle_Summary 的 charge 行 Capacity_Ah。

用法：

```powershell
# 快速：只测 2 只电池（1 环境温 + 1 恒温箱）
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/sit_diagnose.py --max-cells 2

# 全量：20 只电池（约 1~2 小时）
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/sit_diagnose.py
```
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

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
    START_MAX_PCT,
    START_MIN_PCT,
    START_STEP_PCT,
    STEPS_PER_PCT,
    build_segment_index_for_cycle,
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

DEFAULT_MODEL = ROOT / "models" / "temperature_soh" / "normalized_3ch.pt"


# ---------------------------------------------------------------------------
# 片段读取（带循环级缓存：同一循环的 51 个片段共享一次 xlsx 读取）
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _load_charge_curve(cell_id: str, cycle_number: int, data_dir: str):
    """读取一个充电循环（统一结构），供多个片段复用。"""
    return read_charge_cycle(cell_id, int(cycle_number), Path(data_dir))


def build_normalized_inputs(
    cycle: dict,
    index: "object",
    nominal_capacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """把合法片段插值并归一化，返回 (x 矩阵, 对应标签行号)。"""
    x_list: list[np.ndarray] = []
    valid_rows: list[int] = []
    charge = cycle  # sit_io 已返回充电段统一结构
    for i, row in enumerate(index.itertuples(index=False)):
        if not row.is_valid_soh:
            continue
        seg = interpolate_segment(
            charge,
            start_ah=float(row.start_ah),
            end_ah=float(row.end_ah),
            steps_per_pct=STEPS_PER_PCT,
            nominal_capacity=nominal_capacity,
        )
        x = np.stack(
            [
                seg["I"] / nominal_capacity,                    # C-rate
                seg["V"],                                       # V
                seg["capacity"] / nominal_capacity,             # SOC
                (seg["T"] - TEMP_CENTER_C) / TEMP_SCALE_C,      # T'
            ],
            axis=1,
        ).astype(np.float32)
        x_list.append(x)
        valid_rows.append(i)
    if not x_list:
        return np.zeros((0, 101, 4), dtype=np.float32), np.asarray(valid_rows)
    return np.stack(x_list), np.asarray(valid_rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# 单电池评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cell(
    model: TemperatureSohLSTM,
    cell_id: str,
    device: torch.device,
    data_dir: Path,
    max_cycles: int | None = None,
) -> dict:
    """评估一只 SIT 电池，返回误差统计与样本数。"""
    summary = load_cycle_summary(cell_id, data_dir)
    cycles = sorted(summary["Cycle"].unique().tolist())
    if max_cycles is not None:
        cycles = cycles[:max_cycles]

    temperature_c = cell_temperature_c(cell_id, data_dir=data_dir)
    abs_errors: list[float] = []
    n_samples = 0

    for cycle_number in cycles:
        cycle = _load_charge_curve(cell_id, cycle_number, str(data_dir))
        index = build_segment_index_for_cycle(
            cycle,
            cell_id=cell_id,
            cycle_index=int(cycle_number),
            temperature_c=temperature_c,
            nominal_capacity=SIT_NOMINAL_CAPACITY_AH,
            observed_capacity_pct=OBSERVED_CAPACITY_PCT,
            prediction_capacity_pct=0.07,
        )
        x, valid_rows = build_normalized_inputs(cycle, index, SIT_NOMINAL_CAPACITY_AH)
        if len(valid_rows) == 0:
            continue

        # SOH 标签 = 充电容量 / 标称容量（与 Severson Qc/1.1 同口径）。
        row = summary[summary["Cycle"] == cycle_number]
        charge_cap = float(row.loc[row["Type"] == "charge", "Capacity_Ah"].iloc[0])
        soh_label = charge_cap / SIT_NOMINAL_CAPACITY_AH

        x_t = torch.from_numpy(x).to(device)
        pred = model.soh_predict(x_t)  # (n_valid,)
        abs_errors.append((pred.cpu().numpy() - soh_label))
        n_samples += len(valid_rows)

    if not abs_errors:
        return {"mae": float("nan"), "n": 0}
    errors = np.concatenate(abs_errors)
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(errors.mean()),
        "n": n_samples,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Severson 归一化基线模型权重（默认 normalized_3ch.pt）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SIT_DIR)
    parser.add_argument("--max-cells", type=int, default=None,
                        help="只评估前 N 只电池（快速模式）")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="每只电池只取前 N 个循环（快速模式）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.model.exists():
        raise FileNotFoundError(
            f"找不到模型 {args.model}；请先完成 Severson 归一化基线训练"
        )
    model = TemperatureSohLSTM(input_dim=3, use_temp_embed=False)
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"模型: {args.model}  设备: {device}")

    cells = discover_sit_cells(args.data_dir)
    if args.max_cells is not None:
        # 快速模式：按温度组各取一半，保证两种温度都覆盖。
        ambient = cells[cells["temp_group"] == "ambient"]
        chamber = cells[cells["temp_group"] == "chamber40"]
        n_amb = min(len(ambient), args.max_cells // 2 or 1)
        n_cham = min(len(chamber), args.max_cells - n_amb)
        cells = pd_concat([ambient.head(n_amb), chamber.head(n_cham)])

    print(f"评估 {len(cells)} 只 SIT 电池 ...", flush=True)
    rows: list[dict] = []
    t0 = time.perf_counter()
    for pos, (_, cell_row) in enumerate(cells.iterrows(), start=1):
        cell_id = str(cell_row["cell_id"])
        result = evaluate_cell(
            model, cell_id, device, args.data_dir, max_cycles=args.max_cycles
        )
        rows.append(
            {
                "cell_id": cell_id,
                "temp_group": cell_row["temp_group"],
                **result,
            }
        )
        print(
            f"  [{pos}/{len(cells)}] {cell_id:<8} "
            f"{cell_row['temp_group']:<10} MAE={result['mae']:.4f} "
            f"n={result['n']:,}  ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    import pandas as pd

    report = pd.DataFrame(rows)
    print("\n===== 诊断汇总 =====")
    print("按温度组:")
    print(report.groupby("temp_group")["mae"].agg(["mean", "count"]).to_string())
    print("\n全部电池 MAE: %.4f (RMSE %.4f)" % (
        report["mae"].mean(),
        float(np.sqrt(np.mean(report["mae"] ** 2))),
    ))
    print("\n逐电池明细:")
    print(report.to_string(index=False))


def pd_concat(frames) -> "object":
    """轻量拼接（避免顶部 import pandas 拖慢启动）。"""
    import pandas as pd

    return pd.concat([f for f in frames if len(f) > 0], ignore_index=True)


if __name__ == "__main__":
    main()
