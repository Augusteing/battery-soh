"""部分充电片段构建模块。

本模块负责把一条充电曲线切成论文描述的短片段。

论文口径：

- 观测窗口长度为 20% 额定容量；
- 起点从 0% 额定容量移动到 50% 额定容量，每次移动 1%；
- 每 1% 额定容量区间再插值为 5 个等距点；
- 电压预训练时，后续预测窗口长度为 7% 额定容量。

注意：本模块只生成“片段索引”和“插值数组”，不写最终训练缓存，
也不计算 SOH 标签。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 把本目录放进 sys.path，便于像 scripts 一样直接运行。
DL_DIR = Path(__file__).resolve().parent
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from charge import extract_charge_curve  # noqa: E402
from mat_io import discover_batch_files, load_raw_cycle  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]

# 论文复现的默认参数
NOMINAL_CAPACITY_AH = 1.1
OBSERVED_CAPACITY_PCT = 0.20   # 观测窗口长度占额定容量的比例
START_MIN_PCT = 0.00           # 起点最小值
START_MAX_PCT = 0.50           # 起点最大值
START_STEP_PCT = 0.01          # 起点每次移动 1% 额定容量
STEPS_PER_PCT = 5              # 每 1% 额定容量插值成 5 步
PREDICTION_CAPACITY_PCT = 0.07  # 电压预测窗口长度占额定容量的比例


def capacity_grid(
    start_ah: float,
    end_ah: float,
    steps_per_pct: int = STEPS_PER_PCT,
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
) -> np.ndarray:
    """生成一条从 start_ah 到 end_ah 的等距容量网格。

    例如 20% 额定容量窗口，每 1% 区间 5 步，则：
      - 1% 额定容量 = 0.011 Ah；
      - 窗口总长度 = 20 * 0.011 = 0.22 Ah；
      - 步数 = 20 * 5 = 100；
      - 网格点数 = 100 + 1 = 101。
    """
    if steps_per_pct < 1:
        raise ValueError(f"steps_per_pct 必须为正整数，得到 {steps_per_pct}")
    if start_ah > end_ah:
        raise ValueError(f"start_ah={start_ah} 不能大于 end_ah={end_ah}")

    one_percent = 0.01 * nominal_capacity
    n_intervals = int(round((end_ah - start_ah) / one_percent * steps_per_pct))
    if n_intervals < 1:
        raise ValueError(f"窗口太短，无法生成有效网格: {start_ah} ~ {end_ah}")
    return np.linspace(start_ah, end_ah, n_intervals + 1)


def interpolate_segment(
    charge: dict[str, np.ndarray],
    start_ah: float,
    end_ah: float,
    steps_per_pct: int = STEPS_PER_PCT,
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
) -> dict[str, np.ndarray]:
    """把充电曲线插值到指定容量窗口的等距网格上。

    参数
    ----
    charge : extract_charge_curve 返回的充电阶段字典。
    start_ah / end_ah : 片段起点和终点，单位 Ah。

    返回
    ----
    dict:
      - capacity : 等距容量网格，单位 Ah；
      - t / V / I / T : 在容量网格上插值后的曲线。
    """
    q = np.asarray(charge["Qc"], dtype=float)
    if q.size < 2:
        raise ValueError("充电阶段点数不足，无法插值")
    # 充电容量 Qc 的起点可能不是精确 0，而是 3e-6 Ah 这类极小值。
    # 因此用很小的容差判断，并把插值端点夹回真实数据范围。
    # 数据里 Qc 起点约为 3.6e-6 Ah，而不是精确 0；用 0.1 mAh 的容差。
    tolerance = 1e-4
    if start_ah < q.min() - tolerance or end_ah > q.max() + tolerance:
        raise ValueError(
            f"片段 [{start_ah:.4f}, {end_ah:.4f}] Ah 超出充电容量范围 "
            f"[{q.min():.4f}, {q.max():.4f}] Ah"
        )

    start_clipped = float(max(start_ah, q.min()))
    end_clipped = float(min(end_ah, q.max()))
    grid = capacity_grid(start_clipped, end_clipped, steps_per_pct, nominal_capacity)
    out: dict[str, np.ndarray] = {"capacity": grid}

    # 充电曲线必须按 Qc 升序排列，否则 np.interp 会得到错误结果。
    order = np.argsort(q)
    q_sorted = q[order]
    for name in ("t", "V", "I", "T"):
        arr = np.asarray(charge[name], dtype=float)[order]
        out[name] = np.interp(grid, q_sorted, arr)

    return out


def start_positions_ah(
    start_min_pct: float = START_MIN_PCT,
    start_max_pct: float = START_MAX_PCT,
    start_step_pct: float = START_STEP_PCT,
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
) -> np.ndarray:
    """返回所有片段起点，单位 Ah。"""
    n = int(round((start_max_pct - start_min_pct) / start_step_pct)) + 1
    starts_pct = np.linspace(start_min_pct, start_max_pct, n)
    return starts_pct * nominal_capacity


def build_segment_index_for_cycle(
    raw: dict[str, np.ndarray],
    cell_id: str,
    cycle_index: int,
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
    observed_capacity_pct: float = OBSERVED_CAPACITY_PCT,
    prediction_capacity_pct: float = PREDICTION_CAPACITY_PCT,
) -> pd.DataFrame:
    """为一个循环生成所有合法片段的索引表。

    返回的每一行描述一个片段：

    - start_ah / end_ah             : 观测窗口的容量范围；
    - pred_start_ah / pred_end_ah   : 电压预测窗口的容量范围；
    - is_valid_soh                  : 该片段是否可用于 SOH 任务；
    - is_valid_pretrain             : 该片段是否也拥有完整预测窗口。
    """
    charge = extract_charge_curve(raw)
    q_max = float(np.asarray(charge["Qc"]).max())

    observed_len = observed_capacity_pct * nominal_capacity
    pred_len = prediction_capacity_pct * nominal_capacity
    starts = start_positions_ah(nominal_capacity=nominal_capacity)

    rows: list[dict] = []
    for start_ah in starts:
        end_ah = start_ah + observed_len
        pred_start_ah = end_ah
        pred_end_ah = end_ah + pred_len
        rows.append(
            {
                "cell_id": cell_id,
                "cycle_index": cycle_index,
                "start_ah": start_ah,
                "end_ah": end_ah,
                "pred_start_ah": pred_start_ah,
                "pred_end_ah": pred_end_ah,
                "is_valid_soh": end_ah <= q_max,
                "is_valid_pretrain": pred_end_ah <= q_max,
                "n_charge_points": int(len(charge["V"])),
                "q_max_ah": q_max,
            }
        )

    return pd.DataFrame(rows)


def build_segment_index_for_cell(
    mat_path: Path,
    cell_id: str,
    cell_index: int,
    cycles: list[int],
    **kwargs,
) -> pd.DataFrame:
    """为一只电池的若干循环生成片段索引表。"""
    frames: list[pd.DataFrame] = []
    for cycle_index in cycles:
        raw = load_raw_cycle(mat_path, cell_index=cell_index, cycle_index=cycle_index)
        frames.append(
            build_segment_index_for_cycle(raw, cell_id=cell_id, cycle_index=cycle_index, **kwargs)
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """冒烟测试：对第一只电池的前 3 个循环生成片段索引并保存。"""
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cells", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "partial_segments_preview.parquet")
    args = parser.parse_args()

    files = discover_batch_files()
    batch_name = sorted(files)[0]
    mat_path = files[batch_name]

    cell_id = f"{batch_name}_c000"
    cycles = list(range(2, 2 + args.max_cycles))
    table = build_segment_index_for_cell(
        mat_path,
        cell_id=cell_id,
        cell_index=0,
        cycles=cycles,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(table.head(10).to_string(index=False))
    print(f"total rows: {len(table)}")
    print(f"valid_soh: {int(table['is_valid_soh'].sum())}, "
          f"valid_pretrain: {int(table['is_valid_pretrain'].sum())}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
