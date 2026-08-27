"""部分充电片段构建模块（4 通道版本）。

本模块把“统一循环结构”（无论来自 pkl 还是 mat_io）切成论文描述的
短片段，并输出 4 通道 `(I, V, Q, T)` 的插值曲线。

论文口径（与 partial_soh 保持一致）：

- 观测窗口长度为 20% 额定容量（1.1Ah × 20% = 0.22Ah）；
- 起点从 0% 额定容量移动到 50% 额定容量，每次移动 1%；
- 每 1% 额定容量区间插值为 5 个等距点 -> 窗口内 101 个点；
- 电压预训练时，观测窗之后的预测窗口长度为 7% 额定容量。

与 partial_soh 的区别：输入是统一循环结构（`current_in_A` 已是安培、
`time_in_s` 已是秒），输出额外包含温度通道 T，并在索引表里写入该电池
的工作温度（供按温度分层划分验证集）。

注意：本模块只生成“片段索引 + 插值数组”，不计算 SOH 标签、
不写训练缓存。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 把本目录放进 sys.path，便于像 scripts 一样直接运行。
DL_DIR = Path(__file__).resolve().parent
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from mat_io import discover_batch_files, load_unified_cycle  # noqa: E402
from registry import DATASETS, infer_temperature_c, list_cell_files, load_cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# 片段参数（论文口径）
# ---------------------------------------------------------------------------

NOMINAL_CAPACITY_AH = 1.1
OBSERVED_CAPACITY_PCT = 0.20    # 观测窗口长度占额定容量的比例
START_MIN_PCT = 0.00            # 起点最小值（0% 额定容量）
START_MAX_PCT = 0.50            # 起点最大值（50% 额定容量）
START_STEP_PCT = 0.01           # 起点每次移动 1% 额定容量
STEPS_PER_PCT = 5               # 每 1% 额定容量插值成 5 步
PREDICTION_CAPACITY_PCT = 0.07  # 电压预测窗口长度占额定容量的比例

# 容量坐标容差：充电曲线 Qc 的起点可能不是精确 0（例如 3.6e-6 Ah），
# 也可能因为记录缺失而从某个正值开始。插值和合法性判断共用此容差。
CAPACITY_TOLERANCE_AH = 1e-4

# 判断充电段的电流阈值（统一结构里电流单位是安培，充电为正）。
CHARGE_CURRENT_THRESHOLD_A = 0.0

# 4 通道顺序：与 Trainer 约定为 [I, V, Q, T]。
CHANNEL_NAMES = ("I", "V", "Q", "T")


# ---------------------------------------------------------------------------
# 充电阶段提取（统一结构版）
# ---------------------------------------------------------------------------

def extract_charge_curve(cycle: dict[str, Any]) -> dict[str, np.ndarray]:
    """从统一循环结构里提取充电阶段（I > 0 的点）。

    返回的 dict 与 partial_soh 的 charge.py 同构，但输入是统一结构：
      - current_in_A 已是安培；
      - charge_capacity_in_Ah 用作容量坐标 Qc；
      - temperature_in_C 一并筛选。
    """
    i = np.asarray(cycle["current_in_A"], dtype=float)
    v = np.asarray(cycle["voltage_in_V"], dtype=float)
    qc = np.asarray(cycle["charge_capacity_in_Ah"], dtype=float)
    t_s = np.asarray(cycle["time_in_s"], dtype=float)
    t_c = cycle.get("temperature_in_C")

    mask = i > CHARGE_CURRENT_THRESHOLD_A
    if not np.any(mask):
        raise ValueError("该循环没有充电段（I > 0 的点数为 0）")

    out = {
        "t": t_s[mask],
        "V": v[mask],
        "I": i[mask],
        "Qc": qc[mask],
    }
    if t_c is not None and len(t_c) == len(i):
        out["T"] = np.asarray(t_c, dtype=float)[mask]
    else:
        # 极少数循环可能缺温度序列：用空数组占位，插值时 np.interp
        # 会因为没有有效数据而失败，由上层用温度标签兜底。
        out["T"] = np.full(int(mask.sum()), np.nan, dtype=float)
    return out


# ---------------------------------------------------------------------------
# 容量网格与插值
# ---------------------------------------------------------------------------

def capacity_grid(
    start_ah: float,
    end_ah: float,
    steps_per_pct: int = STEPS_PER_PCT,
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
) -> np.ndarray:
    """生成从 start_ah 到 end_ah 的等距容量网格。

    20% 窗口、每 1% 5 步：0.22Ah 窗口 -> 100 个区间 -> 101 个点。
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

    返回：
    {
        "capacity": 等距容量网格 (101,)，
        "t" / "V" / "I" / "T": 在网格上插值后的曲线 (101,)，
    }

    注意：容量网格本身即 Q 通道（容量坐标），因此训练时
    4 通道 `(I, V, Q, T)` 中的 Q 就是 `capacity`。
    """
    q = np.asarray(charge["Qc"], dtype=float)
    if q.size < 2:
        raise ValueError("充电阶段点数不足，无法插值")

    tolerance = CAPACITY_TOLERANCE_AH
    if start_ah < q.min() - tolerance or end_ah > q.max() + tolerance:
        raise ValueError(
            f"片段 [{start_ah:.4f}, {end_ah:.4f}] Ah 超出充电容量范围 "
            f"[{q.min():.4f}, {q.max():.4f}] Ah"
        )

    start_clipped = float(max(start_ah, q.min()))
    end_clipped = float(min(end_ah, q.max()))
    grid = capacity_grid(start_clipped, end_clipped, steps_per_pct, nominal_capacity)
    out: dict[str, np.ndarray] = {"capacity": grid}

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


# ---------------------------------------------------------------------------
# 片段索引
# ---------------------------------------------------------------------------

def build_segment_index_for_cycle(
    cycle: dict[str, Any],
    cell_id: str,
    cycle_index: int,
    temperature_c: float,
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
    observed_capacity_pct: float = OBSERVED_CAPACITY_PCT,
    prediction_capacity_pct: float = PREDICTION_CAPACITY_PCT,
) -> pd.DataFrame:
    """为一个循环生成所有合法片段的索引表。

    返回的每一行描述一个片段：
      - cell_id / cycle_index / temperature_c：定位与温度标签；
      - start_ah / end_ah             ：观测窗口容量范围；
      - pred_start_ah / pred_end_ah   ：电压预测窗口容量范围；
      - is_valid_soh                  ：片段能否用于 SOH 任务；
      - is_valid_pretrain             ：片段是否也拥有完整预测窗口。
    """
    charge = extract_charge_curve(cycle)
    q_min = float(np.asarray(charge["Qc"]).min())
    q_max = float(np.asarray(charge["Qc"]).max())
    tolerance = CAPACITY_TOLERANCE_AH

    observed_len = observed_capacity_pct * nominal_capacity
    pred_len = prediction_capacity_pct * nominal_capacity
    starts = start_positions_ah(nominal_capacity=nominal_capacity)

    rows: list[dict[str, Any]] = []
    for start_ah in starts:
        end_ah = start_ah + observed_len
        pred_start_ah = end_ah
        pred_end_ah = end_ah + pred_len
        rows.append(
            {
                "cell_id": cell_id,
                "cycle_index": cycle_index,
                "temperature_c": temperature_c,
                "start_ah": start_ah,
                "end_ah": end_ah,
                "pred_start_ah": pred_start_ah,
                "pred_end_ah": pred_end_ah,
                "is_valid_soh": (start_ah >= q_min - tolerance)
                and (end_ah <= q_max + tolerance),
                "is_valid_pretrain": (start_ah >= q_min - tolerance)
                and (pred_end_ah <= q_max + tolerance),
                "n_charge_points": int(len(charge["V"])),
                "q_min_ah": q_min,
                "q_max_ah": q_max,
            }
        )
    return pd.DataFrame(rows)


def build_segment_index_for_cell(
    cell: dict[str, Any],
    cycles: list[int],
    nominal_capacity: float = NOMINAL_CAPACITY_AH,
    **kwargs,
) -> pd.DataFrame:
    """为一只电池的若干循环生成片段索引表。

    cell 是统一 cell 结构（registry.load_cell 的 pkl 或
    mat_io.load_unified_cell），cycles 是 1-based 循环号列表。
    temperature_c 可由调用方通过 kwargs 显式传入；若未传，
    从 cycle_data 里第一个非空温度序列取中位数兜底。
    """
    temperature_c = kwargs.pop("temperature_c", None)
    if temperature_c is None:
        for cyc in cell.get("cycle_data", []):
            temps = cyc.get("temperature_in_C")
            if temps is not None and len(temps) > 0:
                arr = np.asarray(temps, dtype=float)
                if np.isfinite(arr).any():
                    temperature_c = float(np.median(arr[np.isfinite(arr)]))
                    break
    if temperature_c is None:
        raise ValueError("无法确定该电池的温度，请显式传入 temperature_c")

    by_cycle = {int(c["cycle_number"]): c for c in cell["cycle_data"]}
    frames: list[pd.DataFrame] = []
    for cycle_index in cycles:
        if cycle_index not in by_cycle:
            continue
        frames.append(
            build_segment_index_for_cycle(
                by_cycle[cycle_index],
                cell_id=cell["cell_id"],
                cycle_index=cycle_index,
                temperature_c=temperature_c,
                nominal_capacity=nominal_capacity,
                **kwargs,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 冒烟测试入口
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """对 SNL（pkl）与 MATR（mat）各一个循环生成片段索引并保存预览。"""
    sys.stdout.reconfigure(encoding="utf-8")
    out_path = ROOT / "data" / "processed" / "temperature_segments_preview.parquet"

    # 1) SNL：第一块 25°C 电池的第 100 个循环
    snl = DATASETS["SNL"]
    snl_files = list_cell_files(snl)
    snl_cell = load_cell(snl, snl_files[1])  # 第二块（25°C）
    snl_temp = infer_temperature_c(snl, snl_cell)
    snl_table = build_segment_index_for_cell(
        snl_cell, cycles=[100], temperature_c=snl_temp
    )
    snl_table.insert(0, "dataset", "SNL")

    # 2) MATR：第一个批次第一只电池的第 2 个循环
    batches = discover_batch_files()
    batch_name = sorted(batches)[0]
    mat_cycle = load_unified_cycle(batches[batch_name], cell_index=0, cycle_index=2)
    mat_cell = {
        "cell_id": f"{batch_name}_c000",
        "cycle_data": [mat_cycle],
        "nominal_capacity_in_Ah": 1.1,
    }
    mat_table = build_segment_index_for_cell(
        mat_cell, cycles=[2], temperature_c=30.0
    )
    mat_table.insert(0, "dataset", "MATR")

    table = pd.concat([snl_table, mat_table], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)

    print("SNL 片段数（cycle 100）:", len(snl_table),
          "有效 soh:", int(snl_table["is_valid_soh"].sum()))
    print("MATR 片段数（cycle 2）:", len(mat_table),
          "有效 soh:", int(mat_table["is_valid_soh"].sum()))

    # 3) 验证 4 通道插值形状：取第一个有效片段
    row = snl_table[snl_table["is_valid_soh"]].iloc[0]
    seg = interpolate_segment(
        extract_charge_curve(snl_cell["cycle_data"][99]),
        start_ah=row["start_ah"],
        end_ah=row["end_ah"],
    )
    print(f"\nSNL 片段 [{row['start_ah']:.3f}, {row['end_ah']:.3f}] Ah 插值后:")
    print(f"  capacity/V/I/T 形状: "
          f"{seg['capacity'].shape}, 通道数=4")
    print(f"  温度范围: {seg['T'].min():.2f} ~ {seg['T'].max():.2f} °C")
    print(f"  电压范围: {seg['V'].min():.3f} ~ {seg['V'].max():.3f} V")
    print(f"  电流范围: {seg['I'].min():.3f} ~ {seg['I'].max():.3f} A")
    print(f"\n预览已保存: {out_path}")


if __name__ == "__main__":
    _smoke_test()
