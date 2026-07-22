"""Stanford 动态循环数据集（Geslin et al., Nature Energy 2024）读取与汇总。

数据来源: https://purl.stanford.edu/td676xr4322
- aging_summary_cell_XXX.csv: 每只电池的逐循环老化汇总（SOH 标签来源）
- raw_data_cell_XXX.csv: 每只电池的原始时间序列（电压/电流/温度，片段数据来源）
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# 放电容量列的候选命名（按真实表头：Normalized Discharge Capacity [-]）
_CAPACITY_CANDIDATES = [
    "normalized discharge capacity [-]",
    "discharge_capacity",
    "discharge capacity (ah)",
    "discharge_capacity_ah",
    "qd",
    "capacity",
]

_CELL_RE = re.compile(r"cell_(\d+)")

# 诊断循环判定阈值：归一化容量明显大于 1 的行为诊断例行（含多段充放电），
# 常规老化循环为满充满放，归一化容量约 0.9-1.1
_DIAG_CAPACITY_THRESHOLD = 1.5

# SOH 基准取前 N 个常规循环的中位数，抑制初期波动
_REF_CYCLES = 10


def detect_capacity_column(columns: list[str]) -> str:
    """在表头中自动定位放电容量列。"""
    lowered = {c.lower().strip(): c for c in columns}
    for cand in _CAPACITY_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    for low, orig in lowered.items():
        if "discharge" in low and ("cap" in low or "ah" in low):
            return orig
    raise KeyError(f"无法识别放电容量列，现有列: {columns}")


def detect_cycle_column(columns: list[str]) -> str:
    lowered = {c.lower().strip(): c for c in columns}
    for cand in ("cycle_index", "cycle", "cycle_number", "cycle number", "cycle_idx"):
        if cand in lowered:
            return lowered[cand]
    for low, orig in lowered.items():
        if "cycle" in low:
            return orig
    raise KeyError(f"无法识别循环序号列，现有列: {columns}")


def load_aging_summary(path: str | Path, drop_diagnostic: bool = True) -> pd.DataFrame:
    """读取单只电池的 aging_summary，并计算 SOH。

    SOH 基准为剔除诊断循环后前 _REF_CYCLES 个常规循环的中位数。
    """
    path = Path(path)
    df = pd.read_csv(path)
    cap_col = detect_capacity_column(list(df.columns))
    cyc_col = detect_cycle_column(list(df.columns))
    df = df.rename(columns={cap_col: "discharge_capacity", cyc_col: "cycle_index"})
    keep = ["cycle_index", "discharge_capacity"]
    cum_col = next((c for c in df.columns if "cumulative" in c.lower()), None)
    if cum_col:
        df = df.rename(columns={cum_col: "cumulative_capacity"})
        keep.append("cumulative_capacity")
    df = df[keep].dropna().sort_values("cycle_index")
    df = df[df["discharge_capacity"] > 0]
    df["is_diagnostic"] = df["discharge_capacity"] > _DIAG_CAPACITY_THRESHOLD
    regular = df[~df["is_diagnostic"]] if drop_diagnostic else df
    q0 = regular["discharge_capacity"].head(_REF_CYCLES).median()
    df["soh"] = df["discharge_capacity"] / q0
    m = _CELL_RE.search(path.stem)
    df["cell_id"] = f"cell_{m.group(1)}" if m else path.stem
    return df.reset_index(drop=True)


def load_protocol_map(summary_dir: str | Path) -> dict[str, str]:
    """读取 protocol_mapping_dic.json（cell_id -> 协议名）。"""
    import json

    path = Path(summary_dir) / "protocol_mapping_dic.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_soh_table(summary_dir: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    """汇总目录下全部 aging_summary，生成 SOH 总表（含协议与协议族）。"""
    summary_dir = Path(summary_dir)
    files = sorted(summary_dir.glob("aging_summary_cell_*.csv"))
    if not files:
        raise FileNotFoundError(f"{summary_dir} 下未找到 aging_summary_cell_*.csv")
    table = pd.concat([load_aging_summary(f) for f in files], ignore_index=True)
    protocols = load_protocol_map(summary_dir)
    if protocols:
        table["protocol"] = table["cell_id"].map(protocols)
        table["protocol_family"] = table["protocol"].str.split("_").str[0]
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            table.to_parquet(out_path, index=False)
        else:
            table.to_csv(out_path, index=False)
    return table


def load_raw_cell(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """读取单只电池的原始时间序列（文件较大，支持 nrows 抽样）。"""
    return pd.read_csv(path, nrows=nrows)
