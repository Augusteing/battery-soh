"""MATR（Severson et al., Nature Energy 2019）LFP 批次数据读取。

数据来源: https://data.matr.io/1/
原始 .mat 为 MATLAB v7 格式（约 2.6-3.0 GB/batch），需要 scipy >= 1.11
（simplify_cells 支持）。加载整 batch 约需 8-16 GB 内存。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 关注的老化汇总字段（Severson batch 结构中 summary 的字段名）
_SUMMARY_FIELDS = ["QDischarge", "QCharge", "IR", "Tmax", "Tavg", "Tmin", "chargetime"]


def load_matr_batch(path: str | Path) -> list[dict]:
    """加载一个 MATR batch，返回电池 dict 列表。"""
    from scipy.io import loadmat

    mat = loadmat(str(path), simplify_cells=True)
    batch = mat["batch"]
    if isinstance(batch, dict):  # 单只电池时 squeeze 成 dict
        batch = [batch]
    return batch


def batch_to_cycle_table(batch: list[dict], batch_name: str) -> pd.DataFrame:
    """把一个 batch 的 summary 信息展开为逐循环长表。"""
    rows = []
    for idx, cell in enumerate(batch):
        summary = cell.get("summary", {})
        if not isinstance(summary, dict) or "QDischarge" not in summary:
            continue
        qd = np.atleast_1d(np.asarray(summary["QDischarge"], dtype=float))
        n = len(qd)
        if n < 10:  # 有效循环太少视为异常电池
            continue
        rec = {
            "cell_id": f"{batch_name}_c{idx:03d}",
            "batch": batch_name,
            "cycle_index": np.arange(1, n + 1),
            "discharge_capacity": qd,
        }
        for field in _SUMMARY_FIELDS[1:]:
            vals = np.atleast_1d(np.asarray(summary.get(field, np.full(n, np.nan)), dtype=float))
            if len(vals) == n:
                rec[field.lower()] = vals
        rec["soh"] = qd / qd[0]
        rows.append(pd.DataFrame(rec))
    if not rows:
        raise ValueError(f"{batch_name}: 未解析到任何电池的 summary 数据")
    return pd.concat(rows, ignore_index=True)


def matr_to_soh_table(path: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    """从 .mat 批次文件生成 SOH 总表（与 Stanford 汇总表同构）。"""
    path = Path(path)
    batch_name = path.stem.replace("MATR_batch_", "")
    table = batch_to_cycle_table(load_matr_batch(path), batch_name)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            table.to_parquet(out_path, index=False)
        else:
            table.to_csv(out_path, index=False)
    return table
