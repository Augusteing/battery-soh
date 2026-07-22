"""MATR（Severson et al., Nature Energy 2019）LFP 批次数据读取。

数据来源: https://data.matr.io/1/
原始 .mat 为 MATLAB v7.3（HDF5）格式（约 2.6-3.0 GB/batch），用 h5py 按
引用惰性读取，无需整文件载入内存。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 关注的老化汇总字段（Severson batch 结构中 summary 的字段名）
_SUMMARY_FIELDS = ["QDischarge", "QCharge", "IR", "Tmax", "Tavg", "Tmin", "chargetime"]


def _read_ref_array(f: h5py.File, group: h5py.Group, index: int) -> h5py.Group:
    """MATLAB v7.3 中 struct 数组以 object 引用存储，解引用取第 index 个元素。"""
    ref = group[index, 0]
    return f[ref]


def iter_cell_summaries(path: str | Path):
    """逐只电池产出 (cell_index, summary_dict)，惰性读取不爆内存。"""
    import h5py

    with h5py.File(str(path), "r") as f:
        batch = f["batch"]
        summary_refs = batch["summary"]
        n_cells = summary_refs.shape[0]
        for i in range(n_cells):
            summary = _read_ref_array(f, summary_refs, i)
            fields = {}
            for field in _SUMMARY_FIELDS:
                if field in summary:
                    fields[field] = np.asarray(summary[field][()]).ravel().astype(float)
            yield i, fields


def matr_to_soh_table(path: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    """从 .mat 批次文件生成 SOH 总表（与 Stanford 汇总表同构）。"""
    path = Path(path)
    batch_name = path.stem.replace("MATR_batch_", "")
    rows = []
    for i, fields in iter_cell_summaries(path):
        qd = fields.get("QDischarge")
        if qd is None or len(qd) < 10:  # 有效循环太少视为异常电池
            continue
        n = len(qd)
        rec = {
            "cell_id": f"{batch_name}_c{i:03d}",
            "batch": batch_name,
            "cycle_index": np.arange(1, n + 1),
            "discharge_capacity": qd,
        }
        for field in _SUMMARY_FIELDS[1:]:
            vals = fields.get(field)
            if vals is not None and len(vals) == n:
                rec[field.lower()] = vals
        # 首个循环可能为化成/初始化段（容量为 0），基准取前 10 个正容量中位数
        positive = qd[qd > 0]
        q0 = np.median(positive[:10]) if len(positive) else np.nan
        rec["soh"] = qd / q0
        rows.append(pd.DataFrame(rec))
    if not rows:
        raise ValueError(f"{batch_name}: 未解析到任何电池的 summary 数据")
    table = pd.concat(rows, ignore_index=True)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            table.to_parquet(out_path, index=False)
        else:
            table.to_csv(out_path, index=False)
    return table
