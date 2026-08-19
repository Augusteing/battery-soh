"""partial_soh 数据质量排除模块（Scientific Reports 2026 口径）。

本模块维护 partial_soh（片段级 SOH 估计）的电池级与循环级排除规则。

核心结论（已对照 Severson 原始复现代码
`petermattia/revisit-severson-et-al/generate_voltage_arrays.m` 复现）：

    我们本地三个 .mat 文件共 140 个 channel：
      2017-05-12（本仓库文件名为 MATR_batch_20170512.mat）: 46 只
      2017-06-30 : 48 只
      2018-04-12 : 46 只

    Severson 2019 原始论文把这 140 个 channel 过滤成 124 只电池（41 + 43 + 40），
    排除 16 只。Scientific Reports 2026 又按“该数据集先前预处理惯例”再排除
    1 只寿命异常短的电池 b2c1，得到 123 只（99 train / 24 test）。

三种口径不要混用：

  - 140 channels：原始文件里所有通道；
  - 124 cells ：Severson 2019 论文口径（140 - 16）；
  - 123 cells ：Scientific Reports 2026 口径（124 - 1），本复现默认。

旧的 World Model 复现（arXiv 2603.10527）用的是另一套更宽松的口径
（140 - 2 = 138，只删 b1c0 / b1c18），与本论文口径不同，仅作历史对照保留。
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# 电池级排除常量
# ---------------------------------------------------------------------------

# 140 channels -> 124 cells 的 16 只排除名单。
# 依据 Severson 2019 原始代码中的三类规则：
#   1) batch 1 中 5 只“实验未跑到终点”的电池；
#   2) batch 2 中 5 只“从 batch 1 续测”的电池（避免同一物理电池重复计数）；
#   3) batch 3 中 channel 46、2 只未衰减到 0.88 Ah 的电池、3 只噪声电池。
SEVERSON_124_EXCLUDED: frozenset[str] = frozenset(
    {
        # batch 1 (2017-05-12)：未跑到寿命终点
        "20170512_c008",
        "20170512_c010",
        "20170512_c012",
        "20170512_c013",
        "20170512_c022",
        # batch 2 (2017-06-30)：从 batch 1 续测
        "2017-06-30_c007",
        "2017-06-30_c008",
        "2017-06-30_c009",
        "2017-06-30_c015",
        "2017-06-30_c016",
        # batch 3 (2018-04-12)：channel 46 + 未衰减到 EOL + 噪声
        "2018-04-12_c002",
        "2018-04-12_c023",
        "2018-04-12_c032",
        "2018-04-12_c037",
        "2018-04-12_c042",
        "2018-04-12_c043",
    }
)

# 124 -> 123 的 1 只异常短寿命电池。
# Scientific Reports 2026 原文：“排除一只寿命异常短的极端异常电池”。
# 即 Severson 数据集里的 b2c1（本仓库 ID：2017-06-30_c001），
# cycle_life 约 148，是 124 只里唯一低于约 300 的极端离群值。
# 第三方复现（Zenodo 20805401）也把 b2c1 称为 “one fast-failing cell”。
ABNORMAL_SHORT_LIVED_CELL: str = "2017-06-30_c001"

# 旧的 World Model 复现口径（140 -> 138），仅作历史对照，不在本模块默认使用。
# 它只排除 batch 1 里两只设备故障电池（容量出现异常尖峰）。
WORLD_MODEL_EXCLUDED: frozenset[str] = frozenset(
    {"20170512_c000", "20170512_c018"}
)

# Severson 约定：cycle 1 有已知数据质量问题，不进入训练/测试。
EXCLUDE_CYCLE_ONE = True


# ---------------------------------------------------------------------------
# 循环级排除（坏循环，只标记、不整删电池）
# ---------------------------------------------------------------------------

# 沿用之前 World Model 全量审计的坏循环结论。
# 这些循环只做标记，构建片段时跳过，不影响整只电池是否被排除。
#
# 注意：20170512_c000 / 20170512_c018 在旧的 World Model 口径里被整只删除，
# 因此它们的容量异常循环当时没有登记到坏循环表。切换到 123 口径后这两只
# 被保留（Severson 124 名单里没有删它们），所以这里必须补登记。
#
# 具体来说：
#   - 20170512_c000 在 cycle 12 有单循环尖峰；
#   - 20170512_c018 的充电电流/充电容量传感器在 cycle 7-40 发生漂移：
#     charge_capacity 以每循环约 +0.0275 Ah 线性爬升（1.088 -> 1.968），
#     最后在 cycle 40 出现 2.97 Ah 的尖峰；放电容量始终正常。
#   这对放电口径无影响，但对论文的“可充电容量”口径影响很大，必须整段标记。
BAD_CYCLES: dict[str, frozenset[int]] = {
    "20170512_c000": frozenset({12}),       # 容量尖峰（q_max 约 1.54）
    "20170512_c018": frozenset(range(7, 41)),  # 充电容量传感器漂移 + 尖峰
    "20170512_c005": frozenset({909}),
    "2017-06-30_c004": frozenset({247}),
    "2017-06-30_c006": frozenset({258}),
    "2017-06-30_c009": frozenset({311}),
    "2017-06-30_c010": frozenset({251}),
    "2017-06-30_c011": frozenset({250}),
    "2017-06-30_c012": frozenset({253}),
    "2017-06-30_c020": frozenset({249}),
    "2017-06-30_c022": frozenset({247}),
    "2017-06-30_c028": frozenset({250}),
    "2017-06-30_c037": frozenset({247}),
    "2017-06-30_c042": frozenset({247}),
    "2017-06-30_c044": frozenset({248}),
    "2018-04-12_c037": frozenset(range(593, 601)),
}


# ---------------------------------------------------------------------------
# 过滤函数
# ---------------------------------------------------------------------------


def _apply_cell_exclusions(
    table: pd.DataFrame, excluded: frozenset[str]
) -> pd.DataFrame:
    """按给定的电池排除集合过滤整只电池，返回副本。"""
    out = table.copy()
    n_before = len(out)
    n_cells_before = out["cell_id"].nunique()

    out = out[~out["cell_id"].isin(excluded)]
    if EXCLUDE_CYCLE_ONE:
        out = out[out["cycle_index"] != 1]

    n_removed = n_before - len(out)
    n_cells_removed = n_cells_before - out["cell_id"].nunique()
    print(
        f"[quality] removed {n_removed} rows / {n_cells_removed} cells; "
        f"remaining {len(out)} rows, {out['cell_id'].nunique()} cells"
    )
    return out


def apply_severson_124(table: pd.DataFrame) -> pd.DataFrame:
    """140 channels -> 124 cells（Severson 2019 论文口径）。"""
    return _apply_cell_exclusions(table, SEVERSON_124_EXCLUDED)


def apply_paper_123(table: pd.DataFrame) -> pd.DataFrame:
    """140 channels -> 123 cells（Scientific Reports 2026 口径，本复现默认）。"""
    excluded = SEVERSON_124_EXCLUDED | {ABNORMAL_SHORT_LIVED_CELL}
    return _apply_cell_exclusions(table, excluded)


def apply_world_model_138(table: pd.DataFrame) -> pd.DataFrame:
    """140 channels -> 138 cells（旧 World Model 复现口径，仅作对照）。"""
    return _apply_cell_exclusions(table, WORLD_MODEL_EXCLUDED)


def apply_exclusions(table: pd.DataFrame) -> pd.DataFrame:
    """partial_soh 的默认过滤入口：等价于 apply_paper_123。"""
    return apply_paper_123(table)


def mark_bad_cycles(table: pd.DataFrame) -> pd.DataFrame:
    """新增 is_bad_cycle 列，标记需要跳过的坏循环。"""
    out = table.copy()
    out["is_bad_cycle"] = False

    n_bad = 0
    for cell_id, cycles in BAD_CYCLES.items():
        mask = (out["cell_id"] == cell_id) & out["cycle_index"].isin(cycles)
        out.loc[mask, "is_bad_cycle"] = True
        n_bad += int(mask.sum())

    print(f"[quality] marked {n_bad} bad cycles")
    return out


if __name__ == "__main__":
    """冒烟测试：打印各口径下的电池数量。"""
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[3]
    table = pd.read_parquet(root / "data" / "processed" / "matr_soh_table.parquet")

    for name, func in (
        ("world_model_138", apply_world_model_138),
        ("severson_124", apply_severson_124),
        ("paper_123", apply_paper_123),
    ):
        print(f"\n== {name} ==")
        filtered = func(table)
        print(filtered.groupby("batch")["cell_id"].nunique().to_string())
