"""数据质量排除规则模块（M2 · 第五步）。

论文口径（World Model, arXiv:2603.10527）:
  - 按 Severson et al. 约定，cycle 1 因已知数据质量问题被排除；
  - batch 1 的 cell 0 与 cell 18 因设备故障被排除，
    得到 138 只可用电池（44 + 48 + 46）。

我们与论文的差异（报告中需注明）:
  - 我们没有论文的 batch 3（2017-07-25），用 batch 2018-04-12 替代；
  - 因此电池 ID 组合不完全一致，但 batch 1 的排除规则完全相同
    （46 -> 44 只）。

两级排除策略（2026-08-12 全量审计后定稿）:
  - 整电池排除（EXCLUDED_CELLS）：只有论文点名的 2 只设备故障电池；
    审计发现 batch 2017-06-30 的 c012/c044 有同型容量越界尖峰，但为
    单循环事件、周围数据完全正常，故不整删，改在循环级处理；
  - 循环级排除（BAD_CYCLES）：审计发现的坏循环（容量尖峰/凹陷、IR=0、
    多循环测量噪声）。窗口构建会避开触碰这些循环的窗口，不插值、不整删。

设计说明（软件工程）
---------------------
- 单一职责：本模块只维护"哪些数据要被排除"的规则，不负责构建窗口；
- 常量集中管理：排除列表集中在此处，便于审计与修改；
- 幂等：apply_exclusions / mark_bad_cycles 可重复调用，结果不变。
"""

from __future__ import annotations

import pandas as pd

# batch 1（2017-05-12）设备故障电池（论文明确点名）
EXCLUDED_CELLS = frozenset({"20170512_c000", "20170512_c018"})

# cycle 1 因已知数据质量问题被排除（Severson 约定）
EXCLUDE_CYCLE_ONE = True

# 审计发现的坏循环（判据见 docs/m1_data_exploration.md 与
# data/processed/_audit_all_cells_qd.csv）：
#   - Qd 越界尖峰（> 1.1 Ah 物理上限）或单循环凹陷（|ΔQd| > 0.05 且下循环恢复）；
#   - 放电中断（Tmax 骤降 2-4°C）或 IR = 0 的测量故障；
#   - 2018-04-12_c037 在 cycle 593-600 的多循环测量噪声（Qd 波动 ±7%）。
# 注：被排除电池（EXCLUDED_CELLS）内的坏循环无需在此重复登记。
BAD_CYCLES: dict[str, frozenset[int]] = {
    "20170512_c005": frozenset({909}),                       # 放电中断 + IR=0
    "2017-06-30_c004": frozenset({247}),
    "2017-06-30_c006": frozenset({258}),
    "2017-06-30_c009": frozenset({311}),
    "2017-06-30_c010": frozenset({251}),
    "2017-06-30_c011": frozenset({250}),
    "2017-06-30_c012": frozenset({253}),                     # 容量越界尖峰
    "2017-06-30_c020": frozenset({249}),
    "2017-06-30_c022": frozenset({247}),
    "2017-06-30_c028": frozenset({250}),
    "2017-06-30_c037": frozenset({247}),
    "2017-06-30_c042": frozenset({247}),
    "2017-06-30_c044": frozenset({248}),                     # 容量越界尖峰
    "2018-04-12_c037": frozenset(range(593, 601)),           # 多循环噪声段
}


def apply_exclusions(labels: pd.DataFrame) -> pd.DataFrame:
    """应用排除规则，返回过滤后的标签表（副本）。

    规则:
      1) 删除 EXCLUDED_CELLS 中的电池（整只移除）；
      2) 若 EXCLUDE_CYCLE_ONE，删除每只电池的 cycle_index == 1 的行。

    注意: 过滤后的表格直接交给窗口构建；SOH 参考循环仍为 cycle 2
    （cycle 1 被排除不影响参考口径）。
    """
    out = labels.copy()
    n_before = len(out)

    # 规则 1：删除设备故障电池
    out = out[~out["cell_id"].isin(EXCLUDED_CELLS)]

    # 规则 2：删除 cycle 1（数据质量）
    if EXCLUDE_CYCLE_ONE:
        out = out[out["cycle_index"] != 1]

    n_removed = n_before - len(out)
    print(f"[data_quality] 排除 {len(EXCLUDED_CELLS)} 只故障电池 + cycle 1，"
          f"移除 {n_removed:,} 行；剩余 {len(out):,} 行, "
          f"{out['cell_id'].nunique()} 只电池")
    return out


def mark_bad_cycles(labels: pd.DataFrame) -> pd.DataFrame:
    """在标签表上新增 is_bad_cycle 列（True = 该循环为坏循环）。

    不删除任何行（保持窗口构建所需的连续循环），由 windows.py
    在构建窗口时避开触碰坏循环的窗口。

    幂等：重复调用会先重置再标记，结果不变。
    """
    out = labels.copy()
    out["is_bad_cycle"] = False
    n_bad = 0
    for cell_id, cycles in BAD_CYCLES.items():
        mask = (out["cell_id"] == cell_id) & out["cycle_index"].isin(cycles)
        out.loc[mask, "is_bad_cycle"] = True
        n_bad += int(mask.sum())
    print(f"[data_quality] 标记坏循环 {n_bad} 个（涉及 {len(BAD_CYCLES)} 只电池）")
    return out
