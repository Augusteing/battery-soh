"""M2 数据集统一构建流水线（最后一环）。

把 M2 的四个数据处理步骤串成一条命令（依赖自动补齐）:

  1) soh_table : 原始 .mat -> 统一 SOH 表
                 （scripts/build_matr_soh_table.py）
  2) labels    : SOH 表 -> 逐循环标签，SOH(k) = Q_discharge(k) / Q(2)
                 （DataLoader/labels.py）
  3) windows   : 标签 -> 窗口索引表（30 输入 + 80 未来，含坏循环清洗）
                 （DataLoader/windows.py）
  4) splits    : 窗口 -> 划分映射（按电池 70/15/15 + 按协议隔离）
                 （DataLoader/splits.py）

设计说明（软件工程）
---------------------
- 单一职责：本脚本只负责"编排与调度"，具体逻辑仍由各阶段模块实现；
- 依赖解析：请求某阶段时自动带上它的上游阶段，保证产物自洽；
- 可复现：默认参数与各阶段模块一致（seed=42、W=30、H=80）；
- 显式失败：任一阶段报错即中断（subprocess check=True），不留半成品。

用法:
    python "src/world_model/DataLoader/build_dataset.py"                # 全量重建
    python "src/world_model/DataLoader/build_dataset.py" --stages windows,splits
    python "src/world_model/DataLoader/build_dataset.py" --keep-all     # 消融：跳过排除规则
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DL_DIR = Path(__file__).resolve().parent

# 各阶段：脚本路径、输出产物、上游依赖
STAGES = {
    "soh_table": {
        "script": ROOT / "scripts" / "build_matr_soh_table.py",
        "out": ROOT / "data/processed/matr_soh_table.parquet",
        "deps": (),
    },
    "labels": {
        "script": DL_DIR / "labels.py",
        "out": ROOT / "data/processed/matr_soh_labels.parquet",
        "deps": ("soh_table",),
    },
    "windows": {
        "script": DL_DIR / "windows.py",
        "out": ROOT / "data/processed/matr_windows.parquet",
        "deps": ("labels",),
    },
    "splits": {
        "script": DL_DIR / "splits.py",
        "out": ROOT / "data/processed/splits.parquet",
        "deps": ("windows",),
    },
}
STAGE_ORDER = ("soh_table", "labels", "windows", "splits")


def resolve_stages(requested: set[str]) -> list[str]:
    """补齐依赖（传递闭包）并按固定执行顺序返回阶段名列表。"""
    need = set(requested)
    queue = list(requested)
    while queue:
        stage = queue.pop()
        for dep in STAGES[stage]["deps"]:
            if dep not in need:
                need.add(dep)
                queue.append(dep)
    return [s for s in STAGE_ORDER if s in need]


def build_command(stage: str, keep_all: bool, seed: int) -> list[str]:
    """构造某阶段的子进程命令（每个路径独立成 argv，天然支持含空格的目录）。"""
    cfg = STAGES[stage]
    cmd = [sys.executable, str(cfg["script"]), "--out", str(cfg["out"])]

    if stage == "labels":
        cmd += ["--input", str(STAGES["soh_table"]["out"])]
    elif stage == "windows":
        cmd += ["--labels", str(STAGES["labels"]["out"])]
        if keep_all:
            cmd.append("--keep-all")
    elif stage == "splits":
        cmd += ["--windows", str(STAGES["windows"]["out"]), "--seed", str(seed)]
    return cmd


def run_pipeline(stages: list[str], keep_all: bool, seed: int) -> None:
    """按顺序执行各阶段，任一失败即中断。"""
    for i, stage in enumerate(stages, 1):
        banner = f"=== 阶段 {i}/{len(stages)}: {stage} ==="
        print("\n" + "=" * len(banner), flush=True)
        print(banner, flush=True)
        print("=" * len(banner), flush=True)
        cmd = build_command(stage, keep_all=keep_all, seed=seed)
        subprocess.run(cmd, check=True)


def summarize() -> None:
    """构建完成后打印各产物的行数/电池数汇总（供人工核对）。"""
    print("\n" + "=" * 60)
    print("构建完成，产物汇总:")
    for stage in STAGE_ORDER:
        path = STAGES[stage]["out"]
        if not path.exists():
            print(f"  {stage:<10} 缺失: {path}")
            continue
        df = pd.read_parquet(path)
        cells = df["cell_id"].nunique() if "cell_id" in df.columns else "-"
        rows = f"{len(df):,} 行"
        extra = ""
        if stage == "splits":
            counts = df["split_by_cell"].value_counts()
            extra = "  [按电池] " + " ".join(f"{k}={int(v)}" for k, v in counts.items())
        print(f"  {stage:<10} {rows}, {cells} 只电池  {extra}")
        print(f"             -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", default="all",
                        help="逗号分隔的阶段名（soh_table,labels,windows,splits），默认 all")
    parser.add_argument("--keep-all", action="store_true",
                        help="透传给 windows 阶段：跳过论文排除规则（消融用）")
    parser.add_argument("--seed", type=int, default=42,
                        help="透传给 splits 阶段的随机种子")
    args = parser.parse_args()

    requested = set(STAGES) if args.stages.lower() == "all" else set(
        s.strip() for s in args.stages.split(",") if s.strip()
    )
    unknown = requested - set(STAGES)
    if unknown:
        parser.error(f"未知阶段: {sorted(unknown)}；可选 {list(STAGES)}")

    stages = resolve_stages(requested)
    print(f"执行顺序: {' -> '.join(stages)}")
    run_pipeline(stages, keep_all=args.keep_all, seed=args.seed)
    summarize()


if __name__ == "__main__":
    main()
