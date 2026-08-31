"""给现有 SIT 缓存追加 T_seq.npy（101 点绝对温度序列，°C）。

背景
----
阿伦尼乌斯极化补偿（V_comp = V + α·I·(T−30)）需要片段级的
101 点绝对温度曲线。现有 sit_cache_dvdq 只有 4 通道 X（I,V,Q,dV/dQ）
与 12 维温度特征，原始温度曲线在特征提取后被丢弃。

本脚本只补温度序列，不重写 X：
  - 逐电池调用 build_cell_samples_full（内部读 xlsx，慢，~3-4 分钟/只）；
  - 按 cell_ids.npy 的原始行位置写入（不假设缓存内电池顺序）；
  - 每行行数必须与缓存一致，否则显式报错（防错位）。

产物：<cache_dir>/T_seq.npy (N, 101) float32，meta.json 增加 shape_tseq。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TR_DIR = ROOT / "src" / "temperature_soh" / "Trainer"
sys.path.insert(0, str(TR_DIR))

from finetune_sit import build_cell_samples_full  # noqa: E402
from sit_io import DEFAULT_SIT_DIR, discover_sit_cells  # noqa: E402

DEFAULT_CACHE_DIR = ROOT / "data" / "processed" / "sit_cache_dvdq"


def append_tseq(cache_dir: Path, data_dir: Path) -> None:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缓存不存在: {cache_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n = int(meta["n"])
    cells_arr = np.load(cache_dir / "cell_ids.npy", allow_pickle=True)
    if len(cells_arr) != n:
        raise ValueError(f"cell_ids 行数 {len(cells_arr)} != meta.n {n}")

    tseq_all = np.zeros((n, 101), np.float32)
    t0 = time.perf_counter()
    for i, cell_id in enumerate(meta["cells"]):
        _, _, _, _, t_seq = build_cell_samples_full(cell_id, data_dir)
        rows = np.flatnonzero(cells_arr == cell_id)
        if len(rows) != len(t_seq):
            raise ValueError(
                f"{cell_id}: 缓存 {len(rows)} 行 != 重建 {len(t_seq)} 行，"
                "行序/行数不匹配，已中止（未写盘）"
            )
        tseq_all[rows] = t_seq
        print(
            f"[append_tseq] {cell_id}: {len(t_seq):,} 片段 "
            f"({i + 1}/{len(meta['cells'])}, {time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    out = cache_dir / "T_seq.npy"
    np.save(out, tseq_all)
    meta["shape_tseq"] = [n, 101]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[append_tseq] 完成：{n:,} × 101 -> {out} "
        f"({(time.perf_counter() - t0) / 60:.1f} 分钟)"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SIT_DIR)
    args = parser.parse_args()
    append_tseq(args.cache_dir, args.data_dir)


if __name__ == "__main__":
    main()
