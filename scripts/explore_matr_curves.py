"""M1 探索脚本：读懂 MATR 原始曲线的结构（教学脚手架）。


用法:
    python scripts/explore_matr_curves.py
    python scripts/explore_matr_curves.py --mat <路径> --cell 0 --cycles 1 5 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def deref(f: h5py.File, value) -> np.ndarray:
    """MATLAB v7.3 里 cell/struct 数组元素是 HDF5 引用，解引用成 numpy 数组。

    value 可能是一个引用（标量对象），也可能已经是一个数组，
    统一先转成数组再取 .item() 拿到引用本身，最后用 f[ref] 解引用。
    """
    ref = np.asarray(value).item()          # 把 (1,) 或标量包装拆成引用
    return np.asarray(f[ref][()])           # 引用 -> 真正的数据 -> numpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat", type=Path, default=ROOT / "data/external/matr/MATR_batch_20170512.mat")
    parser.add_argument("--cell", type=int, default=0)
    parser.add_argument("--cycles", type=int, nargs="+", default=[1, 5, 100])
    args = parser.parse_args()

    with h5py.File(str(args.mat), "r") as f:
        batch = f["batch"]

        print(f"顶层结构: {list(f.keys())}")
        print(f"batch 内字段: {list(batch.keys())}")
        n_cells = batch["summary"].shape[0]
        print(f"电池数: {n_cells}")

        # ---- 第 1 步：认识一只电池的"汇总信息"（你们 SOH 表的来源）----
        summary = f[batch["summary"][args.cell, 0]]
        print(f"\n电池 {args.cell} 的 summary 字段: {sorted(summary.keys())}")

        policy = deref(f, batch["policy_readable"][args.cell]).tobytes()
        policy = policy.decode("utf-16-le", errors="ignore").strip("\x00").strip()
        cycle_life = float(deref(f, batch["cycle_life"][args.cell]).ravel()[0])
        print(f"协议: {policy}   循环寿命(到80%): {cycle_life:.0f}")

        # ---- 第 2 步：进入这只电池的循环曲线 ----
        cyc = f[batch["cycles"][args.cell, 0]]          # 解引用 -> 循环 struct
        print(f"\ncycles 字段: {sorted(cyc.keys())}")
        n_cycles = np.asarray(cyc["V"]).shape[0]
        print(f"该电池循环数: {n_cycles}")

        # ---- 第 3 步：抽查几个循环的 V/I/T/t ----
        for j in args.cycles:
            if j >= n_cycles:
                print(f"  cycle {j} 超出范围，跳过")
                continue
            print(f"\n===== cycle {j} =====")
            for field in ["t", "V", "I", "T"]:
                arr = deref(f, cyc[field][j]).ravel()
                if len(arr) == 0:
                    print(f"  {field}: 空")
                    continue
                info = f"  {field}: len={len(arr)}"
                info += f"  头={np.round(arr[:3], 4)}  尾={np.round(arr[-3:], 4)}"
                if field == "t" and len(arr) > 1:
                    dt = np.diff(arr)
                    info += f"  dt中位数={np.median(dt):.6g}"
                print(info)

        # ---- 第 4 步（作业）：取消注释下面的画图代码，把曲线画出来看形状 ----
        import matplotlib.pyplot as plt
        j = args.cycles[0]
        t = deref(f, cyc["t"][j]).ravel()
        V = deref(f, cyc["V"][j]).ravel()
        I = deref(f, cyc["I"][j]).ravel()
        T = deref(f, cyc["T"][j]).ravel()
        QC = deref(f, cyc["Qc"][j]).ravel()

        Qd = deref(f, cyc["Qdlin"][j]).ravel()
        Td = deref(f, cyc["Tdlin"][j]).ravel()

        print("t:", t.shape, "V:", V.shape, "I:", I.shape, "T:", T.shape)
        print("Qdlin:", Qd.shape, "Tdlin:", Td.shape)

        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 6))
        axes[0].plot(t, V); axes[0].set_ylabel("V")
        axes[1].plot(t, I); axes[1].set_ylabel("I")
        axes[2].plot(t, T); axes[2].set_ylabel("T"); axes[2].set_xlabel("t")
        axes[3].plot(t, QC); axes[3].set_ylabel("QC")
        axes[4].plot(Qd, Td); axes[4].set_ylabel("Td")
        plt.savefig("results/figures/explore_one_cycle.png", dpi=150)


        print("trapz(I>0 段) =", np.trapz(I[I > 0], t[I > 0]))   # 充电电量
        print("Qc 终点 =", QC[-1])
        print(I.min())

if __name__ == "__main__":
    main()