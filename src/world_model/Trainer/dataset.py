"""窗口数据集加载器（M3 · 第一步）。

职责：把 M2 的窗口索引表（matr_windows.parquet）变成可训练的样本。
每个窗口 -> 一个样本:

    X      : (W, 3, Tmax) float32 —— W=30 个连续循环的 V/I/T 曲线
              （每个循环先由 DataLoader.CycleStandardizer 定长到 Tmax=1000）
    y_cur  : float —— 当前 SOH s(k)（论文标签口径 Q(k)/Q(2)）
    y_fut  : (H,) —— 未来 H=80 循环的 SOH 轨迹 s(k+1..k+80)
    ir_0   : float —— 电池初始内阻 R0（第一个正有限 IR，通常为 cycle 2）
    ir_k   : float —— 当前循环 k 的内阻 R_last（供物理损失 L_ir 使用）
    stage  : str —— 当前 SOH 所在老化阶段（s1/s2/s3）

    注意：IR 只作为训练时的物理约束标签（L_ir），绝不进入模型输入
    （论文明确"internal resistance ... excluded from the model input"）。

设计说明（软件工程）
---------------------
- 索引优先、按需读取：窗口表不物化曲线。本类在首次访问某只电池时
  才读取并标准化该电池的全部循环曲线，并用 LRU 缓存最近若干只，
  避免把全部数据（约 1.4GB）一次性压进内存；
- 对齐口径：pos 是"排除 cycle 1 之后"的 0-based 位置，
  原始 .mat 里循环数组下标从 0 开始、cycle 1 在下标 0，
  因此  raw_index = pos + 1（本模块用断言保护该约定）；
- 单一职责：本模块只负责"取出样本"，标准化数值分布（z-score）由
  normalize.py 在训练集上拟合后套用；
- fail fast：找不到 .mat / 曲线与标签长度不齐 / 索引越界，立即报错。

用法（演示）:
    python "src/world_model/Trainer/dataset.py"
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]           # 项目根目录（battery-soh）
DL_DIR = ROOT / "src/world_model/DataLoader"
sys.path.insert(0, str(DL_DIR))                      # DataLoader 不是包，手动加路径
from imean import mean_charging_current              # noqa: E402  充电段平均电流
from standardize_cycle import CycleStandardizer      # noqa: E402  循环定长标准化

def _batch_from_cell_id(cell_id: str) -> str:
    """从 cell_id（如 20170512_c000）取出批次名（20170512）。"""
    return cell_id.rsplit("_", 1)[0]


def _cell_index(cell_id: str) -> int:
    """从 cell_id 取出该电池在批次内的编号（c000 -> 0）。"""
    return int(cell_id.rsplit("_", 1)[1][1:])


def discover_batch_files(mat_dir: Path) -> dict[str, Path]:
    """扫描 data/external/matr/*.mat，建立 批次名 -> 文件路径 的映射。

    批次名与 build_matr_soh_table.py 保持一致：
      优先取文件名中的日期；有横线（2017-06-30）原样保留，
      无横线（20170512）也保留无横线形式，不做格式转换。
    """
    mapping: dict[str, Path] = {}
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{8})")
    for p in sorted(mat_dir.glob("*.mat")):
        m = pattern.search(p.name)
        if m:
            mapping[m.group(1)] = p
    if not mapping:
        raise FileNotFoundError(f"{mat_dir} 下没有找到任何 .mat 文件")
    return mapping


def _deref(f: h5py.File, value) -> np.ndarray:
    """解引用 .mat 里的 HDF5 对象引用，返回 numpy 数组。"""
    ref = np.asarray(value).item()
    return np.asarray(f[ref][()])


def load_cell_curves(mat_path: Path, cell_idx: int,
                     standardizer: CycleStandardizer
                     ) -> tuple[np.ndarray, np.ndarray]:
    """读取某只电池的全部循环曲线并定长标准化。

    返回 (curves, imean):
      - curves: (n_cycles, 3, Tmax) float32；第 j 行 = 原始 cycle j+1
        的曲线（即 raw_index = pos + 1 的来源）；
      - imean : (n_cycles,) float64；每循环的平均充电电流 I_mean
        （充电段 I > 0 的均值，C-rate），作为论文的 action 向量 u(k)。
    """
    with h5py.File(str(mat_path), "r") as f:
        batch = f["batch"]
        if cell_idx >= batch["cycles"].shape[0]:
            raise IndexError(f"cell_idx={cell_idx} 超出 {mat_path.name} 的电池数")
        cyc = f[batch["cycles"][cell_idx, 0]]
        n = np.asarray(cyc["V"]).shape[0]

        curves = np.empty((n, 3, standardizer.tmax), dtype=np.float32)
        imean = np.empty(n, dtype=np.float64)
        for j in range(n):
            raw = {
                "V": _deref(f, cyc["V"][j]).ravel(),
                "I": _deref(f, cyc["I"][j]).ravel(),
                "T": _deref(f, cyc["T"][j]).ravel(),
            }
            # 防御：同族长度不一致说明数据损坏（CycleStandardizer 也会校验）
            if len({len(v) for v in raw.values()}) != 1:
                raise ValueError(f"{mat_path.name} cell {cell_idx} cycle {j+1} "
                                 "V/I/T 长度不一致")
            # I_mean 只在充电段（I > 0）有定义。实测 batch1（20170512）的
            # cycle 1 只有 2 个点且 I 恒为 0（这正是 Severson 约定排除的
            # "cycle 1 数据质量问题"），其他批次无此问题。该位置对应的
            # raw index 0 永远不可能是窗口最后一个观测（pos >= W-1 = 29），
            # 因此置 NaN 即可，训练时永远读不到（__getitem__ 会再校验一次）。
            try:
                imean[j] = mean_charging_current(raw["I"])   # 充电段均值（原始量纲）
            except ValueError:
                imean[j] = np.nan
            curves[j] = standardizer(raw)
    return curves, imean


class LRUCache:
    """极简 LRU 缓存：按访问顺序维护 key，超出容量时淘汰最久未用的。"""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity 必须 >= 1")
        self.capacity = capacity
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str):
        if key not in self._store:
            return None
        self._store.move_to_end(key)               # 刚访问过 -> 移到末尾（最近）
        return self._store[key]

    def put(self, key: str, value: np.ndarray) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:    # 淘汰最久未用的（开头）
            self._store.popitem(last=False)


class WindowDataset:
    """窗口数据集：windows 表 + 标签表 -> 可索引样本。

    参数
    ----
    windows   : matr_windows.parquet 读出的 DataFrame
    labels    : matr_soh_labels.parquet 读出的 DataFrame（需 soh_q2 列）
    mat_dir   : 原始 .mat 所在目录（data/external/matr）
    mode      : 传给 CycleStandardizer 的曲线口径（"full" 或 "discharge"）
    W, H      : 窗口长度与未来地平线（必须与 windows 表构建时一致）
    cache_size: 内存中最多缓存几只电池的曲线
    normalizer: 可选的 z-score 变换器（normalize.ChannelNormalizer）；
                提供时 __getitem__ 的 X 会先标准化再返回（验证/测试复用训练参数）
    """

    def __init__(self, windows: pd.DataFrame, labels: pd.DataFrame,
                 mat_dir: Path, mode: str = "full",
                 W: int = 30, H: int = 80, cache_size: int = 4,
                 normalizer=None):
        self.windows = windows.reset_index(drop=True)
        self.W, self.H = W, H
        self.normalizer = normalizer
        self.standardizer = CycleStandardizer(tmax=1000, mode=mode)
        self.batch_files = discover_batch_files(mat_dir)
        self.cache = LRUCache(cache_size)
        self._imean_cache: dict[str, np.ndarray] = {}   # 每电池的 I_mean 数组

        # 把标签表整理成"按电池的 soh_q2 数组"，pos 直接索引
        self._labels: dict[str, np.ndarray] = {}
        self._ir: dict[str, np.ndarray] = {}
        self._ir0: dict[str, float] = {}
        lab = labels[["cell_id", "cycle_index", "soh_q2"]].copy()
        lab = lab[lab["cell_id"].isin(windows["cell_id"].unique())]  # 只留窗口涉及的电池
        for cell_id, g in lab.groupby("cell_id", sort=False):
            g = g.sort_values("cycle_index")
            # 与 windows.py 相同的口径：cycle 1 不参与 pos 编号
            g = g[g["cycle_index"] != 1]
            self._labels[cell_id] = g["soh_q2"].to_numpy(dtype=np.float64)

        # IR 数组（与 soh_q2 同序对齐 pos）+ 初始内阻 R0
        lab_ir = labels[["cell_id", "cycle_index", "ir"]].copy()
        lab_ir = lab_ir[lab_ir["cell_id"].isin(windows["cell_id"].unique())]
        for cell_id, g in lab_ir.groupby("cell_id", sort=False):
            g = g.sort_values("cycle_index")
            g = g[g["cycle_index"] != 1]
            self._ir[cell_id] = g["ir"].to_numpy(dtype=np.float64)
            valid = g[g["ir"] > 0]
            valid = valid[np.isfinite(valid["ir"])]
            if valid.empty:
                raise ValueError(f"{cell_id}: 没有正有限的内阻值，无法计算 R0")
            self._ir0[cell_id] = float(valid["ir"].iloc[0])   # 首个正有限值 = 初始内阻

        # 预检：所有窗口的 pos / pos+H 都在该电池标签范围内
        for cell_id, sub in self.windows.groupby("cell_id"):
            n = len(self._labels[cell_id])
            bad = (sub["pos"] > n - self.H - 1).sum()
            if bad:
                raise ValueError(f"{cell_id}: {bad} 个窗口超出标签范围，"
                                 "窗口表与标签表不匹配")

    def __len__(self) -> int:
        return len(self.windows)

    def _cell_curves(self, cell_id: str) -> np.ndarray:
        """取某只电池的曲线数组（带 LRU 缓存），同时缓存其 I_mean。"""
        cached = self.cache.get(cell_id)
        if cached is not None:
            return cached
        batch = _batch_from_cell_id(cell_id)
        mat_path = self.batch_files.get(batch)
        if mat_path is None:
            raise FileNotFoundError(f"找不到批次 {batch} 的 .mat 文件")
        curves, imean = load_cell_curves(mat_path, _cell_index(cell_id),
                                         self.standardizer)
        self.cache.put(cell_id, curves)
        self._imean_cache[cell_id] = imean
        return curves

    def preload_all(self, verbose: bool = False) -> None:
        """把本数据集涉及的所有电池曲线一次性载入内存缓存。

        训练时若 LRU 容量小于电池数，随机采样会反复踢出/重载 .mat
        （实测单只电池加载约 3-5 秒，会让每 batch 从 0.3s 涨到 6s）。
        全部电池约 2GB 内存，换来稳定的训练速度；内存紧张时请用小
        cache_size 且不要调用本方法。
        """
        cells = self.windows["cell_id"].unique()
        for cid in cells:
            self._cell_curves(cid)
        if verbose:
            print(f"[dataset] 已预载 {len(cells)} 只电池曲线", flush=True)

    def __getitem__(self, idx: int) -> dict:
        row = self.windows.iloc[idx]
        cell_id = row["cell_id"]
        p = int(row["pos"])

        curves = self._cell_curves(cell_id)
        if p + 1 + self.H > curves.shape[0]:
            raise IndexError(f"{cell_id} pos={p} 超出曲线范围")

        s = self._labels[cell_id]
        u = self._imean_cache[cell_id][p]           # 最后观测循环 k 的 I_mean
        if not np.isfinite(u):                      # fail fast：防御 NaN 流入训练
            raise ValueError(f"{cell_id} pos={p} 的 I_mean 无效（该循环无充电段）")
        ir_k = self._ir[cell_id][p]                 # 当前循环内阻 R_last
        X = curves[p - self.W + 1: p + 1]           # (W, 3, Tmax)
        if self.normalizer is not None:
            X = self.normalizer.transform(X)
        return {
            "X": X,
            "u": u,
            "ir_0": self._ir0[cell_id],
            "ir_k": ir_k,
            "y_cur": s[p],
            "y_fut": s[p + 1: p + self.H + 1],        # (H,)
            "stage": row["stage"],
            "cell_id": cell_id,
            "k": float(row["k"]),
            "pos": p,
        }


def make_stage_weights(windows: pd.DataFrame,
                       stage_boost: dict[str, float] | None = None
                       ) -> np.ndarray:
    """按 (batch, stage) 的逆频率给每个窗口一个采样权重。

    用途：训练时配合 WeightedRandomSampler，让各老化阶段在每批内均衡。
    权重 = 1 / 该窗口在 (batch, stage) 组内的占比，再整体归一化。

    stage_boost：在逆频率之上额外放大某些阶段的权重，例如
      {"s3_aged": 8.0, "s2_mild": 3.0}
    深度老化样本（s3/s4）稀缺时，纯逆频率仍可能不够；放大后每个 epoch
    会更多次抽到老化窗口，代价是健康窗口被相对忽略（可调节倍数折中）。

    注意：只应传入训练集窗口（验证/测试不做重采样）。
    """
    win = windows.copy()
    win["_grp"] = win["batch"] + "|" + win["stage"]
    freq = win["_grp"].value_counts(normalize=True)
    weights = 1.0 / win["_grp"].map(freq).to_numpy(dtype=np.float64)
    if stage_boost:
        boost = win["stage"].map(
            lambda s: stage_boost.get(s, 1.0)).to_numpy(dtype=np.float64)
        weights = weights * boost
    return weights / weights.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=Path,
                        default=ROOT / "data/processed/matr_windows.parquet")
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "data/processed/matr_soh_labels.parquet")
    parser.add_argument("--mat-dir", type=Path,
                        default=ROOT / "data/external/matr")
    parser.add_argument("--n", type=int, default=3, help="演示取几个窗口")
    args = parser.parse_args()

    windows = pd.read_parquet(args.windows)
    labels = pd.read_parquet(args.labels)
    ds = WindowDataset(windows, labels, args.mat_dir)

    print(f"数据集共 {len(ds):,} 个窗口（{ds.windows['cell_id'].nunique()} 只电池）")
    for i in range(min(args.n, len(ds))):
        s = ds[i]
        print(f"\n窗口 {i}: cell={s['cell_id']} k={s['k']:.0f} stage={s['stage']}")
        print(f"  X: {s['X'].shape} {s['X'].dtype}  "
              f"V[{s['X'][:, 0].min():.3f},{s['X'][:, 0].max():.3f}] "
              f"I[{s['X'][:, 1].min():.3f},{s['X'][:, 1].max():.3f}] "
              f"T[{s['X'][:, 2].min():.2f},{s['X'][:, 2].max():.2f}]")
        print(f"  y_cur={s['y_cur']:.4f}  y_fut 头={np.round(s['y_fut'][:3], 4)} "
              f"尾={np.round(s['y_fut'][-3:], 4)}")

    w = make_stage_weights(windows)
    print(f"\n逆频率采样权重: min={w.min():.2e} max={w.max():.2e} sum={w.sum():.3f}")


if __name__ == "__main__":
    main()
