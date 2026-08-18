"""特征标准化模块（M3 · 第二步）：按通道 z-score，只在训练集拟合。

z-score（标准化）公式：x' = (x - mean) / std
  - 每个通道（V / I / T）单独计算 mean 与 std；
  - 结果使每个通道的分布近似"均值 0、标准差 1"，消除不同物理单位
    （V、C-rate、°C）带来的量纲差异，利于神经网络梯度下降。

防泄漏关键：mean / std 属于"训练参数"，只在训练集上计算；
验证集与测试集必须复用训练集保存的参数，绝不能用全集重算。

设计说明（软件工程）
---------------------
- 单一职责：本模块只管"拟合参数 + 变换 + 存取"，不管窗口/曲线读取
  （曲线读取复用 dataset.load_cell_curves）；
- 流式累加：用 sum / sum-of-squares 一次遍历完成，不把全部曲线
  载入内存（训练集 97 只电池约 1GB，也不在内存里翻倍）；
- 可逆：提供 inverse_transform，方便评估阶段把预测还原回原始量纲；
- 幂等：save / load 一次成型，重复运行结果一致。

用法:
    python "src/world_model/Trainer/normalize.py"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/world_model/Trainer"))
sys.path.insert(0, str(ROOT / "src/world_model/DataLoader"))

import data_quality                       # noqa: E402  坏循环清单（统计时剔除）
from dataset import (WindowDataset, _batch_from_cell_id, _cell_index,  # noqa: E402
                     discover_batch_files, load_cell_curves)
from standardize_cycle import CycleStandardizer  # noqa: E402

CHANNELS = ("V", "I", "T")
EPS = 1e-6                                # 防止标准差为 0 时除零


class ChannelNormalizer:
    """按通道的 z-score 变换器（可存取、可逆）。

    参数
    ----
    mean, std : 长度 3 的数组，分别对应 V / I / T 通道
    """

    def __init__(self, mean: np.ndarray | None = None,
                 std: np.ndarray | None = None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean 与 std 必须同时提供或同时为空")

    # ---------- 拟合 ----------

    def fit(self, X: np.ndarray) -> "ChannelNormalizer":
        """从形状 (N, C, T) 的曲线数组拟合每通道 mean/std（就地返回自身）。

        C 固定为 3（V/I/T），统计在 N 和 T 两个维度上做。
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3 or X.shape[1] != 3:
            raise ValueError(f"期望形状 (N, 3, T)，实际 {X.shape}")
        n = X.shape[0] * X.shape[2]
        mean = X.mean(axis=(0, 2))
        var = X.var(axis=(0, 2))
        self.mean = mean
        self.std = np.sqrt(np.maximum(var, 0.0))
        self.std = np.maximum(self.std, EPS)
        print(f"[normalize] fit: n={n:,} 每通道 mean={mean.round(4)} std={self.std.round(4)}")
        return self

    # ---------- 变换 / 逆变换 ----------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """(X - mean) / std；输入 (..., 3, T)，通道在倒数第二维。"""
        if self.mean is None:
            raise RuntimeError("尚未 fit 或 load，无法 transform")
        X = np.asarray(X, dtype=np.float64)
        # 把 (3,) 的统计量重塑成可沿"倒数第二维"广播的形状（如 (1,3,1)）
        broadcast = (1,) * (X.ndim - 2) + (3, 1)
        z = (X - self.mean.reshape(broadcast)) / self.std.reshape(broadcast)
        return z.astype(np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """把标准化后的值还原回原始量纲（评估/绘图用）。"""
        if self.mean is None:
            raise RuntimeError("尚未 fit 或 load，无法 inverse_transform")
        Z = np.asarray(Z, dtype=np.float64)
        broadcast = (1,) * (Z.ndim - 2) + (3, 1)
        x = Z * self.std.reshape(broadcast) + self.mean.reshape(broadcast)
        return x.astype(np.float32)

    # ---------- 存取 ----------

    def save(self, path: Path) -> None:
        """把参数写成 JSON（均值/标准差各一行列表）。"""
        if self.mean is None:
            raise RuntimeError("没有参数可保存，请先 fit 或 load")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "channels": list(CHANNELS),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "note": "z-score 参数，仅在训练集拟合；验证/测试必须复用此文件",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[normalize] 参数已保存 -> {path}")

    @classmethod
    def load(cls, path: Path) -> "ChannelNormalizer":
        """从 JSON 读取参数。"""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(mean=np.asarray(payload["mean"]), std=np.asarray(payload["std"]))

    def __repr__(self) -> str:
        if self.mean is None:
            return "ChannelNormalizer(未拟合)"
        return (f"ChannelNormalizer(mean={self.mean.round(4).tolist()}, "
                f"std={self.std.round(4).tolist()})")


def fit_from_train_cells(cell_ids: list[str], mat_dir: Path,
                         mode: str = "full") -> ChannelNormalizer:
    """流式遍历训练电池的全部循环，拟合每通道 mean/std。

    用"一次遍历 + 累加 sum / sum-of-squares"避免把全部曲线同时载入内存：
        mean = sum / n
        std  = sqrt(sumsq / n - mean^2)
    统计时会剔除 data_quality.BAD_CYCLES 中的坏循环（它们已被窗口清洗排除）。
    """
    batch_files = discover_batch_files(mat_dir)
    stdz = CycleStandardizer(tmax=1000, mode=mode)
    sums = np.zeros(3, dtype=np.float64)
    sumsq = np.zeros(3, dtype=np.float64)
    n_total = 0

    for cell_id in cell_ids:
        mat_path = batch_files[_batch_from_cell_id(cell_id)]
        curves, _imean = load_cell_curves(mat_path, _cell_index(cell_id), stdz)
        curves = curves.astype(np.float64)

        # 坏循环剔除：raw 数组第 j 行 = cycle j+1（j 从 0 起）
        bad_raw_idx = {c - 1 for c in data_quality.BAD_CYCLES.get(cell_id, ())}
        keep = [j for j in range(curves.shape[0]) if j not in bad_raw_idx]
        curves = curves[keep]

        sums += curves.sum(axis=(0, 2))
        sumsq += (curves * curves).sum(axis=(0, 2))
        n_total += curves.shape[0] * curves.shape[2]

    mean = sums / n_total
    var = np.maximum(sumsq / n_total - mean * mean, 0.0)
    std = np.maximum(np.sqrt(var), EPS)
    return ChannelNormalizer(mean=mean, std=std)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", type=Path,
                        default=ROOT / "data/processed/splits.parquet")
    parser.add_argument("--windows", type=Path,
                        default=ROOT / "data/processed/matr_windows.parquet")
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "data/processed/matr_soh_labels.parquet")
    parser.add_argument("--mat-dir", type=Path,
                        default=ROOT / "data/external/matr")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data/processed/normalizer.json")
    args = parser.parse_args()

    # 训练集 = 按电池划分中的 train 电池（统计量的唯一合法来源）
    splits = pd.read_parquet(args.splits)
    train_cells = splits.loc[splits["split_by_cell"] == "train", "cell_id"].tolist()
    print(f"训练集电池: {len(train_cells)} 只")

    normalizer = fit_from_train_cells(train_cells, args.mat_dir)
    print(normalizer)
    normalizer.save(args.out)

    # 演示：取一个训练窗口，看标准化前后对比
    windows = pd.read_parquet(args.windows)
    labels = pd.read_parquet(args.labels)
    train_windows = windows[windows["cell_id"].isin(train_cells)]
    ds = WindowDataset(train_windows, labels, args.mat_dir, normalizer=normalizer)
    s = ds[0]
    print("\n演示窗口（标准化后）:")
    print(f"  V: mean={s['X'][:, 0].mean():.4f} std={s['X'][:, 0].std():.4f}")
    print(f"  I: mean={s['X'][:, 1].mean():.4f} std={s['X'][:, 1].std():.4f}")
    print(f"  T: mean={s['X'][:, 2].mean():.4f} std={s['X'][:, 2].std():.4f}")


if __name__ == "__main__":
    main()
