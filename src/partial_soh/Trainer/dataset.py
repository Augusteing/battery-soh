"""partial_soh 训练数据集（方案 A：惰性加载）。

本模块把 `partial_segments_index.parquet`（片段索引）包装成 PyTorch Dataset。

核心思想：

    索引表只存“几何 + 标签”，不存曲线。训练时按需做三件事：

        1. 根据 (cell_id, cycle_index) 找到原始 MAT 中的那个循环；
        2. 提取充电段（I > 0）；
        3. 把该片段 [start_ah, end_ah] 上的 V/I/T 插值到固定 101 点容量网格，
           得到模型输入 x ∈ R^(101 x 3)。

    三个输入通道依次是：电流 I、电压 V、容量坐标 Q。

因为 51 个片段共享同一个循环，重复读 MAT 会很慢，所以本模块用 LRU 缓存
把“已读过的充电曲线”缓存在内存里。这样每个循环只读一次，后续 50 个片段
直接复用缓存。
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 让本文件能 import 到 DataLoader 里的模块。
ROOT = Path(__file__).resolve().parents[3]
DL_DIR = ROOT / "src" / "partial_soh" / "DataLoader"
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from mat_io import discover_batch_files, read_raw_cycle_from_file  # noqa: E402
from charge import extract_charge_curve  # noqa: E402
from segments import interpolate_segment, NOMINAL_CAPACITY_AH  # noqa: E402


# ---------------------------------------------------------------------------
# 进程内共享充电曲线缓存
# ---------------------------------------------------------------------------
# 同一个训练进程里会创建多个 Dataset 实例（如 soh 任务与 pretrain 任务），
# 它们需要读取的是同一批充电曲线。如果不共享缓存，每条曲线会被重复
# 读取/提取一次，预加载时间翻倍。这里用模块级 LRU 缓存存一份，
# 所有实例共用；上限设成能容纳 train + test 全部唯一曲线的数量级，
# 非预加载模式下也不会无限增长。
# ---------------------------------------------------------------------------
_SHARED_CACHE_MAX = 200_000
_SHARED_CHARGE_CACHE: OrderedDict[
    tuple[str, int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
] = OrderedDict()


def _shared_cache_get(
    key: tuple[str, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """取共享缓存；命中时把 key 移到队尾（LRU 语义）。"""
    value = _SHARED_CHARGE_CACHE.get(key)
    if value is not None:
        _SHARED_CHARGE_CACHE.move_to_end(key)
    return value


def _shared_cache_put(
    key: tuple[str, int, int],
    value: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """写入共享缓存；超过上限时淘汰最久未使用的条目。"""
    _SHARED_CHARGE_CACHE[key] = value
    _SHARED_CHARGE_CACHE.move_to_end(key)
    while len(_SHARED_CHARGE_CACHE) > _SHARED_CACHE_MAX:
        _SHARED_CHARGE_CACHE.popitem(last=False)


def _cell_index_from_id(cell_id: str) -> int:
    """从 cell_id（如 2017-06-30_c000）解析批次内 0-based 下标。"""
    suffix = cell_id.rsplit("_", 1)[1]
    if not suffix.startswith("c"):
        raise ValueError(f"cell_id 格式错误: {cell_id}")
    return int(suffix[1:])


def _batch_from_id(cell_id: str) -> str:
    """从 cell_id 中提取批次名（如 2017-06-30）。"""
    return cell_id.rsplit("_", 1)[0]


class PartialSohDataset(Dataset):
    """按片段索引惰性生成 (输入, 标签) 的 Dataset。"""

    def __init__(
        self,
        index_path: Path,
        mat_dir: Path,
        split: str = "train",
        task: str = "soh",
        cache_size: int | None = 8192,
        preload: bool = False,
    ) -> None:
        """初始化数据集。

        参数
        ----
        index_path : partial_segments_index.parquet 路径。
        mat_dir    : 原始 .mat 文件目录。
        split      : 只保留哪个划分（train / test）。
        task       : "soh" 返回 SOH 回归样本；
                     "pretrain" 返回电压预测样本。
        cache_size : 充电曲线 LRU 缓存大小（按循环数计）。
        preload    : True 时在初始化阶段把本数据集需要的全部充电曲线
                     一次性读进内存，后续不再读 MAT。适合全量训练。
        """
        if task not in ("soh", "pretrain"):
            raise ValueError(f"task 必须是 'soh' 或 'pretrain'，得到 {task}")

        self.task = task
        self.mat_dir = Path(mat_dir)
        self.files = discover_batch_files(self.mat_dir)
        # 一次性打开所有批次文件并复用句柄，避免每条曲线都重新 open/close。
        # 这是全量预加载的主要提速点。
        self._handles: dict[str, h5py.File] = {
            batch: h5py.File(str(path), "r") for batch, path in self.files.items()
        }

        index = pd.read_parquet(index_path)
        index = index[index["split"] == split].copy()

        # 两个任务对片段合法性的要求不同：
        #   SOH 任务只需要观测窗口存在；
        #   预训练还需要 7% 预测窗口存在。
        valid_col = "is_valid_soh" if task == "soh" else "is_valid_pretrain"
        index = index[index[valid_col]].copy()

        # 把列抽成 numpy 数组，避免 __getitem__ 里反复做 pandas 行访问。
        self.cell_ids = index["cell_id"].to_numpy(dtype=str)
        self.cycle_indices = index["cycle_index"].to_numpy(dtype=np.int64)
        self.start_ahs = index["start_ah"].to_numpy(dtype=np.float32)
        self.end_ahs = index["end_ah"].to_numpy(dtype=np.float32)
        self.pred_start_ahs = index["pred_start_ah"].to_numpy(dtype=np.float32)
        self.pred_end_ahs = index["pred_end_ah"].to_numpy(dtype=np.float32)
        self.soh_nominals = index["soh_nominal"].to_numpy(dtype=np.float32)

        # 同循环一致性需要知道“哪些片段属于同一个循环”。
        # 用 (cell_id, cycle_index) 分组并编号 0, 1, 2, ...，
        # 同一循环的所有片段共享同一个编号。
        self._group_ids = (
            index.groupby(["cell_id", "cycle_index"], sort=False)
            .ngroup()
            .to_numpy(dtype=np.int64)
        )

        # 预加载需要把全部曲线都留在缓存里，所以关闭 LRU 上限。
        if preload:
            cache_size = None

        # 把“读充电曲线”这一步包一层缓存。注意 lru_cache 要求参数可哈希，
        # 这里传的都是字符串 / 整数，没问题。
        self._load_charge = lru_cache(maxsize=cache_size)(self._read_charge_curve)

        if preload:
            self._preload_charges()
            # 预加载完成后所有曲线都在内存里，文件句柄可以关掉。
            self.close()

    def _preload_charges(self) -> None:
        """把本数据集涉及的所有 (cell_id, cycle_index) 充电曲线读进缓存。

        随机打乱后，同一循环的 51 个片段会散落在整个 epoch 里；如果只靠
        小容量 LRU 缓存，绝大多数访问都会回落到 MAT 读，训练极慢。预加载
        用约 1 GB 内存换掉这个瓶颈，但“插值”仍然按需进行，符合方案 A。
        """
        unique = sorted(set(zip(self.cell_ids.tolist(), self.cycle_indices.tolist())))
        print(f"[dataset] 预加载 {len(unique)} 条充电曲线 ...")
        for pos, (cell_id, cycle_index) in enumerate(unique, start=1):
            self._load_charge(cell_id, int(cycle_index))
            if pos % 5000 == 0:
                print(f"[dataset] 预加载进度 {pos}/{len(unique)}")

    def _read_charge_curve(
        self, cell_id: str, cycle_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """真正读取一个循环的充电曲线（未缓存的核心函数）。"""
        batch = _batch_from_id(cell_id)
        cell_index = _cell_index_from_id(cell_id)
        if batch not in self.files:
            raise FileNotFoundError(f"找不到批次文件: {batch}")

        # 先查进程内共享缓存：soh 任务读过的曲线，pretrain 任务直接复用。
        key = (batch, cell_index, int(cycle_index))
        cached = _shared_cache_get(key)
        if cached is not None:
            return cached

        raw = read_raw_cycle_from_file(
            self._handles[batch], cell_index, int(cycle_index)
        )
        charge = extract_charge_curve(raw)
        value = (
            np.asarray(charge["t"], dtype=np.float32),
            np.asarray(charge["V"], dtype=np.float32),
            np.asarray(charge["I"], dtype=np.float32),
            np.asarray(charge["T"], dtype=np.float32),
            np.asarray(charge["Qc"], dtype=np.float32),
        )
        _shared_cache_put(key, value)
        return value

    def close(self) -> None:
        """关闭所有已打开的 MAT 文件句柄（幂等）。"""
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass

    def _interpolate_window(
        self,
        charge: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        start_ah: float,
        end_ah: float,
    ) -> np.ndarray:
        """把充电曲线插值到 [start_ah, end_ah]，返回 (n_points, 3) 输入张量。"""
        t, v, i, temp, qc = charge
        seg = interpolate_segment(
            {"t": t, "V": v, "I": i, "T": temp, "Qc": qc},
            start_ah=float(start_ah),
            end_ah=float(end_ah),
        )
        # 通道顺序：[I, V, Q]。第三通道是容量坐标，不是温度。
        x = np.stack([seg["I"], seg["V"], seg["capacity"]], axis=1).astype(np.float32)
        return x

    def __len__(self) -> int:
        return len(self.cell_ids)

    def group_ids(self) -> np.ndarray:
        """返回每个样本所属循环的编号（int64 数组）。

        同一个 (cell_id, cycle_index) 的所有片段共享同一个编号，
        SameCycleBatchSampler 靠它把同循环片段聚到同一批次。
        """
        return self._group_ids

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (x, y)。

        x : (101, 3) 的 float32 张量；
        y : 取决于 task——
              soh      -> 标量 SOH，形状 ()；
              pretrain -> 下一步电压 V[1:101]，形状 (100,)。
                         这是“预测下一时刻电压”的密集监督目标。
        """
        cell_id = self.cell_ids[idx]
        cycle_index = int(self.cycle_indices[idx])
        charge = self._load_charge(cell_id, cycle_index)

        x = self._interpolate_window(
            charge, self.start_ahs[idx], self.end_ahs[idx]
        )

        if self.task == "soh":
            y = np.asarray(self.soh_nominals[idx], dtype=np.float32)
        else:
            # 预训练目标：把观测窗内的电压序列向后移一位，作为“下一步电压”。
            # x 的第 1 通道是电压，因此 y = V[1:101]。
            y = x[1:, 1]

        return torch.from_numpy(x), torch.from_numpy(y)


class MemmapSohDataset(Dataset):
    """从磁盘 memmap 缓存直接读取片段（训练提速用）。

    背景
    ----
    片段是静态数据：同一个片段每次插值结果完全一样。`build_cache.py`
    一次性把全部片段插值好写入磁盘，本类只做“按行切片 + 转张量”。
    训练时 CPU 几乎不干活，瓶颈回到 GPU。

    与 PartialSohDataset 的区别：
      - 不做 MAT 读取、不做插值（都已在构建缓存时完成）；
      - 两个任务（soh / pretrain）共享同一份输入，只在 __getitem__
        里按任务返回不同标签。
    """

    def __init__(self, cache_dir: Path, split: str, task: str = "soh") -> None:
        if task not in ("soh", "pretrain"):
            raise ValueError(f"task 必须是 'soh' 或 'pretrain'，得到 {task}")
        self.task = task
        cache_dir = Path(cache_dir)

        meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        shape = tuple(int(v) for v in meta[f"shape_{split}"])  # (N, 101, 3)
        self._x = np.memmap(
            str(cache_dir / f"X_{split}.npy"),
            dtype=np.float32,
            mode="r",
            shape=shape,
        )
        self._y = np.load(cache_dir / f"y_{split}.npy")
        self._pretrain_mask = np.load(cache_dir / f"is_valid_pretrain_{split}.npy")
        self._group_ids = np.load(cache_dir / f"group_ids_{split}.npy")

        if task == "pretrain":
            # 预训练任务只需要“拥有完整 7% 预测窗口”的片段。
            self._valid = np.flatnonzero(self._pretrain_mask)
        else:
            # SOH 任务：缓存里全是 is_valid_soh 的片段，全部可用。
            self._valid = np.arange(len(self._y))

    def __len__(self) -> int:
        return len(self._valid)

    def group_ids(self) -> np.ndarray:
        """返回每个样本所属循环的编号（int64）。

        与 PartialSohDataset.group_ids 语义一致，供同循环一致性采样使用。
        """
        return self._group_ids

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (x, y)，形状与 PartialSohDataset 完全一致。"""
        row = int(self._valid[idx])
        # np.array(...) 会拷贝一份普通可写 ndarray，
        # 避免 torch.from_numpy 对只读 memmap 的警告。
        x = torch.from_numpy(np.array(self._x[row]))  # (101, 3)
        if self.task == "soh":
            y = torch.from_numpy(np.asarray(self._y[row]))  # 标量 SOH
        else:
            # 与 PartialSohDataset 一致：下一步电压 V[1:101]。
            y = x[1:, 1]
        return x, y

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """批量读取（PyTorch 2.1+ 的 DataLoader 会优先调用本方法）。

        作用
        ----
        默认情况下 DataLoader 会逐个调用 __getitem__ 取 4096 个样本，
        每个样本都有一次 Python 层开销（约 30μs），合计每步 100ms+，
        CPU 成为瓶颈。这里用 numpy 的 fancy indexing 一次性取出整个
        batch（高级索引本身就会拷贝），把 4096 次调用降成 1 次。

        返回
        ----
        样本列表 [(x_i, y_i), ...]，交给 default_collate 组装成
        (B, 101, 3) 与 (B,)（或 (B, 100)）张量。
        """
        rows = self._valid[np.asarray(indices)]
        x = torch.from_numpy(np.array(self._x[rows]))  # (B, 101, 3)
        if self.task == "soh":
            y = torch.from_numpy(np.array(self._y[rows]))  # (B,)
        else:
            y = x[:, 1:, 1]  # (B, 100)
        return [(x[i], y[i]) for i in range(len(rows))]


if __name__ == "__main__":
    """冒烟测试：取 train 前 3 个样本，打印形状与数值。"""
    import argparse

    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "processed" / "partial_segments_index.parquet")
    parser.add_argument("--mat-dir", type=Path, default=ROOT / "data" / "external" / "matr")
    parser.add_argument("--task", choices=("soh", "pretrain"), default="soh")
    args = parser.parse_args()

    ds = PartialSohDataset(args.index, args.mat_dir, split="train", task=args.task)
    print(f"task={args.task}, 样本数={len(ds)}")
    for i in range(3):
        x, y = ds[i]
        print(f"\n样本 {i}:")
        print("  x.shape =", tuple(x.shape), " dtype =", x.dtype)
        print("  y.shape =", tuple(y.shape), " dtype =", y.dtype)
        print("  x[0]   =", x[0].numpy())
        print("  y      =", y.numpy())
