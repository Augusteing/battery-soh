"""temperature_soh 训练数据集（方案 A：惰性加载，支持温度标量）。

本模块把 `segment_index.parquet`（统一片段索引表）包装成 PyTorch Dataset。
与 partial_soh 的 PartialSohDataset 的区别：

  1. 输入是 **4 通道** `(I, V, Q, T')`，T 是温度曲线；
  2. 温度采用物理量纲归一化 `T' = (T - 25) / 10`，**不用全体 z-score**
     （z-score 会被 30°C 恒温主导，失去温度的物理含义）。

温度嵌入（SOH 决策层融合）支持：

  - SOH 任务额外返回循环级温度标量（摄氏度标量），供模型的
    TemperatureEmbedding 使用（对齐 arXiv 2504.00393）；
  - 标量来源 = 片段温度曲线均值（Severson 恒温数据上等价于
    循环平均自热温度）；未来接入 SIT/电科院等有环境温度的
    数据集时，可替换为真实的工况温度。

核心思想（沿用方案 A 惰性加载）：

    索引表只存“几何 + 标签”，不存曲线。训练时按需：
      1. 根据 (cell_id, cycle_index) 找到原始 MAT 中的那个循环；
      2. 提取充电段（I > 0），统一单位（C-rate -> A、分钟 -> 秒）；
      3. 把该片段 [start_ah, end_ah] 上的 I/V/Q/T 插值到 101 点容量网格，
         得到模型输入 x ∈ R^(101 x 4)。

通道顺序（与 Trainer 约定）：[I, V, Q, T']。

性能：51 个片段共享同一个循环，重复读 MAT 会很慢，因此用 LRU 缓存
把“已读过的充电曲线”缓存在内存里（每个循环只读一次，之后复用）。
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 让本文件能 import 到 temperature_soh/DataLoader 里的模块。
ROOT = Path(__file__).resolve().parents[3]
DL_DIR = ROOT / "src" / "temperature_soh" / "DataLoader"
if str(DL_DIR) not in sys.path:
    sys.path.insert(0, str(DL_DIR))

from mat_io import (  # noqa: E402
    convert_cycle_to_unified,
    discover_batch_files,
    read_raw_cycle_from_file,
)
from segments import interpolate_segment  # noqa: E402


# 温度归一化常数（物理量纲，非统计量纲）。
# Severson 电池在 30°C 恒温箱，温度曲线因自热在 26~42°C 之间波动。
# 以 25°C 为物理零点、10°C 为尺度，T' ≈ 0.1~1.7，量纲含义清晰。
TEMP_CENTER_C = 25.0
TEMP_SCALE_C = 10.0

# 通道顺序：与 Trainer/model.py 的 input_dim=4 约定一致。
CHANNEL_NAMES = ("I", "V", "Q", "T")


# ---------------------------------------------------------------------------
# 进程内共享数组缓存（memmap 模式用）
# ---------------------------------------------------------------------------
# soh 任务和 pretrain 任务需要同一份 X_train（约 5 GB）。
# 如果各自 np.fromfile 一份，双份拷贝实测会导致内存换页、训练卡死。
# 这里按 (split, kind) 只加载一次，所有 Dataset 实例共用。
_ARRAY_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _load_shared_array(
    cache_dir: Path, split: str, kind: str, n_rows: int, channels: int
) -> np.ndarray:
    """按 (split, kind, channels) 加载一次并缓存；kind: "x" 或 "future"。"""
    key = (split, kind, channels)
    arr = _ARRAY_CACHE.get(key)
    if arr is None:
        name = "X" if kind == "x" else "X_future"
        # 磁盘缓存固定是 4 通道（101 或 36 点），先按完整形状加载再切片。
        n_points = 101 if kind == "x" else 36
        arr = np.fromfile(
            cache_dir / f"{name}_{split}.npy", dtype=np.float32
        ).reshape(n_rows, n_points, 4)
        # 3 通道对照：缓存是 4 通道，只需切掉温度通道（最后一位）。
        if channels < arr.shape[-1]:
            arr = arr[..., :channels]
        _ARRAY_CACHE[key] = arr
    return arr


def _cell_index_from_id(cell_id: str) -> int:
    """从 cell_id（如 2017-06-30_c000）解析批次内 0-based 下标。"""
    suffix = cell_id.rsplit("_", 1)[1]
    if not suffix.startswith("c"):
        raise ValueError(f"cell_id 格式错误: {cell_id}")
    return int(suffix[1:])


def _batch_from_id(cell_id: str) -> str:
    """从 cell_id 中提取批次名（如 2017-06-30）。"""
    return cell_id.rsplit("_", 1)[0]


def _normalize_temperature(t: np.ndarray) -> np.ndarray:
    """物理量纲归一化：T' = (T - 25) / 10。"""
    return (t - TEMP_CENTER_C) / TEMP_SCALE_C


class TemperatureSohDataset(Dataset):
    """按片段索引惰性生成 (输入, 标签[, 温度标量]) 的 Dataset。"""

    def __init__(
        self,
        index_path: Path,
        mat_dir: Path,
        split: str = "train",
        task: str = "soh",
        cache_size: int | None = 8192,
        preload: bool = False,
        channels: int = 4,
    ) -> None:
        """初始化数据集。

        参数
        ----
        index_path : temperature_soh 的 segment_index.parquet 路径。
        mat_dir    : 原始 .mat 文件目录。
        split      : 只保留哪个划分（train / val / test）。
        task       : "soh" 返回 SOH 回归样本；
                     "pretrain" 返回电压预测样本。
        cache_size : 充电曲线 LRU 缓存大小（按循环数计）。
        preload    : True 时把本数据集需要的全部充电曲线一次性读进内存。
        """
        if task not in ("soh", "pretrain"):
            raise ValueError(f"task 必须是 'soh' 或 'pretrain'，得到 {task}")
        if channels not in (3, 4):
            raise ValueError(f"channels 必须是 3 或 4，得到 {channels}")

        self.task = task
        self.channels = channels
        self.mat_dir = Path(mat_dir)
        self.files = discover_batch_files(self.mat_dir)
        # 一次性打开所有批次文件并复用句柄，避免每条曲线都重新 open/close。
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
        self.soh_values = index["soh"].to_numpy(dtype=np.float32)

        # 同循环一致性需要知道“哪些片段属于同一个循环”。
        self._group_ids = (
            index.groupby(["cell_id", "cycle_index"], sort=False)
            .ngroup()
            .to_numpy(dtype=np.int64)
        )

        if preload:
            cache_size = None
        self._load_charge = lru_cache(maxsize=cache_size)(self._read_charge_curve)

        if preload:
            self._preload_charges()
            self.close()

    def _preload_charges(self) -> None:
        """把本数据集涉及的所有充电曲线读进缓存（预加载模式）。"""
        unique = sorted(set(zip(self.cell_ids.tolist(), self.cycle_indices.tolist())))
        print(f"[dataset] 预加载 {len(unique)} 条充电曲线 ...", flush=True)
        for pos, (cell_id, cycle_index) in enumerate(unique, start=1):
            self._load_charge(cell_id, int(cycle_index))
            if pos % 5000 == 0:
                print(f"[dataset] 预加载进度 {pos}/{len(unique)}", flush=True)

    def _read_charge_curve(
        self, cell_id: str, cycle_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """读取一个循环的充电曲线（统一单位），返回 (t, V, I, T, Qc)。"""
        batch = _batch_from_id(cell_id)
        cell_index = _cell_index_from_id(cell_id)
        if batch not in self.files:
            raise FileNotFoundError(f"找不到批次文件: {batch}")

        raw = read_raw_cycle_from_file(
            self._handles[batch], cell_index, int(cycle_index)
        )
        # 统一单位：I C-rate -> A（×1.1），t 分钟 -> 秒（×60）。
        cycle = convert_cycle_to_unified(raw, cycle_number=int(cycle_index))

        # 提取充电段（I > 0）。temperature_soh 的 segments 统一结构版，
        # 返回 dict 含 t/V/I/Qc/T。
        charge = {
            "t": np.asarray(cycle["time_in_s"], dtype=float),
            "V": np.asarray(cycle["voltage_in_V"], dtype=float),
            "I": np.asarray(cycle["current_in_A"], dtype=float),
            "Qc": np.asarray(cycle["charge_capacity_in_Ah"], dtype=float),
            "T": np.asarray(cycle["temperature_in_C"], dtype=float),
        }
        mask = charge["I"] > 0.0
        for key in charge:
            charge[key] = charge[key][mask]

        if charge["V"].size < 2:
            raise ValueError(
                f"{cell_id} cycle {cycle_index} 充电段点数不足"
            )

        return (
            np.asarray(charge["t"], dtype=np.float32),
            np.asarray(charge["V"], dtype=np.float32),
            np.asarray(charge["I"], dtype=np.float32),
            np.asarray(charge["T"], dtype=np.float32),
            np.asarray(charge["Qc"], dtype=np.float32),
        )

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
        """把充电曲线插值到 [start_ah, end_ah]，返回 (n_points, 4) 输入。"""
        t, v, i, temp, qc = charge
        seg = interpolate_segment(
            {"t": t, "V": v, "I": i, "T": temp, "Qc": qc},
            start_ah=float(start_ah),
            end_ah=float(end_ah),
        )
        # 通道顺序：[I, V, Q, T']。T 归一化到 (T-25)/10。
        x = np.stack(
            [
                seg["I"],
                seg["V"],
                seg["capacity"],
                _normalize_temperature(seg["T"]),
            ],
            axis=1,
        ).astype(np.float32)
        # 3 通道对照：只保留 [I, V, Q]，切掉温度通道。
        if self.channels < x.shape[1]:
            x = x[:, : self.channels]
        return x

    def _interpolate_pred_window(
        self,
        charge: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        pred_start_ah: float,
        pred_end_ah: float,
    ) -> np.ndarray:
        """把未来 7% 预测窗口插值到等距容量网格，返回 (36, 4)。"""
        return self._interpolate_window(charge, pred_start_ah, pred_end_ah)

    def __len__(self) -> int:
        return len(self.cell_ids)

    def group_ids(self) -> np.ndarray:
        """返回每个样本所属循环的编号（int64 数组）。"""
        return self._group_ids

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (x, y[, 温度标量])。

        x : (101, 4) 的 float32 张量 [I, V, Q, T']；
        y : 取决于 task——
              soh      -> 标量 SOH，形状 ()；
              pretrain -> (下一步电压 V[1:101] (100,), 未来窗 x_future (36, 4))；
        soh 任务额外返回温度标量：本循环充电段的平均温度（摄氏度）。
        """
        cell_id = self.cell_ids[idx]
        cycle_index = int(self.cycle_indices[idx])
        charge = self._load_charge(cell_id, cycle_index)

        x = self._interpolate_window(
            charge, self.start_ahs[idx], self.end_ahs[idx]
        )

        if self.task == "soh":
            y = np.asarray(self.soh_values[idx], dtype=np.float32)
            # 温度标量 = 充电段平均温度（摄氏度）。
            # charge 是 (t, V, I, T, Qc) 元组，下标 3 为温度曲线。
            temp_celsius = np.asarray(np.mean(charge[3]), dtype=np.float32)
            return (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(temp_celsius),
            )
        else:
            # 预训练目标：观测窗内电压序列后移一位作为“下一步电压”。
            y = x[1:, 1]
            x_future = self._interpolate_pred_window(
                charge, self.pred_start_ahs[idx], self.pred_end_ahs[idx]
            )
            return (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(x_future),
            )

        return torch.from_numpy(x), torch.from_numpy(y)


class MemmapSohDataset(Dataset):
    """从磁盘 memmap 缓存直接读取 4 通道片段（训练提速用）。

    背景
    ----
    片段是静态数据：同一个片段每次插值结果完全一样。`build_cache.py`
    一次性把全部片段插值好写入磁盘，本类只做“按行切片 + 转张量”。
    训练时 CPU 几乎不干活，瓶颈回到 GPU。

    与 TemperatureSohDataset 的区别：
      - 不做 MAT 读取、不做插值（都已在构建缓存时完成）；
      - 两个任务（soh / pretrain）共享同一份输入，只在 __getitem__
        里按任务返回不同标签。
    """

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        task: str = "soh",
        in_memory: bool = True,
        channels: int = 4,
    ) -> None:
        if task not in ("soh", "pretrain"):
            raise ValueError(f"task 必须是 'soh' 或 'pretrain'，得到 {task}")
        if channels not in (3, 4):
            raise ValueError(f"channels 必须是 3 或 4，得到 {channels}")
        self.task = task
        self.channels = channels
        cache_dir = Path(cache_dir)

        meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        shape = tuple(int(v) for v in meta[f"shape_{split}"])  # (N, 101, 4)
        shape = shape[:-1] + (channels,)  # 按需要切成 (N, 101, 3) 或 (N, 101, 4)
        if in_memory:
            # 一次性把整个片段矩阵读进 RAM（约 5 GB），多个 Dataset 共享
            # 同一份数组，避免 soh/pretrain 各复制一份导致内存翻倍。
            self._x = _load_shared_array(
                cache_dir, split, "x", shape[0], channels
            )
        else:
            self._x = np.memmap(
                str(cache_dir / f"X_{split}.npy"),
                dtype=np.float32,
                mode="r",
                shape=(shape[0], 101, 4),
            )[..., :channels]
        self._y = np.load(cache_dir / f"y_{split}.npy")
        self._pretrain_mask = np.load(cache_dir / f"is_valid_pretrain_{split}.npy")
        self._group_ids = np.load(cache_dir / f"group_ids_{split}.npy")
        # 循环级温度标量（摄氏度）：SOH 决策层温度嵌入用。
        # 若缓存里还没有，则从完整 4 通道缓存流式生成并落盘。
        self._temp_scalars = self._load_or_build_temp_scalars(cache_dir, split, shape[0])
        self._x_future: np.ndarray | None = None
        if task == "pretrain":
            # 未来 7% 预测窗（36 点），只在预训练任务需要时载入。
            future_shape = tuple(int(v) for v in meta[f"shape_future_{split}"])
            future_shape = future_shape[:-1] + (channels,)
            if in_memory:
                self._x_future = _load_shared_array(
                    cache_dir, split, "future", future_shape[0], channels
                )
            else:
                self._x_future = np.memmap(
                    str(cache_dir / f"X_future_{split}.npy"),
                    dtype=np.float32,
                    mode="r",
                    shape=(future_shape[0], 36, 4),
                )[..., :channels]

        if task == "pretrain":
            # 预训练任务只需要“拥有完整 7% 预测窗口”的片段。
            self._valid = np.flatnonzero(self._pretrain_mask)
        else:
            # SOH 任务：缓存里全是 is_valid_soh 的片段，全部可用。
            self._valid = np.arange(len(self._y))

    def __len__(self) -> int:
        return len(self._valid)

    def group_ids(self) -> np.ndarray:
        """返回每个样本所属循环的编号（int64）。"""
        return self._group_ids

    @staticmethod
    def _load_or_build_temp_scalars(
        cache_dir: Path, split: str, n_rows: int
    ) -> np.ndarray:
        """读取（或首次生成）片段级温度标量数组（摄氏度, float32）。

        生成方式：从磁盘缓存 X_{split}.npy 的第 4 列（归一化温度 T'）
        按片段求平均，再换算回摄氏度：T = T' * 10 + 25。

        注意：这里用 memmap 读取，只加载温度列，不把整份 6.5GB
        缓存读进内存；生成结果（约 16MB）落盘后下次直接读。
        """
        path = cache_dir / f"temp_scalars_{split}.npy"
        if path.exists():
            return np.load(path)
        full = np.memmap(
            str(cache_dir / f"X_{split}.npy"),
            dtype=np.float32,
            mode="r",
            shape=(n_rows, 101, 4),
        )
        # memmap 按需读取：温度列每行 404 字节，共 n_rows 行。
        t_norm_mean = full[..., 3].mean(axis=1)
        t_celsius = (t_norm_mean * TEMP_SCALE_C + TEMP_CENTER_C).astype(np.float32)
        np.save(path, t_celsius)
        print(f"[dataset] 温度标量已生成并落盘: {path} "
              f"(范围 {t_celsius.min():.2f}~{t_celsius.max():.2f}°C)", flush=True)
        return t_celsius

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (x, y[, 温度标量])，与 TemperatureSohDataset 一致。"""
        row = int(self._valid[idx])
        x = torch.from_numpy(np.array(self._x[row]))  # (101, 4)
        if self.task == "soh":
            y = torch.from_numpy(np.asarray(self._y[row]))  # 标量 SOH
            temp_celsius = torch.from_numpy(
                np.asarray(self._temp_scalars[row], dtype=np.float32)
            )
            return x, y, temp_celsius
        else:
            y = x[1:, 1]  # 下一步电压 V[1:101]
            x_future = torch.from_numpy(np.array(self._x_future[row]))  # (36, 4)
            return x, y, x_future
        return x, y

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """批量读取（PyTorch 2.1+ 的 DataLoader 会优先调用本方法）。

        用 numpy 的 fancy indexing 一次性取出整个 batch，
        把 4096 次 Python 层调用降成 1 次。
        配合 trainer 里的 identity collate 使用。
        """
        rows = self._valid[np.asarray(indices)]
        # fancy indexing 本身就会拷贝出新数组；in_memory 模式下
        # _x 是普通可写数组，无需再包一层 np.array()。
        x = torch.from_numpy(self._x[rows])  # (B, 101, 4)
        if self.task == "soh":
            y = torch.from_numpy(np.asarray(self._y[rows]))  # (B,)
            temp_celsius = torch.from_numpy(
                np.asarray(self._temp_scalars[rows], dtype=np.float32)
            )  # (B,)
            return x, y, temp_celsius
        else:
            y = x[:, 1:, 1]  # (B, 100)
            x_future = torch.from_numpy(self._x_future[rows])  # (B, 36, 4)
            return x, y, x_future
