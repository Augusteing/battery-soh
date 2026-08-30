"""Severson 测试集逐片段预测保存脚本（实验章 5.2 绘图数据源）。

为什么需要这个脚本
------------------
trainer.py 的 evaluate_mae 只打印聚合 MAE（1.40%），不保留逐片段结果。
而 5.2 的四张图（散点 / 轨迹 / 老化阶段分桶 / 按电池误差）都需要
(cell_id, cycle_index, soh_true, soh_pred) 四列，聚合指标无法回溯。

数据流
------
    segment_index.parquet（split=test 且 is_valid_soh，按行序）
        └─ 与 cache/X_test.npy 的行序一一对应（build_cache.py 用同一过滤）
        └─ X_test[..., :3] 即 3 通道模型输入 [I, V, Q]（C-rate / V / SOC）
    模型 normalized_3ch.pt → 批量前向 → soh_pred
    输出 parquet：cell_id, cycle_index, soh_true, soh_pred

为什么可以直接用缓存：
  - 缓存的 4 通道是 build_cache.py 归一化后的输入，3 通道基线只切前 3 列；
  - 索引文件（2026-08-27）早于缓存（2026-08-28）构建，行序无漂移。

运行：
```powershell
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/eval_severson_preds.py
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
TR_DIR = Path(__file__).resolve().parent
if str(TR_DIR) not in sys.path:
    sys.path.insert(0, str(TR_DIR))

from model import TemperatureSohLSTM  # noqa: E402

DEFAULT_INDEX = ROOT / "data" / "processed" / "temperature_soh" / "segment_index.parquet"
DEFAULT_CACHE_DIR = ROOT / "data" / "processed" / "temperature_soh" / "cache"
DEFAULT_MODEL = ROOT / "models" / "temperature_soh" / "normalized_3ch.pt"
DEFAULT_OUT = ROOT / "data" / "processed" / "temperature_soh" / "severson_test_preds_3ch.parquet"


def load_test_table(index_path: Path) -> pd.DataFrame:
    """读取测试集片段索引表（与缓存同序），返回 (cell_id, cycle_index, soh_true)。

    过滤条件与 build_cache.py 完全一致：split=test 且 is_valid_soh，
    保持 parquet 原始行序——这是与 cache/X_test.npy 对齐的前提。
    """
    index = pd.read_parquet(index_path)
    table = index[
        (index["split"] == "test") & (index["is_valid_soh"])
    ].reset_index(drop=True)
    return table[["cell_id", "cycle_index", "soh"]].rename(
        columns={"soh": "soh_true"}
    )


def load_model(model_path: Path, device: torch.device) -> TemperatureSohLSTM:
    """加载 3 通道无温度的 Severson 归一化基线模型。"""
    return load_model_channels(model_path, device, channels=3)


def load_model_channels(
    model_path: Path, device: torch.device, channels: int
) -> TemperatureSohLSTM:
    """加载指定通道数的 Severson 基线模型（3=无温度，4=带温度通道）。"""
    if channels not in (3, 4):
        raise ValueError(f"channels 必须是 3 或 4，得到 {channels}")
    model = TemperatureSohLSTM(input_dim=channels, use_temp_embed=False)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict_test_set(
    model: TemperatureSohLSTM,
    cache_dir: Path,
    n_rows: int,
    batch_size: int,
    device: torch.device,
    channels: int = 3,
) -> np.ndarray:
    """在测试集缓存上批量前向，返回 (n_rows,) 预测。"""
    # memmap 只读映射 4 通道缓存，按需切前 3 列 [I, V, Q] 或全 4 列。
    x_full = np.memmap(
        str(cache_dir / "X_test.npy"),
        dtype=np.float32,
        mode="r",
        shape=(n_rows, 101, 4),
    )
    preds: list[np.ndarray] = []
    for start in range(0, n_rows, batch_size):
        # np.array 复制为可写数组，避免 PyTorch 对 memmap 只读数组的警告。
        xb = torch.from_numpy(
            np.array(x_full[start : start + batch_size, :, :channels])
        )
        preds.append(model.soh_predict(xb.to(device)).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--channels", type=int, default=3, choices=(3, 4),
                        help="3=无温度基线，4=带温度通道基线")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((args.cache_dir / "meta.json").read_text(encoding="utf-8"))
    n_rows = int(meta["n_test"])

    table = load_test_table(args.index)
    if len(table) != n_rows:
        raise RuntimeError(
            f"索引有效测试行数 {len(table)} 与缓存 {n_rows} 不一致，"
            "行序对齐不可靠，请先重建缓存"
        )

    model = load_model_channels(args.model, device, args.channels)
    t0 = time.perf_counter()
    pred = predict_test_set(
        model, args.cache_dir, n_rows, args.batch_size, device, args.channels
    )
    print(f"前向完成：{n_rows:,} 片段，{time.perf_counter() - t0:.0f}s", flush=True)

    table["soh_pred"] = pred.astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)

    # 合理性校验：聚合 MAE 应与 README 记录的 1.40% 基线接近。
    err = pred - table["soh_true"].to_numpy(dtype=np.float32)
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    print(f"校验：MAE={mae * 100:.2f}%  RMSE={rmse * 100:.2f}%  "
          f"（基线记录 1.40%）", flush=True)
    print(f"已保存: {args.out}", flush=True)


if __name__ == "__main__":
    main()
