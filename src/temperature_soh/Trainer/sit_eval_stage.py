"""按 SOH 阶段分桶评估 SIT few-shot 模型（比赛相关视角）。

背景：SIT 部分电池极端深衰减（SOH 降到 0.06~0.08），而 BMS 通常在
SOH ≥ 0.8 工作。把误差按标签 SOH 分桶，区分"常规老化区"（比赛相关）
与"极端深衰减区"（研究性极限）。

用法：
```powershell
& "E:\conda\envs\battery-soh\python.exe" src/temperature_soh/Trainer/sit_eval_stage.py --model models/temperature_soh/sit_fewshot_14cell_C.pt --cells 001-2,001-7,001-8,002-3,002-7,101-3
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
TR_DIR = Path(__file__).resolve().parent
DL_DIR = ROOT / "src" / "temperature_soh" / "DataLoader"
for d in (TR_DIR, DL_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from finetune_sit import get_cell_samples  # noqa: E402
from model import TemperatureSohLSTM  # noqa: E402
from sit_io import DEFAULT_SIT_DIR, discover_sit_cells  # noqa: E402

DEFAULT_CACHE_DIR = ROOT / "data" / "processed" / "sit_cache"
BUCKETS = [(0.9, 1.01, "SOH≥0.9"), (0.7, 0.9, "0.7-0.9"), (0.5, 0.7, "0.5-0.7"), (0.0, 0.5, "<0.5")]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cells", required=True, help="评估电池（逗号分隔）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SIT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemperatureSohLSTM(input_dim=3, use_temp_embed=False)
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    temp_of = dict(zip(discover_sit_cells(args.data_dir)["cell_id"],
                       discover_sit_cells(args.data_dir)["temp_group"]))

    bucket_err: dict[str, list[float]] = {b[2]: [] for b in BUCKETS}
    print(f"{'电池':<8}{'温度组':<10}{'MAE':>8}")
    for cell_id in cells:
        x, y = get_cell_samples(cell_id, args.data_dir, args.cache_dir)
        preds = []
        for start in range(0, len(x), 4096):
            xb = torch.from_numpy(x[start : start + 4096]).to(device)
            with torch.no_grad():
                preds.append(model.soh_predict(xb).cpu().numpy())
        pred = np.concatenate(preds)
        err = pred - y
        print(f"{cell_id:<8}{temp_of.get(cell_id, '?'):<10}{np.abs(err).mean() * 100:>7.2f}%")
        for lo, hi, name in BUCKETS:
            mask = (y >= lo) & (y < hi)
            if mask.sum() > 0:
                bucket_err[name].extend(err[mask].tolist())

    print("\n===== 按 SOH 阶段（全部测试电池合并）=====")
    for lo, hi, name in BUCKETS:
        errs = bucket_err[name]
        if errs:
            e = np.asarray(errs)
            print(f"{name:<10}: n={len(e):>7,}  MAE={np.abs(e).mean() * 100:6.2f}%  "
                  f"bias={e.mean() * 100:6.2f}%")


if __name__ == "__main__":
    main()
