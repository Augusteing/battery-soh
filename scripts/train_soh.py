"""Round 1 SOH 估计共享管线。

流程：特征表 -> 滑动窗口样本 -> 按电池/按协议划分 -> 训练集标准化
      -> 模型训练（early stopping）-> 测试评估（整体/按电池/按老化阶段分桶）。

用法:
    python scripts/train_soh.py --model gru --split by_cell
    python scripts/train_soh.py --model patchtst --split by_policy --epochs 30
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
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from battery_soh.models.sequence_models import MODELS  # noqa: E402

FEATURE_COLS = [
    "ir", "tmax", "tavg", "tmin", "chargetime",
    "cumulative_charge", "temp_amp",
    "ir_mean10", "ir_std10", "tavg_mean10", "tavg_std10",
    "ir_deriv10", "capacity_deriv10", "chargetime_ratio",
]
META_COLS = ["cell_id", "cycle_index", "batch", "policy", "soh"]

SOH_BUCKETS = [(0.95, float("inf"), ">=0.95"), (0.90, 0.95, "0.90-0.95"),
               (0.85, 0.90, "0.85-0.90"), (0.80, 0.85, "0.80-0.85"), (-float("inf"), 0.80, "<0.80")]


def build_samples(df: pd.DataFrame, window: int, stride: int, soh_min: float, soh_max: float):
    """按电池切出滑动窗口样本，返回 (X float32, y float32, meta DataFrame)。"""
    df = df.sort_values(["cell_id", "cycle_index"]).copy()
    df["soh"] = df["soh"].clip(soh_min, soh_max)

    Xs, ys, metas = [], [], []
    for cell_id, g in df.groupby("cell_id", sort=False):
        arr = g[FEATURE_COLS].to_numpy(dtype=np.float64)
        y = g["soh"].to_numpy(dtype=np.float64)
        if len(arr) < window:
            continue
        Xw = sliding_window_view(arr, window, axis=0).transpose(0, 2, 1)  # (n-w+1, w, F)
        yw = y[window - 1:]
        meta = g.iloc[window - 1:][META_COLS].reset_index(drop=True)
        if stride > 1:
            Xw = Xw[::stride]
            yw = yw[::stride]
            meta = meta.iloc[::stride].reset_index(drop=True)
        Xs.append(Xw.astype(np.float32))
        ys.append(yw.astype(np.float32))
        metas.append(meta)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    meta = pd.concat(metas, ignore_index=True)
    return X, y, meta


def make_splits(meta: pd.DataFrame, mode: str, seed: int):
    """返回训练/验证/测试的 cell_id 集合（按电池随机 or 按协议隔离）。"""
    rng = np.random.default_rng(seed)
    if mode == "by_cell":
        cells = np.array(sorted(meta["cell_id"].unique()))
        perm = rng.permutation(len(cells))
        n1, n2 = int(len(cells) * 0.7), int(len(cells) * 0.85)
        tr = set(cells[perm[:n1]].tolist())
        va = set(cells[perm[n1:n2]].tolist())
        te = set(cells[perm[n2:]].tolist())
    elif mode == "by_policy":
        policy_of_cell = meta.groupby("cell_id")["policy"].first()
        policies = np.array(sorted(policy_of_cell.unique()))
        perm = rng.permutation(len(policies))
        n1, n2 = int(len(policies) * 0.7), int(len(policies) * 0.85)
        tr_p = set(policies[perm[:n1]].tolist())
        va_p = set(policies[perm[n1:n2]].tolist())
        te_p = set(policies[perm[n2:]].tolist())
        tr = set(policy_of_cell[policy_of_cell.isin(tr_p)].index)
        va = set(policy_of_cell[policy_of_cell.isin(va_p)].index)
        te = set(policy_of_cell[policy_of_cell.isin(te_p)].index)
    else:
        raise ValueError(f"unknown split mode: {mode}")
    return tr, va, te


def standardize(X_tr, X_va, X_te):
    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + 1e-8
    return (X_tr - mu) / sd, (X_va - mu) / sd, (X_te - mu) / sd


def make_loader(X, y, batch_size, shuffle, seed):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=0)


def evaluate(model, loader, device):
    model.eval()
    ys, yhats = [], []
    with torch.no_grad():
        for xb, yb in loader:
            yhats.append(model(xb.to(device)).cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(yhats)


def metrics(y, yhat):
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.clip(y, 1e-6, None))) * 100.0
    return {"mae": mae * 100.0, "rmse": rmse * 100.0, "mape": mape}  # 百分数刻度（SOH*100）


def train(model, tr_loader, va_loader, epochs, lr, patience, device, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_mae, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        y, yh = evaluate(model, va_loader, device)
        m = metrics(y, yh)
        print(f"  epoch {epoch:02d}  train_mse={total / n:.6f}  val_mae={m['mae']:.4f}%  ({time.time() - t0:.1f}s)")
        if m["mae"] < best_mae:
            best_mae = m["mae"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop @ epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return best_mae


def bucket_report(y, yhat, meta):
    df = pd.DataFrame({"cell_id": meta["cell_id"].values, "y": y, "yhat": yhat})
    out = {}
    for lo, hi, name in SOH_BUCKETS:
        sub = df[(df["y"] > lo) & (df["y"] <= hi)]
        if len(sub):
            out[name] = float(np.mean(np.abs(sub["y"] - sub["yhat"]))) * 100.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=ROOT / "data/processed/matr_features.parquet")
    parser.add_argument("--model", choices=list(MODELS), default="gru")
    parser.add_argument("--split", choices=["by_cell", "by_policy"], default="by_cell")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--soh-min", type=float, default=0.7)
    parser.add_argument("--soh-max", type=float, default=1.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/metrics/round1")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[1/5] load & build samples (window={args.window}, stride={args.stride})")
    df = pd.read_parquet(args.features)
    X, y, meta = build_samples(df, args.window, args.stride, args.soh_min, args.soh_max)
    print(f"  samples={len(y):,}  cells={meta['cell_id'].nunique()}")

    print(f"[2/5] split by {args.split} (seed={args.seed})")
    tr, va, te = make_splits(meta, args.split, args.seed)
    mask = lambda s: meta["cell_id"].isin(s).to_numpy()  # noqa: E731
    X_tr, X_va, X_te = X[mask(tr)], X[mask(va)], X[mask(te)]
    y_tr, y_va, y_te = y[mask(tr)], y[mask(va)], y[mask(te)]
    meta_tr, meta_va, meta_te = meta[mask(tr)], meta[mask(va)], meta[mask(te)]
    print(f"  train cells={len(tr)} val={len(va)} test={len(te)} | samples {len(y_tr)}/{len(y_va)}/{len(y_te)}")

    print("[3/5] standardize on train")
    X_tr, X_va, X_te = standardize(X_tr, X_va, X_te)

    print(f"[4/5] train {args.model} ({args.device})")
    model = MODELS[args.model](c_in=len(FEATURE_COLS), seq_len=args.window).to(args.device)
    tr_loader = make_loader(X_tr, y_tr, args.batch, True, args.seed)
    va_loader = make_loader(X_va, y_va, args.batch, False, args.seed)
    te_loader = make_loader(X_te, y_te, args.batch, False, args.seed)
    best_val_mae = train(model, tr_loader, va_loader, args.epochs, args.lr, args.patience, args.device, args.seed)

    print("[5/5] evaluate on test")
    y_te, yhat_te = evaluate(model, te_loader, args.device)
    pooled = metrics(y_te, yhat_te)

    pred = meta_te[["cell_id", "cycle_index", "batch", "policy"]].copy()
    pred["soh"] = y_te
    pred["soh_pred"] = yhat_te

    cell_mae = pred.assign(err=(pred["soh"] - pred["soh_pred"]).abs()).groupby("cell_id")["err"].mean() * 100.0
    report = {
        "model": args.model,
        "split": args.split,
        "window": args.window,
        "seed": args.seed,
        "best_val_mae": best_val_mae,
        "test": pooled,
        "test_cell_averaged_mae": float(cell_mae.mean()),
        "test_by_aging_stage": bucket_report(y_te, yhat_te, meta_te),
        "train_cells": len(tr), "val_cells": len(va), "test_cells": len(te),
        "seconds": round(time.time() - t0, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.split}_s{args.seed}_w{args.window}"
    with open(args.out_dir / f"{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    pred.to_parquet(args.out_dir / f"{tag}_pred.parquet", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved -> {args.out_dir / tag}_*.{{json,parquet}}")


if __name__ == "__main__":
    sys.exit(main())