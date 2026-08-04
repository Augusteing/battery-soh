"""A 基线：特征回归预测 SOH。

模型: Ridge / HistGradientBoosting（sklearn，无需额外安装）
验证: 按电池分组 5 折 / 按协议分组 5 折 / 电池内时间切分（前 70% -> 后 30%）
输出: results/metrics/matr_baseline_metrics.json + 预测-真值散点图

用法:
    python scripts/train_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

TARGET = "soh"

# 不含：cell_id/policy/batch/cycle_life（分组与标签），
# 不含 discharge_capacity/charge_capacity 水平值（避免平凡特征）
FEATURES = [
    "cycle_index",
    "ir",
    "tavg",
    "tmax",
    "tmin",
    "chargetime",
    "temp_amp",
    "cumulative_charge",
    "ir_mean10",
    "ir_std10",
    "tavg_mean10",
    "tavg_std10",
    "ir_deriv10",
    "capacity_deriv10",
    "chargetime_ratio",
]

MODELS = {
    "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
    "hist_gb": HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=4, random_state=0
    ),
}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "max_abs_error": float(np.max(np.abs(y_pred - y_true))),
        "n": int(len(y_true)),
    }


def group_cv(features: pd.DataFrame, groups: np.ndarray, label: str, out: dict) -> None:
    """GroupKFold：按电池或按协议分组，测试组在训练中完全不可见。"""
    gkf = GroupKFold(n_splits=5)
    for model_name, model in MODELS.items():
        y_true_all, y_pred_all = [], []
        for train_idx, test_idx in gkf.split(features, features[TARGET], groups):
            X_tr = features.iloc[train_idx][FEATURES]
            X_te = features.iloc[test_idx][FEATURES]
            model.fit(X_tr, features.iloc[train_idx][TARGET])
            y_true_all.append(features.iloc[test_idx][TARGET].values)
            y_pred_all.append(model.predict(X_te))
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        out[label][model_name] = evaluate(y_true, y_pred)
        out[label][model_name]["pred"] = y_pred.tolist()
        out[label][model_name]["true"] = y_true.tolist()


def temporal_split(features: pd.DataFrame, label: str, out: dict, frac: float = 0.7) -> None:
    """电池内时间切分：前 frac 循环训练，后 (1-frac) 循环测试（模拟在线预测）。"""
    train_idx, test_idx = [], []
    for _cell_id, g in features.groupby("cell_id", sort=False):
        n = len(g)
        cut = int(n * frac)
        train_idx.extend(g.index[:cut].tolist())
        test_idx.extend(g.index[cut:].tolist())
    for model_name, model in MODELS.items():
        model.fit(features.iloc[train_idx][FEATURES], features.iloc[train_idx][TARGET])
        y_true = features.iloc[test_idx][TARGET].values
        y_pred = model.predict(features.iloc[test_idx][FEATURES])
        out[label][model_name] = evaluate(y_true, y_pred)
        out[label][model_name]["pred"] = y_pred.tolist()
        out[label][model_name]["true"] = y_true.tolist()


def plot_results(out: dict, fig_path: Path) -> None:
    schemes = ["cv_cell", "cv_policy", "temporal"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, scheme in zip(axes, schemes):
        res = out[scheme]["hist_gb"]
        ax.scatter(res["pred"], res["true"], s=4, alpha=0.25, color="steelblue")
        ax.plot([0, 1], [0, 1], "r--", lw=1)
        ax.set_title(f"{scheme}: MAE={res['mae'] * 100:.2f}%")
        ax.set_xlabel("predicted SOH")
        ax.set_ylabel("true SOH")
        ax.set_xlim(0.5, 1.05)
        ax.set_ylim(0.5, 1.05)
        ax.grid(alpha=0.3)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    print(f"saved -> {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=ROOT / "data/processed/matr_features.parquet")
    parser.add_argument("--out-json", type=Path, default=ROOT / "results/metrics/matr_baseline_metrics.json")
    parser.add_argument("--out-fig", type=Path, default=ROOT / "results/figures/matr_baseline_pred.png")
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    missing = [c for c in FEATURES if c not in features.columns]
    if missing:
        raise SystemExit(f"特征缺失: {missing}")

    out: dict = {
        "dataset": "MATR 20170512",
        "n_rows": int(len(features)),
        "n_cells": int(features["cell_id"].nunique()),
        "n_policies": int(features["policy"].nunique()),
        "features": FEATURES,
        "results": {"cv_cell": {}, "cv_policy": {}, "temporal": {}},
    }

    group_cv(features, features["cell_id"].values, "cv_cell", out["results"])
    group_cv(features, features["policy"].values, "cv_policy", out["results"])
    temporal_split(features, "temporal", out["results"])

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {args.out_json}")
    for scheme, res in out["results"].items():
        for model_name, m in res.items():
            print(f"{scheme:10s} {model_name:8s} MAE={m['mae'] * 100:.2f}%  RMSE={m['rmse'] * 100:.2f}%  R2={m['r2']:.3f}")

    plot_results(out["results"], args.out_fig)


if __name__ == "__main__":
    sys.exit(main())
