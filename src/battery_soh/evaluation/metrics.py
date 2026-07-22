"""SOC/SOH 估计误差指标。"""

from __future__ import annotations

import numpy as np


def _as_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_pred - y_true


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差。"""
    err = _as_error(y_true, y_pred)
    return float(np.sqrt(np.mean(err**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差。"""
    err = _as_error(y_true, y_pred)
    return float(np.mean(np.abs(err)))


def max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """最大绝对误差（用于边界/最坏情况分析）。"""
    err = _as_error(y_true, y_pred)
    return float(np.max(np.abs(err)))
