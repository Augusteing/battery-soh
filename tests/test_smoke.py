"""冒烟测试：包可导入、指标计算正确。"""

import numpy as np

import battery_soh
from battery_soh.evaluation import mae, max_abs_error, rmse


def test_version() -> None:
    assert battery_soh.__version__ == "0.1.0"


def test_metrics_perfect_prediction() -> None:
    y = np.array([0.1, 0.5, 0.9])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert max_abs_error(y, y) == 0.0


def test_metrics_known_values() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.03, 0.04])
    assert np.isclose(rmse(y_true, y_pred), np.sqrt((0.03**2 + 0.04**2) / 2))
    assert np.isclose(mae(y_true, y_pred), 0.035)
    assert np.isclose(max_abs_error(y_true, y_pred), 0.04)
