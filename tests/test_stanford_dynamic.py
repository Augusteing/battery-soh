"""Stanford 动态数据集 loader 测试（使用合成 CSV，不依赖真实下载）。"""

import pandas as pd
import pytest

from battery_soh.data.stanford_dynamic import (
    build_soh_table,
    detect_capacity_column,
    load_aging_summary,
)


@pytest.fixture()
def fake_summaries(tmp_path):
    for cell, q0 in (("003", 1.10), ("004", 1.08)):
        df = pd.DataFrame(
            {
                "cycle_index": [1, 2, 3, 4],
                "discharge_capacity": [q0, q0 * 0.99, q0 * 0.97, q0 * 0.94],
            }
        )
        df.to_csv(tmp_path / f"aging_summary_cell_{cell}.csv", index=False)
    return tmp_path


def test_detect_capacity_column() -> None:
    assert detect_capacity_column(["cycle_index", "Discharge_Capacity"]) == "Discharge_Capacity"
    with pytest.raises(KeyError):
        detect_capacity_column(["a", "b"])


def test_load_aging_summary_soh(fake_summaries) -> None:
    df = load_aging_summary(fake_summaries / "aging_summary_cell_003.csv")
    assert df["cell_id"].iloc[0] == "cell_003"
    assert df["soh"].iloc[0] == pytest.approx(1.0)
    assert df["soh"].iloc[-1] == pytest.approx(0.94)
    assert df["cycle_index"].is_monotonic_increasing


def test_build_soh_table(fake_summaries, tmp_path) -> None:
    out = tmp_path / "soh.csv"
    table = build_soh_table(fake_summaries, out)
    assert set(table["cell_id"].unique()) == {"cell_003", "cell_004"}
    assert len(table) == 8
    assert out.exists()
    with pytest.raises(FileNotFoundError):
        build_soh_table(tmp_path / "empty_dir")


def test_real_schema_with_diagnostic_cycles(tmp_path) -> None:
    """真实表头 + 诊断循环（容量>1.5）应被剔除出 SOH 基准，但保留在表中。"""
    df = pd.DataFrame(
        {
            "Cycle": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "Normalized Charge Capacity [-]": [1.88, 6.2] + [1.0] * 10,
            "Normalized Discharge Capacity [-]": [2.098, 6.168] + [0.93 - i * 0.005 for i in range(10)],
            "Normalized Cumulative Capacity [-]": [3.978, 16.349] + [18.0 + i for i in range(10)],
        }
    )
    p = tmp_path / "aging_summary_cell_089.csv"
    df.to_csv(p, index=False)
    out = load_aging_summary(p)
    assert out["is_diagnostic"].tolist()[:2] == [True, True]
    assert out["is_diagnostic"].tolist()[2:] == [False] * 10
    # SOH 基准为常规循环中位数（约 0.9075），首个常规循环 SOH 应接近 1 而不被诊断值拉偏
    regular = out[~out["is_diagnostic"]]
    assert regular["soh"].iloc[0] == pytest.approx(0.93 / regular["discharge_capacity"].head(10).median())
    assert "cumulative_capacity" in out.columns
