import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from forgery.calibration import brier, calibrate, ece_from_table, reliability_table


def test_perfect_scores_have_zero_error():
    y = np.array([0, 0, 1, 1])
    score = y.astype(float)
    assert brier(y, score) == 0.0
    assert ece_from_table(reliability_table(y, score), len(y)) == pytest.approx(0.0)


def test_constant_score_error():
    y = np.array([0, 1, 0, 1])
    score = np.full(4, 0.9)
    assert brier(y, score) == pytest.approx((0.81 + 0.01 + 0.81 + 0.01) / 4)
    assert ece_from_table(reliability_table(y, score), len(y)) == pytest.approx(0.4)


def test_calibrate_returns_three_versions():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"is_forged": rng.random(60) < 0.5})
    frame["mean_probability"] = np.clip(frame.is_forged * 0.4 + rng.random(60) * 0.6, 0, 1)
    summary, tables = calibrate(frame, frame)
    assert set(summary) == {"raw score", "Platt", "isotonic"}
    assert all(0.0 <= values["brier"] <= 1.0 for values in summary.values())
