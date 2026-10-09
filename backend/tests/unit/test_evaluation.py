import numpy as np
import pytest

from app.evaluation import metrics
from app.evaluation.significance import bootstrap_nwrmsle_difference, diebold_mariano


def test_nwrmsle_matches_hand_computation() -> None:
    pred = np.array([[0.0, 1.0], [2.0, 2.0]])
    actual = np.array([[1.0, 1.0], [2.0, 0.0]])
    weight = np.array([1.0, 1.25])
    expected = np.sqrt((1 * 1 + 1 * 0 + 1.25 * 0 + 1.25 * 4) / (2 * 1 + 2 * 1.25))
    assert metrics.nwrmsle(pred, actual, weight) == pytest.approx(expected)


def test_unit_metrics() -> None:
    pred, actual = np.array([12.0, 8.0]), np.array([10.0, 10.0])
    assert metrics.wape(pred, actual) == pytest.approx(0.2)
    assert metrics.bias(pred, actual) == pytest.approx(0.0)
    assert metrics.coverage(np.array([0, 9]), np.array([5, 11]), np.array([6, 10])) == 0.5


def test_pinball_penalises_the_right_side() -> None:
    actual = np.array([10.0])
    assert metrics.pinball(np.array([8.0]), actual, 0.9) == pytest.approx(1.8)
    assert metrics.pinball(np.array([12.0]), actual, 0.9) == pytest.approx(0.2)


def test_diebold_mariano_detects_a_clearly_better_model(rng: np.random.Generator) -> None:
    better = rng.normal(1.0, 0.05, 40)
    worse = better + 0.2 + rng.normal(0, 0.05, 40)
    result = diebold_mariano(better, worse, max_lag=3)
    assert result["statistic"] < 0
    assert result["p_value"] < 0.01


def test_bootstrap_of_identical_models_is_centred_on_zero(rng: np.random.Generator) -> None:
    actual = rng.random((500, 16))
    pred = actual + rng.normal(0, 0.1, actual.shape)
    out = bootstrap_nwrmsle_difference(pred, pred, actual, np.ones(500), resamples=60, seed=1)
    assert out["difference"] == 0
    assert out["ci_low"] == pytest.approx(0) and out["ci_high"] == pytest.approx(0)
