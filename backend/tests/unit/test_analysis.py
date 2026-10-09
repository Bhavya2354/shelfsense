import numpy as np

from app.analysis.effects import _fe_ols
from app.analysis.stationarity import stationarity_tests


def test_two_way_fixed_effects_recover_the_true_effect(rng: np.random.Generator) -> None:
    n_series, n_days, true_beta = 200, 60, 0.4
    series = np.repeat(np.arange(n_series), n_days)
    days = np.tile(np.arange(n_days), n_series)
    series_effect = rng.normal(0, 2, n_series)[series]
    day_effect = rng.normal(0, 1, n_days)[days]
    # Treatment correlated with the fixed effects: naive OLS would be biased.
    treated = (rng.random(series.size) < 0.2 + 0.1 * (series_effect > 0)).astype(float)
    y = series_effect + day_effect + true_beta * treated + rng.normal(0, 0.3, series.size)
    beta, se = _fe_ols(y, treated[:, None], [series, days], series, iterations=20)
    assert abs(beta[0] - true_beta) < 3 * se[0]
    assert se[0] < 0.05


def test_stationarity_verdicts(rng: np.random.Generator) -> None:
    random_walk = np.cumsum(rng.normal(0, 1, 600))
    findings = dict(stationarity_tests(random_walk, alpha=0.05))
    assert findings["log_level"]["verdict"] == "non-stationary"
    assert findings["log_diff"]["verdict"] == "stationary"
