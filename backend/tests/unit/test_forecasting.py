from datetime import timedelta

import numpy as np
import pytest

from app.config import ForecastSettings
from app.features.context import Context
from app.features.panel import Panel
from app.features.windows import build_window, stack_windows
from app.forecasting import baselines, embedding_mlp, gradient_boosting  # noqa: F401 (registration)
from app.forecasting.ensemble import blend, fit_blend_weights
from app.forecasting.intervals import fit_residual_quantiles
from app.forecasting.registry import create_item_model, item_model_names

SETTINGS = ForecastSettings(
    chronos_model_id="unused",
    lgbm_max_rounds=40,
    mlp_epochs=2,
    mlp_batch_size=256,
    n_jobs=2,
)


@pytest.fixture
def windows(panel: Panel, context: Context):  # type: ignore[no-untyped-def]
    last = panel.last_observed - timedelta(days=panel.horizon - 1)
    make = lambda d: build_window(panel, context, d, perishable_weight=1.25)  # noqa: E731
    train = stack_windows([make(last - timedelta(days=7 * k)) for k in range(4, 8)])
    return train, make(last - timedelta(days=21)), make(last)


def test_registry_lists_all_item_models() -> None:
    assert {"moving_average", "weekday_average", "lightgbm", "embedding_mlp"} <= set(
        item_model_names()
    )
    with pytest.raises(ValueError, match="unknown item model"):
        create_item_model("prophet", SETTINGS)


@pytest.mark.parametrize("name", ["moving_average", "weekday_average", "lightgbm", "embedding_mlp"])
def test_every_model_returns_non_negative_horizon_forecasts(name: str, windows) -> None:  # type: ignore[no-untyped-def]
    train, valid, test = windows
    model = create_item_model(name, SETTINGS)
    model.fit(train, valid)
    pred = model.predict(test)
    assert pred.shape == test.target.shape
    assert np.all(pred >= 0) and np.all(np.isfinite(pred))


def test_lightgbm_beats_a_flat_average_on_seasonal_data(windows) -> None:  # type: ignore[no-untyped-def]
    train, valid, test = windows
    errors = {}
    for name in ("moving_average", "lightgbm"):
        model = create_item_model(name, SETTINGS)
        model.fit(train, valid)
        errors[name] = np.mean((model.predict(test) - test.target) ** 2)
    assert errors["lightgbm"] < errors["moving_average"]


def test_blend_weights_are_convex_and_prefer_the_better_model(rng: np.random.Generator) -> None:
    target = rng.random((300, 4))
    preds = {"good": target + rng.normal(0, 0.05, target.shape), "bad": rng.random(target.shape)}
    weights = fit_blend_weights(preds, target, np.ones(300))
    assert sum(weights.values()) == pytest.approx(1)
    assert weights["good"] > 0.8
    assert blend(preds, weights).shape == target.shape


def test_conformal_intervals_reach_nominal_coverage(rng: np.random.Generator) -> None:
    n, h = 4000, 16
    velocity = rng.gamma(2, 1, n)
    point = np.log1p(velocity)[:, None].repeat(h, axis=1)
    noise = lambda: rng.normal(0, 0.3, (n, h))  # noqa: E731
    calibration = fit_residual_quantiles(point, point + noise(), velocity, (0.1, 0.9), 5)
    actual = point + noise()
    lo, hi = calibration.interval(point, velocity)
    coverage = np.mean((actual >= lo) & (actual <= hi))
    assert 0.76 < coverage < 0.84
