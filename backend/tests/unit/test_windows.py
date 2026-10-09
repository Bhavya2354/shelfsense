from datetime import timedelta

import numpy as np
import pytest

from app.features.context import Context
from app.features.panel import Panel
from app.features.windows import LOOKBACK, build_window, stack_windows


def _origin(panel: Panel) -> object:
    return panel.last_observed - timedelta(days=panel.horizon - 1)


def test_features_ignore_sales_on_or_after_the_origin(panel: Panel, context: Context) -> None:
    origin = _origin(panel)
    before = build_window(panel, context, origin, perishable_weight=1.25)  # type: ignore[arg-type]
    t = panel.day(origin)  # type: ignore[arg-type]
    panel.sales[:, t:] = 99.0  # tamper with the future
    after = build_window(panel, context, origin, perishable_weight=1.25)  # type: ignore[arg-type]
    np.testing.assert_array_equal(before.x, after.x)


def test_targets_are_the_next_horizon_days(panel: Panel, context: Context) -> None:
    origin = _origin(panel)
    window = build_window(panel, context, origin, perishable_weight=1.25)  # type: ignore[arg-type]
    t = panel.day(origin)  # type: ignore[arg-type]
    assert window.target is not None
    np.testing.assert_array_equal(window.target, panel.sales[:, t : t + panel.horizon])


def test_known_future_inputs_are_included(panel: Panel, context: Context) -> None:
    origin = _origin(panel)
    window = build_window(panel, context, origin, perishable_weight=1.25)  # type: ignore[arg-type]
    t = panel.day(origin)  # type: ignore[arg-type]
    np.testing.assert_array_equal(window.column("promo_d3"), panel.promo[:, t + 3])


def test_perishables_get_the_competition_weight(panel: Panel, context: Context) -> None:
    window = build_window(panel, context, _origin(panel), perishable_weight=1.25)  # type: ignore[arg-type]
    perishable = panel.keys["perishable"].to_numpy()
    assert np.all(window.weight[perishable] == 1.25)
    assert np.all(window.weight[~perishable] == 1.0)


def test_origin_without_enough_history_is_rejected(panel: Panel, context: Context) -> None:
    with pytest.raises(ValueError, match="history"):
        build_window(
            panel, context, panel.start + timedelta(days=LOOKBACK - 1), perishable_weight=1.0
        )


def test_stacking_requires_targets(panel: Panel, context: Context) -> None:
    future = panel.last_observed + timedelta(days=1)
    unlabeled = build_window(panel, context, future, perishable_weight=1.0)
    assert unlabeled.target is None
    with pytest.raises(ValueError, match="targets"):
        stack_windows([unlabeled])
