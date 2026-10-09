from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.config import InventorySettings
from app.inventory.newsvendor import critical_ratio, fit_cover_distribution, period_sums
from app.inventory.policies import calibrate, evaluate_policies, item_classes, plan_orders


def test_critical_ratio() -> None:
    assert critical_ratio(0.3, 0.7) == pytest.approx(0.3)
    assert critical_ratio(0.3, 0.03) == pytest.approx(0.909, abs=1e-3)


def test_period_sums_keep_only_complete_periods() -> None:
    first = date(2017, 7, 26)
    daily = pl.DataFrame(
        {
            "store_nbr": [1] * 16,
            "item_nbr": [7] * 16,
            "first_day": [first] * 16,
            "target_date": [first + timedelta(days=d) for d in range(16)],
            "actual": [1.0] * 16,
        }
    )
    weekly = period_sums(daily, 7, ["actual"])
    assert sorted(weekly["actual"].to_list()) == [7.0, 7.0]  # day 15 and 16 are dropped


def test_quantile_orders_grow_with_service_level(rng: np.random.Generator) -> None:
    point = rng.gamma(3, 3, 5000)
    actual = point * np.exp(rng.normal(0, 0.3, 5000))
    dist = fit_cover_distribution(point, actual, buckets=4, samples=300, seed=0)
    low, high = dist.quantile(point, 0.3), dist.quantile(point, 0.9)
    assert np.all(high >= low)
    lost, left = dist.expected_mismatch(point, high)
    assert np.all(lost >= 0) and np.all(left >= 0)
    assert lost.mean() < left.mean()  # a 90% service level over-stocks on average


def _frame(
    rng: np.random.Generator, first: date, n_items: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = []
    for item in range(n_items):
        level = rng.gamma(2, 3)
        for d in range(16):
            actual = rng.poisson(level)
            rows.append(
                (1, item, first + timedelta(days=d), first, float(actual), level, level * 0.8)
            )
    daily = pl.DataFrame(
        rows,
        schema=["store_nbr", "item_nbr", "target_date", "first_day", "actual", "p50", "naive"],
        orient="row",
    )
    items = pl.DataFrame(
        {
            "item_nbr": list(range(n_items)),
            "family": ["DAIRY"] * n_items,
            "item_class": [1] * n_items,
            "perishable": [i % 2 == 0 for i in range(n_items)],
        }
    )
    return daily, items


def test_newsvendor_has_the_lowest_mismatch_cost(rng: np.random.Generator) -> None:
    settings = InventorySettings()
    older, items = _frame(rng, date(2017, 7, 12), 400)
    newer, _ = _frame(rng, date(2017, 7, 26), 400)

    dists = calibrate(older, items, settings, buckets=3, seed=0)
    results = evaluate_policies(newer, items, dists, settings)
    totals = {
        r["policy"]: r["total_cost"] for r in results.filter(pl.col("segment") == "all").to_dicts()
    }
    assert totals["newsvendor"] <= min(totals["naive"], totals["forecast"])

    plan = plan_orders(newer.drop("actual", "naive"), items, dists, settings)
    assert plan.height == 400
    assert set(plan["cover_days"].unique()) == {c.cover_days for c in item_classes(settings)}
    assert (plan["order_quantity"] >= 0).all()
