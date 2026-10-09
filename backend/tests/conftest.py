from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.features.context import Context, build_context
from app.features.panel import Panel

HORIZON = 16


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def panel(rng: np.random.Generator) -> Panel:
    """Small synthetic panel with weekly seasonality and random promotions."""
    n, days = 240, 200
    keys = pl.DataFrame(
        {
            "store_nbr": rng.integers(1, 6, n),
            "item_nbr": np.arange(n) // 2 + 100,
            "family": rng.choice(["DAIRY", "PRODUCE", "BREAD"], n),
            "item_class": rng.integers(1, 5, n),
            "perishable": rng.random(n) < 0.3,
            "city": rng.choice(["Quito", "Cuenca"], n),
            "store_type": rng.choice(["A", "B"], n),
            "cluster": rng.integers(1, 4, n),
        }
    ).with_columns(
        state=pl.when(pl.col("city") == "Quito")
        .then(pl.lit("Pichincha"))
        .otherwise(pl.lit("Azuay"))
    )
    level = rng.gamma(2.0, 2.0, n)[:, None]
    weekly = 1 + 0.4 * np.sin(np.arange(days) * 2 * np.pi / 7)
    sales = np.log1p(rng.poisson(level * weekly)).astype(np.float32)
    promo = (rng.random((n, days + HORIZON)) < 0.1).astype(np.float32)
    start = date(2017, 1, 1)
    return Panel(keys, start, start + timedelta(days=days - 1), HORIZON, sales, promo)


@pytest.fixture
def context(panel: Panel) -> Context:
    holidays = pl.DataFrame(
        {
            "date": [date(2017, 6, 1), date(2017, 6, 5), date(2017, 6, 9)],
            "holiday_type": ["Holiday", "Holiday", "Work Day"],
            "locale": ["National", "Local", "National"],
            "locale_name": ["Ecuador", "Quito", "Ecuador"],
            "description": ["a", "b", "c"],
            "transferred": [False, False, False],
        }
    )
    return build_context(panel, holidays, None)
