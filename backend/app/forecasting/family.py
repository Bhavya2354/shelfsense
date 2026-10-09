"""Store x product-family forecasting on the national > store > store-family hierarchy.

Statistical models (seasonal naive, ETS, ARIMA) and neural models (N-BEATS,
TFT) forecast every level; their mean is then reconciled with MinTrace so the
store-family forecasts add up to the store and national ones.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import cast

import pandas as pd
import polars as pl

from app.config import ForecastSettings
from app.storage.warehouse import Warehouse

logger = logging.getLogger(__name__)

HIERARCHY = [["country"], ["country", "store_nbr"], ["country", "store_nbr", "family"]]
BOTTOM_LEVEL = "country/store_nbr/family"
INTERVAL_LEVEL = 80
STAT_MODELS = ("SeasonalNaive", "AutoETS", "AutoARIMA")
NEURAL_MODELS = ("NBEATS", "TFT")
ENSEMBLE = "Ensemble"
RECONCILED = "ensemble_mint"


@dataclass(frozen=True)
class FamilyHierarchy:
    frame: pd.DataFrame  # unique_id, ds, y (NaN on future days), promo
    summing: pd.DataFrame
    tags: dict[str, object]
    last_observed: date


def load_hierarchy(
    wh: Warehouse, last_observed: date, history_days: int, horizon: int
) -> FamilyHierarchy:
    """Daily units and items-on-promotion per store-family, zero-filled, plus future promos."""
    start = last_observed - timedelta(days=history_days + 3 * horizon)
    end = last_observed + timedelta(days=horizon)
    bottom = wh.frame(
        """
        WITH days AS (SELECT range::DATE AS ds FROM range(?::DATE, ?::DATE + 1, INTERVAL 1 DAY)),
        pairs AS (SELECT DISTINCT s.store_nbr, i.family FROM stores s CROSS JOIN items i),
        observed AS (
            SELECT store_nbr, i.family, date AS ds,
                   sum(greatest(unit_sales, 0)) AS y, sum(onpromotion::INT) AS promo
            FROM sales JOIN items i USING (item_nbr)
            WHERE date BETWEEN ? AND ? GROUP BY ALL
        ),
        planned AS (
            SELECT store_nbr, i.family, date AS ds, sum(onpromotion::INT) AS promo
            FROM test JOIN items i USING (item_nbr)
            WHERE date BETWEEN ? AND ? GROUP BY ALL
        )
        SELECT 'all' AS country, p.store_nbr::VARCHAR AS store_nbr, p.family, d.ds,
               CASE WHEN d.ds <= ? THEN coalesce(o.y, 0) END AS y,
               coalesce(o.promo, pl.promo, 0)::DOUBLE AS promo
        FROM pairs p CROSS JOIN days d
        LEFT JOIN observed o USING (store_nbr, family, ds)
        LEFT JOIN planned pl USING (store_nbr, family, ds)
        """,
        [start, end, start, last_observed, last_observed + timedelta(days=1), end, last_observed],
    )
    # Drop store-families that never sell (some stores carry no items of a family).
    active = bottom.group_by("store_nbr", "family").agg(pl.col("y").sum()).filter(pl.col("y") > 0)
    bottom = bottom.join(active.select("store_nbr", "family"), on=["store_nbr", "family"])
    from hierarchicalforecast.utils import aggregate

    frame, summing, tags = aggregate(bottom.to_pandas(), HIERARCHY, exog_vars={"promo": "sum"})
    future = frame["ds"] > pd.Timestamp(last_observed)
    frame.loc[future, "y"] = float("nan")
    return FamilyHierarchy(frame, summing, tags, last_observed)


def _long(forecasts: pd.DataFrame, model: str, point: str) -> pd.DataFrame:
    lo, hi = f"{model}-lo-{INTERVAL_LEVEL}", f"{model}-hi-{INTERVAL_LEVEL}"
    return pd.DataFrame(
        {
            "unique_id": forecasts["unique_id"],
            "ds": forecasts["ds"],
            "model_name": model,
            "p10": forecasts[lo] if lo in forecasts else float("nan"),
            "p50": forecasts[point],
            "p90": forecasts[hi] if hi in forecasts else float("nan"),
        }
    )


def forecast_family(
    data: FamilyHierarchy, first_day: date, settings: ForecastSettings
) -> pl.DataFrame:
    """Forecast `horizon` days starting at `first_day` using only data before it."""
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import MinTrace
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NBEATS, TFT
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

    h = settings.forecast_horizon
    first = pd.Timestamp(first_day)
    history_start = first - pd.Timedelta(days=settings.family_history_days)
    frame = data.frame
    train = frame[(frame["ds"] >= history_start) & (frame["ds"] < first)]
    future = frame[(frame["ds"] >= first) & (frame["ds"] < first + pd.Timedelta(days=h))]

    stats = StatsForecast(
        models=[
            SeasonalNaive(season_length=7),
            AutoETS(season_length=7),
            AutoARIMA(season_length=7),
        ],
        freq="D",
        n_jobs=settings.n_jobs,
    ).forecast(h=h, df=train[["unique_id", "ds", "y"]], level=[INTERVAL_LEVEL])

    loss = MQLoss(level=[INTERVAL_LEVEL])
    common = {
        "h": h,
        "input_size": settings.family_input_size,
        "loss": loss,
        "max_steps": settings.family_max_steps,
        "scaler_type": "robust",
        "random_seed": settings.random_seed,
        "enable_progress_bar": False,
        "logger": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        neural = NeuralForecast(
            models=[NBEATS(**common), TFT(**common, futr_exog_list=["promo"], hidden_size=64)],
            freq="D",
        )
        neural.fit(train[["unique_id", "ds", "y", "promo"]])
        neural_fc = neural.predict(futr_df=future[["unique_id", "ds", "promo"]])

    merged = cast(pd.DataFrame, stats).merge(neural_fc, on=["unique_id", "ds"])
    for model in NEURAL_MODELS:
        merged[model] = merged[f"{model}-median"]
    merged[ENSEMBLE] = merged[["AutoETS", "AutoARIMA", *NEURAL_MODELS]].mean(axis=1)

    reconciled = HierarchicalReconciliation(
        [MinTrace(method="wls_struct", nonnegative=True)]
    ).reconcile(Y_hat_df=merged[["unique_id", "ds", ENSEMBLE]], S_df=data.summing, tags=data.tags)
    mint_column = next(c for c in reconciled.columns if c.startswith(f"{ENSEMBLE}/"))
    merged = merged.merge(
        reconciled[["unique_id", "ds", mint_column]].rename(columns={mint_column: RECONCILED}),
        on=["unique_id", "ds"],
    )

    parts = [_long(merged, m, m) for m in (*STAT_MODELS, ENSEMBLE, RECONCILED)]
    parts += [_long(merged, m, f"{m}-median") for m in NEURAL_MODELS]
    out = pl.from_pandas(pd.concat(parts, ignore_index=True))
    return out.with_columns(
        pl.col("ds").cast(pl.Date),
        *(pl.col(c).clip(lower_bound=0) for c in ("p10", "p50", "p90")),
    )
