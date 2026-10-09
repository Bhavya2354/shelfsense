"""Replenishment policies, replayed against actual demand, and the released order plan.

Policies compared on the most recent backtest fold:
* naive      - order the recent two-week average for the cover period
* forecast   - order the median forecast
* newsvendor - order the critical-ratio quantile of forecast demand

Recorded sales are a lower bound on true demand when shelves ran empty, so
lost-sale figures are conservative.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from app.config import InventorySettings
from app.inventory.newsvendor import (
    CoverDistribution,
    critical_ratio,
    fit_cover_distribution,
    period_sums,
)

POLICIES = ("naive", "forecast", "newsvendor")
_COLUMNS = ["actual", "p50", "naive"]


@dataclass(frozen=True)
class ItemClass:
    name: str
    perishable: bool
    cover_days: int
    service_level: float
    overage_cost: float


def item_classes(settings: InventorySettings) -> tuple[ItemClass, ItemClass]:
    return (
        ItemClass(
            "perishable",
            True,
            settings.cover_days_perishable,
            critical_ratio(settings.underage_cost, settings.overage_cost_perishable),
            settings.overage_cost_perishable,
        ),
        ItemClass(
            "shelf_stable",
            False,
            settings.cover_days_shelf_stable,
            critical_ratio(settings.underage_cost, settings.overage_cost_shelf_stable),
            settings.overage_cost_shelf_stable,
        ),
    )


def _periods(
    daily: pl.DataFrame, items: pl.DataFrame, cls: ItemClass, columns: list[str]
) -> pl.DataFrame:
    subset = daily.join(items, on="item_nbr").filter(pl.col("perishable") == cls.perishable)
    sums = period_sums(subset, cls.cover_days, columns)
    return sums.join(items.select("item_nbr", "family"), on="item_nbr")


def calibrate(
    backtest: pl.DataFrame,
    items: pl.DataFrame,
    settings: InventorySettings,
    buckets: int,
    seed: int,
) -> dict[str, CoverDistribution]:
    """Learn period-demand uncertainty per item class from one backtest fold."""
    out = {}
    for cls in item_classes(settings):
        periods = _periods(backtest, items, cls, _COLUMNS)
        out[cls.name] = fit_cover_distribution(
            periods["p50"].to_numpy(),
            periods["actual"].to_numpy(),
            buckets,
            settings.residual_samples_per_bucket,
            seed,
        )
    return out


def evaluate_policies(
    backtest: pl.DataFrame,
    items: pl.DataFrame,
    distributions: dict[str, CoverDistribution],
    settings: InventorySettings,
) -> pl.DataFrame:
    rows = []
    for cls in item_classes(settings):
        periods = _periods(backtest, items, cls, _COLUMNS)
        demand = periods["actual"].to_numpy()
        orders = {
            "naive": periods["naive"].to_numpy(),
            "forecast": periods["p50"].to_numpy(),
            "newsvendor": distributions[cls.name].quantile(
                periods["p50"].to_numpy(), cls.service_level
            ),
        }
        for policy, order in orders.items():
            outcome = periods.select("family").with_columns(
                demand=pl.Series(demand),
                lost=pl.Series(np.maximum(demand - order, 0)),
                leftover=pl.Series(np.maximum(order - demand, 0)),
            )
            for segment, part in [(cls.name, outcome), *outcome.group_by("family")]:
                name = segment if isinstance(segment, str) else f"{cls.name}:{segment[0]}"
                rows.append(
                    _summarise(policy, name, part, settings.underage_cost, cls.overage_cost)
                )
    frame = pl.DataFrame(rows)
    totals = (
        frame.filter(pl.col("segment").is_in(["perishable", "shelf_stable"]))
        .group_by("policy")
        .agg(pl.col("demand_units", "lost_units", "leftover_units", "total_cost").sum())
        .with_columns(
            pl.lit("all").alias("segment"),
            (1 - pl.col("lost_units") / pl.col("demand_units")).alias("fill_rate"),
        )
    )
    return pl.concat([frame, totals.select(frame.columns)])


def _summarise(
    policy: str, segment: str, part: pl.DataFrame, underage: float, overage: float
) -> dict[str, object]:
    demand, lost, left = (float(part[c].sum()) for c in ("demand", "lost", "leftover"))
    return {
        "policy": policy,
        "segment": segment,
        "demand_units": demand,
        "lost_units": lost,
        "leftover_units": left,
        "fill_rate": 1 - lost / demand if demand else 1.0,
        "total_cost": underage * lost + overage * left,
    }


def plan_orders(
    release: pl.DataFrame,
    items: pl.DataFrame,
    distributions: dict[str, CoverDistribution],
    settings: InventorySettings,
) -> pl.DataFrame:
    """Order quantity for the first cover period of the released forecast."""
    parts = []
    for cls in item_classes(settings):
        periods = _periods(release, items, cls, ["p50"]).filter(pl.col("period") == 0)
        point = periods["p50"].to_numpy()
        order = np.ceil(distributions[cls.name].quantile(point, cls.service_level))
        lost, left = distributions[cls.name].expected_mismatch(point, order)
        parts.append(
            periods.select("store_nbr", "item_nbr").with_columns(
                pl.lit(cls.cover_days).alias("cover_days"),
                pl.lit(cls.service_level).alias("service_level"),
                pl.Series("expected_demand", point),
                pl.Series("order_quantity", order),
                pl.Series("safety_stock", order - point),
                pl.Series("expected_lost_units", lost),
                pl.Series("expected_leftover_units", left),
            )
        )
    return pl.concat(parts)
