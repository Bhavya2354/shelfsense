"""Newsvendor ordering: stock the critical-ratio quantile of demand over the cover period.

Demand uncertainty for a whole cover period (one day for perishables, a week
for shelf-stable goods) is learned empirically: the log-ratio of actual to
forecast period demand on a backtest fold, bucketed by forecast level. This
keeps the day-to-day correlation of errors that summing daily intervals would lose.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl


def critical_ratio(underage: float, overage: float) -> float:
    """Optimal in-stock probability: cost of a lost sale over total mismatch cost."""
    return underage / (underage + overage)


def period_sums(daily: pl.DataFrame, cover_days: int, columns: list[str]) -> pl.DataFrame:
    """Sum daily columns over consecutive, complete cover periods per series.

    `daily` needs store_nbr, item_nbr, target_date and first_day (the first
    forecast day of the run the rows belong to).
    """
    horizon = (
        daily.select((pl.col("target_date") - pl.col("first_day")).dt.total_days().max()).item() + 1
    )
    periods = horizon // cover_days
    return (
        daily.with_columns(
            ((pl.col("target_date") - pl.col("first_day")).dt.total_days() // cover_days).alias(
                "period"
            )
        )
        .filter(pl.col("period") < periods)
        .group_by("first_day", "store_nbr", "item_nbr", "period")
        .agg(*(pl.col(c).sum() for c in columns))
        .with_columns(
            (pl.col("first_day") + pl.duration(days=pl.col("period") * cover_days)).alias(
                "period_start"
            )
        )
    )


@dataclass(frozen=True)
class CoverDistribution:
    edges: np.ndarray
    residuals: list[np.ndarray]  # log1p residual samples, one array per level bucket

    def _bucket(self, point: np.ndarray) -> np.ndarray:
        level = np.log1p(np.maximum(point, 0))
        return np.clip(
            np.searchsorted(self.edges, level, side="right") - 1, 0, len(self.residuals) - 1
        )

    def quantile(self, point: np.ndarray, q: float) -> np.ndarray:
        shift = np.array([np.quantile(r, q) for r in self.residuals])
        return np.maximum(np.expm1(np.log1p(np.maximum(point, 0)) + shift[self._bucket(point)]), 0)

    def expected_mismatch(
        self, point: np.ndarray, order: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected units short and units left over for each row."""
        lost = np.zeros_like(point, dtype=np.float64)
        left = np.zeros_like(point, dtype=np.float64)
        buckets = self._bucket(point)
        for b, samples in enumerate(self.residuals):
            rows = buckets == b
            if not rows.any():
                continue
            demand = np.maximum(np.expm1(np.log1p(point[rows])[:, None] + samples[None, :]), 0)
            lost[rows] = np.maximum(demand - order[rows, None], 0).mean(axis=1)
            left[rows] = np.maximum(order[rows, None] - demand, 0).mean(axis=1)
        return lost, left


def fit_cover_distribution(
    point: np.ndarray, actual: np.ndarray, buckets: int, samples: int, seed: int
) -> CoverDistribution:
    level = np.log1p(np.maximum(point, 0))
    inner = np.quantile(level, np.linspace(0, 1, buckets + 1)[1:-1])
    edges = np.unique(np.concatenate([[-np.inf], inner]))
    residual = np.log1p(np.maximum(actual, 0)) - level
    which = np.clip(np.searchsorted(edges, level, side="right") - 1, 0, len(edges) - 1)
    rng = np.random.default_rng(seed)
    pools = []
    for b in range(len(edges)):
        pool = residual[which == b]
        pool = pool if pool.size else residual
        pools.append(rng.choice(pool, size=min(samples, pool.size), replace=False))
    return CoverDistribution(edges, pools)
