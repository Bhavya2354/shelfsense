"""Split-conformal prediction intervals.

Residual quantiles are learned on a held-out backtest fold, separately for
each forecast day and for each sales-velocity bucket (slow sellers are far
noisier in log space than fast ones), then added to new point forecasts.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualQuantiles:
    edges: np.ndarray  # velocity bucket boundaries
    lower: np.ndarray  # [buckets, horizon]
    upper: np.ndarray  # [buckets, horizon]
    quantiles: tuple[float, float]

    def bucket(self, velocity: np.ndarray) -> np.ndarray:
        return _bucket(self.edges, velocity)

    def interval(self, point: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bounds in log1p space, clipped at zero."""
        b = self.bucket(velocity)
        return np.clip(point + self.lower[b], 0, None), np.clip(point + self.upper[b], 0, None)


def _bucket(edges: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, velocity, side="right") - 1, 0, len(edges) - 1)


def fit_residual_quantiles(
    point: np.ndarray,
    actual: np.ndarray,
    velocity: np.ndarray,
    quantiles: tuple[float, float],
    buckets: int,
) -> ResidualQuantiles:
    # Series with no recent sales form their own bucket; the rest split by quantile.
    active = velocity[velocity > 0]
    inner = np.quantile(active, np.linspace(0, 1, buckets)[1:-1]) if active.size else np.array([])
    edges = np.unique(np.concatenate([[-np.inf, 1e-9], inner]))
    residual = actual - point
    which = _bucket(edges, velocity)
    n_buckets = len(edges)
    lower = np.zeros((n_buckets, point.shape[1]), dtype=np.float32)
    upper = np.zeros_like(lower)
    for b in range(n_buckets):
        rows = residual[which == b] if np.any(which == b) else residual
        lower[b] = np.quantile(rows, quantiles[0], axis=0)
        upper[b] = np.quantile(rows, quantiles[1], axis=0)
    return ResidualQuantiles(edges, lower, upper, quantiles)
