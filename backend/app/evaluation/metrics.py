"""Forecast accuracy metrics.

Item-level inputs are log1p arrays shaped [series, horizon] (the competition
metric NWRMSLE is a weighted RMSE in that space); unit-based metrics convert
back with expm1 first.
"""

import numpy as np


def _cell_weights(weight: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(weight[:, None], shape)


def nwrmsle(pred: np.ndarray, actual: np.ndarray, weight: np.ndarray) -> float:
    """Normalized weighted RMSLE (Favorita metric; perishables weigh 1.25)."""
    w = _cell_weights(weight, pred.shape)
    return float(np.sqrt(np.sum(w * (pred - actual) ** 2) / np.sum(w)))


def nwrmsle_by_day(pred: np.ndarray, actual: np.ndarray, weight: np.ndarray) -> np.ndarray:
    w = _cell_weights(weight, pred.shape)
    return np.sqrt(np.sum(w * (pred - actual) ** 2, axis=0) / np.sum(w, axis=0))


def wape(pred_units: np.ndarray, actual_units: np.ndarray) -> float:
    """Weighted absolute percentage error: total absolute error over total demand."""
    return float(np.abs(pred_units - actual_units).sum() / max(actual_units.sum(), 1e-9))


def bias(pred_units: np.ndarray, actual_units: np.ndarray) -> float:
    """Signed over-forecast share: positive means forecasting too much."""
    return float((pred_units - actual_units).sum() / max(actual_units.sum(), 1e-9))


def mae(pred_units: np.ndarray, actual_units: np.ndarray) -> float:
    return float(np.abs(pred_units - actual_units).mean())


def coverage(lower: np.ndarray, upper: np.ndarray, actual: np.ndarray) -> float:
    return float(((actual >= lower) & (actual <= upper)).mean())


def pinball(pred: np.ndarray, actual: np.ndarray, quantile: float) -> float:
    diff = actual - pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def rmsle(pred_units: np.ndarray, actual_units: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(pred_units) - np.log1p(actual_units)) ** 2)))


def mase(pred: np.ndarray, actual: np.ndarray, scale: np.ndarray) -> float:
    """Mean absolute scaled error; `scale` is each series' in-sample seasonal-naive MAE."""
    errors = np.abs(pred - actual).mean(axis=1) / np.maximum(scale, 1e-9)
    return float(np.mean(errors[np.isfinite(errors)]))


def item_scores(pred: np.ndarray, actual: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    units_pred, units_actual = np.expm1(pred), np.expm1(actual)
    return {
        "nwrmsle": nwrmsle(pred, actual, weight),
        "wape": wape(units_pred, units_actual),
        "bias": bias(units_pred, units_actual),
        "mae_units": mae(units_pred, units_actual),
    }
