"""Convex blend of model forecasts, weights fitted by non-negative least squares."""

import numpy as np
from scipy.optimize import nnls


def fit_blend_weights(
    predictions: dict[str, np.ndarray], target: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    """Weights >= 0 summing to one that minimise the weighted squared error."""
    names = list(predictions)
    root_w = np.sqrt(np.repeat(weight[:, None], target.shape[1], axis=1)).ravel()
    design = np.column_stack([predictions[n].ravel() * root_w for n in names])
    coefs, _ = nnls(design.astype(np.float64), (target.ravel() * root_w).astype(np.float64))
    if coefs.sum() == 0:
        coefs = np.ones(len(names))
    coefs = coefs / coefs.sum()
    return {n: float(c) for n, c in zip(names, coefs, strict=True)}


def blend(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(predictions[n] * w for n, w in weights.items() if w > 0)
    return np.asarray(total, dtype=np.float32)
