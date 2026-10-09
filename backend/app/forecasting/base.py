"""The contract every item-level model implements (Strategy pattern).

Models work in log1p space because the competition metric (NWRMSLE) is a
weighted RMSE there; callers convert back to units with `expm1`.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from app.features.windows import DesignMatrix


@runtime_checkable
class ItemForecaster(Protocol):
    name: str

    def fit(self, train: DesignMatrix, valid: DesignMatrix) -> None:
        """Learn from stacked training windows; `valid` is for early stopping only."""

    def predict(self, data: DesignMatrix) -> np.ndarray:
        """Return log1p forecasts shaped [series, horizon], clipped at zero."""
