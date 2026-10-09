"""Reference forecasts every learned model has to beat."""

import numpy as np

from app.config import ForecastSettings
from app.features.windows import DesignMatrix
from app.forecasting.registry import register_item_model


class MovingAverage:
    """Flat forecast at the mean of the last two weeks."""

    name = "moving_average"

    def __init__(self, horizon: int) -> None:
        self._horizon = horizon

    def fit(self, train: DesignMatrix, valid: DesignMatrix) -> None:
        return None

    def predict(self, data: DesignMatrix) -> np.ndarray:
        return np.repeat(data.column("mean_14")[:, None], self._horizon, axis=1)


class WeekdayAverage:
    """Seasonal average: each day gets the mean of the same weekday over four weeks."""

    name = "weekday_average"

    def __init__(self, horizon: int) -> None:
        self._horizon = horizon

    def fit(self, train: DesignMatrix, valid: DesignMatrix) -> None:
        return None

    def predict(self, data: DesignMatrix) -> np.ndarray:
        return np.column_stack([data.column(f"dow{k % 7}_mean_4w") for k in range(self._horizon)])


@register_item_model(MovingAverage.name)
def _moving_average(settings: ForecastSettings) -> MovingAverage:
    return MovingAverage(settings.forecast_horizon)


@register_item_model(WeekdayAverage.name)
def _weekday_average(settings: ForecastSettings) -> WeekdayAverage:
    return WeekdayAverage(settings.forecast_horizon)
