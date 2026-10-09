"""Exogenous signals aligned to the panel: holidays by scope, and city weather.

Everything local is stored once per city (22 rows) with a series-to-city index,
instead of once per store-item series (~200k rows).
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl

from app.features.panel import Panel

# A transferred holiday was moved to another date (that date appears as type
# "Transfer"); a "Work Day" makes up for a bridge holiday and is a normal day.
_NON_HOLIDAY_TYPES = ("Work Day",)


@dataclass(frozen=True)
class Context:
    """Day axes span `panel.start` to the end of the forecast window."""

    series_city: np.ndarray  # [series] index into the city axis
    national_holiday: np.ndarray  # [days]
    local_holiday: np.ndarray  # [cities, days] regional (province) or city holiday
    rain_mm: np.ndarray  # [cities, days]
    temperature: np.ndarray  # [cities, days]


def build_context(panel: Panel, holidays: pl.DataFrame, weather: pl.DataFrame | None) -> Context:
    n_days = panel.sales.shape[1] + panel.horizon
    dates = pl.date_range(
        panel.start, panel.start + timedelta(days=n_days - 1), "1d", eager=True
    ).alias("date")
    day_of = {d: i for i, d in enumerate(dates.to_list())}

    places = panel.keys.select("city", "state").unique().sort("city")
    cities = places["city"].to_list()
    city_index = {c: i for i, c in enumerate(cities)}
    series_city = panel.keys["city"].replace_strict(city_index).to_numpy()

    active = holidays.filter(
        ~pl.col("transferred")
        & ~pl.col("holiday_type").is_in(_NON_HOLIDAY_TYPES)
        & pl.col("date").is_in(dates.implode())
    )
    national = np.zeros(n_days, dtype=np.float32)
    for d in active.filter(pl.col("locale") == "National")["date"]:
        national[day_of[d]] = 1.0

    local = np.zeros((len(cities), n_days), dtype=np.float32)
    for locale, column in (("Regional", "state"), ("Local", "city")):
        for place, d in (
            active.filter(pl.col("locale") == locale).select("locale_name", "date").rows()
        ):
            local[(places[column] == place).to_numpy(), day_of[d]] = 1.0

    rain = np.zeros((len(cities), n_days), dtype=np.float32)
    temperature = np.zeros((len(cities), n_days), dtype=np.float32)
    if weather is not None:
        grid = (
            pl.DataFrame({"city": cities})
            .join(dates.to_frame(), how="cross")
            .join(weather, on=["city", "date"], how="left")
            .sort("city", "date")
        )
        rain = grid["precipitation_sum"].fill_null(0).to_numpy().reshape(len(cities), n_days)
        temperature = (
            grid["temperature_2m_mean"]
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .to_numpy()
            .reshape(len(cities), n_days)
        )

    return Context(
        series_city=series_city,
        national_holiday=national,
        local_holiday=local,
        rain_mm=rain.astype(np.float32),
        temperature=temperature.astype(np.float32),
    )
