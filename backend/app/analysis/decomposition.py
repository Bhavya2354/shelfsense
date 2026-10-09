"""Trend and seasonality: MSTL decomposition and Hyndman's strength measures."""

import calendar
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from statsmodels.tsa.seasonal import MSTL, STL

Finding = tuple[str, dict[str, Any]]


def _strength(component: np.ndarray, remainder: np.ndarray) -> float:
    """1 - Var(R) / Var(C + R), floored at zero (Wang, Smith & Hyndman, 2006)."""
    return float(max(0.0, 1 - np.var(remainder) / np.var(component + remainder)))


def date_bounds(frame: pl.DataFrame) -> tuple[date, date]:
    first, last = frame.select(pl.col("date").min(), pl.col("date").max()).row(0)
    return first, last


def daily_index(frame: pl.DataFrame) -> pl.Series:
    first, last = date_bounds(frame)
    return pl.Series("date", pl.date_range(first, last, "1d", eager=True))


def complete_daily(frame: pl.DataFrame, value: str) -> np.ndarray:
    """Fill closed days (e.g. 25 December) by linear interpolation so periods stay aligned."""
    full = daily_index(frame).to_frame()
    return (
        full.join(frame.select("date", value), on="date", how="left")
        .with_columns(pl.col(value).interpolate())
        .get_column(value)
        .to_numpy()
    )


def national_seasonality(daily: pl.DataFrame) -> list[Finding]:
    series = np.log1p(complete_daily(daily, "unit_sales"))
    result = MSTL(series, periods=(7, 365), stl_kwargs={"robust": True}).fit()
    weekly, yearly = result.seasonal[:, 0], result.seasonal[:, 1]
    start_dow = date_bounds(daily)[0].weekday()
    dow_effect = [float(np.mean(weekly[(d - start_dow) % 7 :: 7])) for d in range(7)]
    months = daily_index(daily).dt.month().to_numpy()
    month_effect = [float(np.mean(yearly[months == m])) for m in range(1, 13)]
    return [
        (
            "national",
            {
                "trend_strength": _strength(result.trend, result.resid),
                "weekly_strength": _strength(weekly, result.resid),
                "yearly_strength": _strength(yearly, result.resid),
                "weekday_effect_pct": {
                    calendar.day_name[d]: round(100 * (np.exp(v) - 1), 2)
                    for d, v in enumerate(dow_effect)
                },
                "month_effect_pct": {
                    calendar.month_abbr[m + 1]: round(100 * (np.exp(v) - 1), 2)
                    for m, v in enumerate(month_effect)
                },
                "trend_growth_pct": round(
                    100 * (np.exp(result.trend[-365:].mean() - result.trend[:365].mean()) - 1), 2
                ),
            },
        )
    ]


def family_weekly_strength(daily: pl.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    for (family,), group in daily.group_by("family", maintain_order=True):
        values = complete_daily(group, "unit_sales")
        if np.count_nonzero(values) < 8 * 7:
            continue
        fit = STL(np.log1p(values), period=7, robust=True).fit()
        findings.append(
            (
                str(family),
                {
                    "weekly_strength": _strength(fit.seasonal, fit.resid),
                    "trend_strength": _strength(fit.trend, fit.resid),
                    "mean_daily_units": float(values.mean()),
                },
            )
        )
    return findings
