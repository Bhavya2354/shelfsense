"""Causal-style effect estimates: calendar, promotions, rain and oil.

Each estimate reports a percent effect with a 95% confidence interval and uses
standard errors that respect the data's dependence: Newey-West for a single
daily series, cluster-robust for panels.
"""

import warnings
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import grangercausalitytests

from app.analysis.decomposition import complete_daily, daily_index

Finding = tuple[str, dict[str, Any]]
_Z = 1.959964
_MID_MONTH_PAYDAY = 15


def _pct(coef: float) -> float:
    return round(float(100 * (np.exp(coef) - 1)), 2)


def _effect(coef: float, se: float, p: float | None = None) -> dict[str, Any]:
    if p is None:
        p = float(2 * stats.norm.sf(abs(coef / se)))
    return {
        "effect_pct": _pct(coef),
        "ci_low_pct": _pct(coef - _Z * se),
        "ci_high_pct": _pct(coef + _Z * se),
        "p_value": float(p),
        "coef": float(coef),
        "std_error": float(se),
    }


# ---------------------------------------------------------------- calendar


def calendar_effects(daily: pl.DataFrame, holidays: pl.DataFrame, hac_lags: int) -> list[Finding]:
    """Log national sales on weekday, month, trend, payday and national-event indicators."""
    y = np.log1p(complete_daily(daily, "unit_sales"))
    frame = (
        daily_index(daily)
        .to_frame()
        .with_columns(
            trend=(pl.col("date") - pl.col("date").min()).dt.total_days() / 365.25,
            dow=pl.col("date").dt.weekday(),
            month=pl.col("date").dt.month(),
            payday=(pl.col("date").dt.day() == _MID_MONTH_PAYDAY)
            | (pl.col("date") == pl.col("date").dt.month_end()),
        )
    )
    frame = frame.with_columns(payday_next=pl.col("payday").shift(1, fill_value=False))

    national = holidays.filter((pl.col("locale") == "National") & ~pl.col("transferred"))
    quake = national.filter(pl.col("description").str.contains("Terremoto"))
    other = national.join(quake, on="date", how="anti")
    indicators = {
        f"national_{t.lower().replace(' ', '_')}": other.filter(pl.col("holiday_type") == t)["date"]
        for t in other["holiday_type"].unique().sort()
    }
    indicators["earthquake_relief_period"] = quake["date"]
    for name, event_dates in indicators.items():
        frame = frame.with_columns(pl.col("date").is_in(event_dates.implode()).alias(name))

    design = frame.to_dummies(["dow", "month"], drop_first=True).drop("date").cast(pl.Float64)
    model = OLS(y, add_constant(design.to_numpy()))
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    names = ["const", *design.columns]
    reported = ["payday", "payday_next", *indicators]
    return [
        (
            name,
            {**_effect(fit.params[i], fit.bse[i], fit.pvalues[i]), "days": int(design[name].sum())},
        )
        for i, name in enumerate(names)
        if name in reported
    ] + [("model", {"r_squared": float(fit.rsquared), "observations": int(fit.nobs)})]


# ------------------------------------------------------- panel estimators


def _demean(values: np.ndarray, groups: list[np.ndarray], iterations: int) -> np.ndarray:
    """Remove several sets of fixed effects by alternating projections."""
    out = values.astype(np.float64).copy()
    for _ in range(iterations):
        for g in groups:
            sums = np.bincount(g, weights=out)
            counts = np.bincount(g)
            out -= (sums / np.maximum(counts, 1))[g]
    return out


def _fe_ols(
    y: np.ndarray,
    x: np.ndarray,
    groups: list[np.ndarray],
    cluster: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Within estimator with cluster-robust (CR1) standard errors."""
    y_t = _demean(y, groups, iterations)
    x_t = np.column_stack([_demean(x[:, j], groups, iterations) for j in range(x.shape[1])])
    xtx_inv = np.linalg.inv(x_t.T @ x_t)
    beta = xtx_inv @ x_t.T @ y_t
    resid = y_t - x_t @ beta
    scores = np.zeros((cluster.max() + 1, x.shape[1]))
    np.add.at(scores, cluster, x_t * resid[:, None])
    n_clusters = int(np.unique(cluster).size)
    meat = scores.T @ scores
    correction = n_clusters / (n_clusters - 1)
    cov = correction * xtx_inv @ meat @ xtx_inv
    return beta, np.sqrt(np.diag(cov))


def _codes(series: pl.Series) -> np.ndarray:
    return series.rank("dense").cast(pl.Int64).to_numpy() - 1


def promotion_lift(panel: pl.DataFrame, iterations: int, min_series: int) -> list[Finding]:
    """Per family: log sales on promotion with store-item and date fixed effects."""
    findings: list[Finding] = []
    for (family,), group in panel.group_by("family", maintain_order=True):
        series = _codes(group.select(pl.struct("store_nbr", "item_nbr").hash()).to_series())
        n_series = int(series.max() + 1)
        if n_series < min_series or group["promo"].std() == 0:
            continue
        beta, se = _fe_ols(
            group["log_sales"].to_numpy(),
            group["promo"].to_numpy()[:, None].astype(np.float64),
            [series, _codes(group["date"])],
            series,
            iterations,
        )
        findings.append(
            (
                str(family),
                {
                    **_effect(beta[0], se[0]),
                    "series": n_series,
                    "promo_share": float(group["promo"].mean()),  # type: ignore[arg-type]
                },
            )
        )
    return sorted(findings, key=lambda f: -f[1]["effect_pct"])


def rain_effect(
    transactions: pl.DataFrame, weather: pl.DataFrame, heavy_rain_mm: float, iterations: int
) -> list[Finding]:
    """Store transactions on local rain, with store and date fixed effects.

    Date effects absorb everything national (holidays, paydays, trend), so the
    estimate comes only from rain differing between cities on the same day.
    """
    panel = transactions.join(weather, on=["city", "date"], how="inner").drop_nulls(
        "precipitation_sum"
    )
    stores = _codes(panel["store_nbr"])
    x = np.column_stack(
        [
            (panel["precipitation_sum"] >= heavy_rain_mm).cast(pl.Float64).to_numpy(),
            panel["temperature_2m_mean"].fill_null(strategy="mean").to_numpy(),
        ]
    )
    beta, se = _fe_ols(
        np.log(panel["transactions"].to_numpy()),
        x,
        [stores, _codes(panel["date"])],
        stores,
        iterations,
    )
    return [
        ("heavy_rain_day", {**_effect(beta[0], se[0]), "threshold_mm": heavy_rain_mm}),
        ("per_degree_celsius", _effect(beta[1], se[1])),
        ("panel", {"observations": panel.height, "stores": int(stores.max() + 1)}),
    ]


# -------------------------------------------------------------------- oil


def oil_granger(daily: pl.DataFrame, oil: pl.DataFrame, max_lag: int) -> list[Finding]:
    """Does last weeks' oil price movement help predict this week's sales growth?"""
    weekly = (
        daily.join(oil, on="date", how="left")
        .sort("date")
        .group_by_dynamic("date", every="1w")
        .agg(pl.col("unit_sales").sum(), pl.col("price").mean())
        .drop_nulls()
        .with_columns(
            sales_growth=pl.col("unit_sales").log().diff(),
            oil_change=pl.col("price").log().diff(),
        )
        .drop_nulls()
    )
    data = weekly.select("sales_growth", "oil_change").to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        tests = grangercausalitytests(data, maxlag=max_lag)
    by_lag = {
        str(lag): {
            "f_statistic": float(res[0]["ssr_ftest"][0]),
            "p_value": float(res[0]["ssr_ftest"][1]),
        }
        for lag, res in tests.items()
    }
    level_corr = float(np.corrcoef(weekly["unit_sales"], weekly["price"])[0, 1])
    growth_corr = float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
    return [
        (
            "oil_to_sales",
            {
                "by_lag": by_lag,
                "min_p_value": min(v["p_value"] for v in by_lag.values()),
                "weeks": weekly.height,
                "level_correlation": level_corr,
                "growth_correlation": growth_corr,
            },
        )
    ]
