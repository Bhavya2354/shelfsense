"""Unit-root (ADF) and level-stationarity (KPSS) tests, which make opposite null hypotheses."""

import warnings
from typing import Any

import numpy as np
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

Finding = tuple[str, dict[str, Any]]


def _verdict(adf_p: float, kpss_p: float, alpha: float) -> str:
    adf_stationary, kpss_stationary = adf_p < alpha, kpss_p >= alpha
    if adf_stationary and kpss_stationary:
        return "stationary"
    if not adf_stationary and not kpss_stationary:
        return "non-stationary"
    return "trend-stationary" if kpss_stationary else "difference-stationary"


def stationarity_tests(series: np.ndarray, alpha: float) -> list[Finding]:
    findings: list[Finding] = []
    for name, values in (("log_level", series), ("log_diff", np.diff(series))):
        adf_stat, adf_p, *_ = adfuller(values, autolag="AIC", result_object=False)
        with warnings.catch_warnings():
            # KPSS p-values are table-interpolated and clipped to [0.01, 0.1].
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_stat, kpss_p, *_ = kpss(values, regression="c", nlags="auto", result_object=False)
        findings.append(
            (
                name,
                {
                    "adf_statistic": float(adf_stat),
                    "adf_pvalue": float(adf_p),
                    "kpss_statistic": float(kpss_stat),
                    "kpss_pvalue": float(kpss_p),
                    "verdict": _verdict(float(adf_p), float(kpss_p), alpha),
                },
            )
        )
    return findings
