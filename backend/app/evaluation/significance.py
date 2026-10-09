"""Is one model's lower error real or noise? Two complementary tests.

* Diebold-Mariano (with the Harvey-Leybourne-Newbold small-sample correction)
  on the daily loss series: accounts for errors being correlated in time.
* A cluster bootstrap over series for the NWRMSLE difference: accounts for
  errors being correlated within a store-item across forecast days.
"""

import numpy as np
from scipy import stats


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, max_lag: int) -> dict[str, float]:
    """Tests H0: equal expected loss. Negative statistic means model A is more accurate."""
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = d.size
    centred = d - d.mean()
    lags = min(max_lag, n - 2)
    variance = np.dot(centred, centred) / n
    for k in range(1, lags + 1):
        variance += 2 * np.dot(centred[k:], centred[:-k]) / n
    statistic = d.mean() / np.sqrt(max(variance, 1e-18) / n)
    h = lags + 1
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    statistic *= correction
    p_value = 2 * stats.t.sf(abs(statistic), df=n - 1)
    return {"statistic": float(statistic), "p_value": float(p_value), "periods": float(n)}


def bootstrap_nwrmsle_difference(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    actual: np.ndarray,
    weight: np.ndarray,
    *,
    resamples: int,
    seed: int,
    chunk: int = 50,
) -> dict[str, float]:
    """95% interval for NWRMSLE(A) - NWRMSLE(B), resampling whole series."""
    horizon = actual.shape[1]
    sq_a = (weight[:, None] * (pred_a - actual) ** 2).sum(axis=1)
    sq_b = (weight[:, None] * (pred_b - actual) ** 2).sum(axis=1)
    mass = weight * horizon
    rng = np.random.default_rng(seed)
    n = actual.shape[0]
    diffs: list[np.ndarray] = []
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(size)])
        total = counts @ mass
        diffs.append(np.sqrt(counts @ sq_a / total) - np.sqrt(counts @ sq_b / total))
    sample = np.concatenate(diffs)
    point = float(np.sqrt(sq_a.sum() / mass.sum()) - np.sqrt(sq_b.sum() / mass.sum()))
    low, high = np.quantile(sample, [0.025, 0.975])
    return {
        "difference": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "share_a_better": float((sample < 0).mean()),
    }
