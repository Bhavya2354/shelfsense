"""Design matrices: one row per series for a forecast origin, built only from the past.

`origin` is the index of the first forecast day; features read days `< origin`
for sales and days `>= origin` only for inputs known in advance (promotions,
holidays). Targets are the next `horizon` days of log1p sales.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from app.features.context import Context
from app.features.panel import Panel

MEAN_WINDOWS = (3, 7, 14, 28, 56, 112)
LOOKBACK = max(MEAN_WINDOWS)
EWM_ALPHAS = (0.1, 0.3)
CATEGORICAL = ("store_nbr", "family", "item_class", "city", "state", "store_type", "cluster")


@dataclass(frozen=True)
class DesignMatrix:
    x: np.ndarray  # float32 [series, features]
    names: list[str]
    target: np.ndarray | None  # float32 [series, horizon] in log1p units
    weight: np.ndarray  # [series]
    origin: date

    @property
    def categorical(self) -> list[str]:
        return [c for c in CATEGORICAL if c in self.names]

    def column(self, name: str) -> np.ndarray:
        return self.x[:, self.names.index(name)]


def static_codes(keys: pl.DataFrame) -> dict[str, np.ndarray]:
    """Integer codes for categorical attributes, stable for a given panel."""
    codes: dict[str, np.ndarray] = {}
    for name in CATEGORICAL:
        column = keys[name]
        codes[name] = (
            column.cast(pl.Utf8).rank("dense").to_numpy() - 1
            if column.dtype == pl.Utf8
            else column.to_numpy()
        )
    codes["perishable"] = keys["perishable"].cast(pl.Int8).to_numpy()
    # Used for cross-store aggregates only; too many levels to feed models directly.
    codes["item"] = keys["item_nbr"].rank("dense").cast(pl.Int64).to_numpy() - 1
    return codes


def _group_mean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    sums = np.bincount(groups, weights=values)
    counts = np.bincount(groups)
    means: np.ndarray = (sums / np.maximum(counts, 1))[groups].astype(np.float32)
    return means


def build_window(
    panel: Panel,
    ctx: Context,
    origin: date,
    *,
    perishable_weight: float,
    codes: dict[str, np.ndarray] | None = None,
) -> DesignMatrix:
    t = panel.day(origin)
    h = panel.horizon
    if t < LOOKBACK:
        raise ValueError(f"origin {origin} leaves less than {LOOKBACK} days of history")
    y, p = panel.sales, panel.promo
    codes = codes if codes is not None else static_codes(panel.keys)
    f: dict[str, np.ndarray] = {}

    # Level and volatility over several look-backs.
    for n in MEAN_WINDOWS:
        f[f"mean_{n}"] = y[:, t - n : t].mean(axis=1)
    for n in (7, 28):
        f[f"std_{n}"] = y[:, t - n : t].std(axis=1)
    f["max_7"] = y[:, t - 7 : t].max(axis=1)
    for n in (28, LOOKBACK):
        f[f"zero_share_{n}"] = (y[:, t - n : t] == 0).mean(axis=1)

    # Recency: days since the last sale and since the first sale in the look-back.
    sold = y[:, t - LOOKBACK : t] > 0
    any_sale = sold.any(axis=1)
    f["days_since_sale"] = np.where(any_sale, np.argmax(sold[:, ::-1], axis=1), LOOKBACK)
    f["days_on_shelf"] = np.where(any_sale, LOOKBACK - np.argmax(sold, axis=1), 0)

    for alpha in EWM_ALPHAS:
        weights = alpha * (1 - alpha) ** np.arange(56)[::-1]
        f[f"ewm_{alpha}"] = y[:, t - 56 : t] @ (weights / weights.sum()).astype(np.float32)

    # Same-weekday means; column j matches forecast days j, j+7, j+14.
    for j in range(7):
        for weeks in (4, 16):
            days = [t - 7 * w + j for w in range(1, weeks + 1)]
            f[f"dow{j}_mean_{weeks}w"] = y[:, days].mean(axis=1)

    # Promotions: recent intensity and the plan for every forecast day.
    for n in (14, 56):
        f[f"promo_last_{n}"] = p[:, t - n : t].sum(axis=1)
    for k in range(h):
        f[f"promo_d{k}"] = p[:, t + k]
    f["promo_next_total"] = p[:, t : t + h].sum(axis=1)

    # Cross-store and store-family context for the same item.
    item = codes["item"]
    store_family = codes["store_nbr"] * 1000 + codes["family"]
    _, store_family = np.unique(store_family, return_inverse=True)
    for n in (7, 28):
        f[f"item_mean_{n}"] = _group_mean(f[f"mean_{n}"], item)
    f["item_promo_next_share"] = _group_mean(f["promo_next_total"] / h, item)
    f["store_family_mean_28"] = _group_mean(f["mean_28"], store_family)

    # Calendar and weather known for the forecast days or from the recent past.
    city = ctx.series_city
    for k in range(h):
        f[f"national_holiday_d{k}"] = np.full(panel.n_series, ctx.national_holiday[t + k])
        f[f"local_holiday_d{k}"] = ctx.local_holiday[city, t + k]
    f["rain_last_7"] = ctx.rain_mm[city, t - 7 : t].sum(axis=1)
    f["temperature_last_7"] = ctx.temperature[city, t - 7 : t].mean(axis=1)

    for name in (*CATEGORICAL, "perishable"):
        f[name] = codes[name]

    names = list(f)
    x = np.column_stack([f[n].astype(np.float32) for n in names])
    observed = panel.sales.shape[1]
    target = y[:, t : t + h].copy() if t + h <= observed else None
    weight = np.where(codes["perishable"] == 1, perishable_weight, 1.0).astype(np.float32)
    return DesignMatrix(x=x, names=names, target=target, weight=weight, origin=origin)


def stack_windows(windows: list[DesignMatrix]) -> DesignMatrix:
    """Concatenate several origins into one training set (rows = series x origins)."""
    targets = [w.target for w in windows]
    if any(tg is None for tg in targets):
        raise ValueError("training windows need observed targets")
    return DesignMatrix(
        x=np.concatenate([w.x for w in windows]),
        names=windows[0].names,
        target=np.concatenate(targets),  # type: ignore[arg-type]
        weight=np.concatenate([w.weight for w in windows]),
        origin=windows[-1].origin,
    )
