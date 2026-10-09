"""Training pipeline: rolling-origin backtests, then the forecast that gets released.

Every fold starts on the same weekday as the release forecast, and each model
only sees data that was available before the fold's first forecast day.
"""

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from app import config
from app.config import ForecastSettings
from app.evaluation import metrics
from app.evaluation.significance import bootstrap_nwrmsle_difference, diebold_mariano
from app.features.context import Context, build_context
from app.features.panel import Panel, build_panel
from app.features.windows import DesignMatrix, build_window, stack_windows, static_codes
from app.forecasting import baselines, embedding_mlp, gradient_boosting  # noqa: F401 (registration)
from app.forecasting.ensemble import blend, fit_blend_weights
from app.forecasting.family import BOTTOM_LEVEL, FamilyHierarchy, forecast_family, load_hierarchy
from app.forecasting.foundation import forecast_foundation
from app.forecasting.intervals import ResidualQuantiles, fit_residual_quantiles
from app.forecasting.registry import create_item_model
from app.forecasting.tuning import tune_lightgbm, tune_mlp
from app.storage.catalog import Table
from app.storage.database import create_db_engine, session_factory, transaction
from app.storage.repositories import ModelRunRepository
from app.storage.warehouse import open_warehouse

logger = logging.getLogger(__name__)

ENSEMBLE = "ensemble"
BOTTOM_UP = "item_ensemble_bottom_up"


@dataclass
class ItemFold:
    first_day: date
    actual: np.ndarray
    weight: np.ndarray
    velocity: np.ndarray
    predictions: dict[str, np.ndarray]
    blend_weights: dict[str, float]


@dataclass
class ItemRelease:
    first_day: date
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    promo: np.ndarray
    blend_weights: dict[str, float]
    feature_importance: dict[str, float] = field(default_factory=dict)


class ItemLevelTrainer:
    def __init__(self, panel: Panel, ctx: Context, settings: ForecastSettings) -> None:
        self._panel = panel
        self._ctx = ctx
        self._s = settings
        self._codes = static_codes(panel.keys)
        horizon, stride = settings.forecast_horizon, settings.window_stride_days
        self._min_gap = math.ceil(horizon / stride)
        self.release_day = panel.last_observed + timedelta(days=1)

    def fold_days(self) -> list[date]:
        stride = self._s.window_stride_days
        days = [
            self.release_day - timedelta(days=stride * (self._min_gap + 2 * i))
            for i in range(self._s.backtest_folds)
        ]
        return sorted(days)

    def tune(self) -> dict[str, Any]:
        """Search hyperparameters on windows that end before the first backtest fold."""
        stride = timedelta(days=self._s.window_stride_days)
        first_fold = self.fold_days()[0]
        valid = self._window(first_fold - stride * self._min_gap)
        train = stack_windows(
            [
                self._window(first_fold - stride * k)
                for k in range(self._min_gap + 1, self._min_gap + 1 + self._s.item_train_windows)
            ]
        )
        tuned: dict[str, Any] = {}
        if "lightgbm" in self._s.item_models:
            tuned |= tune_lightgbm(train, valid, self._s)
        if "embedding_mlp" in self._s.item_models:
            tuned |= tune_mlp(train, valid, self._s)
        self._s = self._s.model_copy(update=tuned)
        return tuned

    @property
    def settings(self) -> ForecastSettings:
        return self._s

    def _window(self, origin: date) -> DesignMatrix:
        return build_window(
            self._panel,
            self._ctx,
            origin,
            perishable_weight=self._s.perishable_weight,
            codes=self._codes,
        )

    def _fit_predict(
        self, first_day: date
    ) -> tuple[dict[str, np.ndarray], dict[str, float], DesignMatrix, dict[str, float]]:
        stride = timedelta(days=self._s.window_stride_days)
        valid = self._window(first_day - stride * self._min_gap)
        train = stack_windows(
            [
                self._window(first_day - stride * k)
                for k in range(self._min_gap + 1, self._min_gap + 1 + self._s.item_train_windows)
            ]
        )
        target = self._window(first_day)
        logger.info(
            "fitting item models", extra={"first_day": first_day, "train_rows": train.x.shape[0]}
        )
        valid_preds: dict[str, np.ndarray] = {}
        target_preds: dict[str, np.ndarray] = {}
        importance: dict[str, float] = {}
        for name in self._s.item_models:
            model = create_item_model(name, self._s)
            model.fit(train, valid)
            valid_preds[name] = model.predict(valid)
            target_preds[name] = model.predict(target)
            importance = getattr(model, "feature_importance", importance)
            logger.info("model done", extra={"model": name, "first_day": first_day})
        if valid.target is None:
            raise ValueError("validation window has no targets")
        weights = fit_blend_weights(valid_preds, valid.target, valid.weight)
        target_preds[ENSEMBLE] = blend(target_preds, weights)
        return target_preds, weights, target, importance

    def backtest(self) -> list[ItemFold]:
        folds = []
        for first_day in self.fold_days():
            preds, weights, target, _ = self._fit_predict(first_day)
            if target.target is None:
                raise ValueError(f"fold {first_day} has no observed targets")
            folds.append(
                ItemFold(
                    first_day,
                    target.target,
                    target.weight,
                    target.column("mean_28"),
                    preds,
                    weights,
                )
            )
        return folds

    def release(self, calibration: ResidualQuantiles) -> ItemRelease:
        preds, weights, target, importance = self._fit_predict(self.release_day)
        point = preds[ENSEMBLE]
        lower, upper = calibration.interval(point, target.column("mean_28"))
        t = self._panel.day(self.release_day)
        promo = self._panel.promo[:, t : t + self._s.forecast_horizon]
        return ItemRelease(self.release_day, point, lower, upper, promo, weights, importance)


# --------------------------------------------------------------------- scoring


def _item_scores(folds: list[ItemFold]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, list[float]]] = {}
    for fold in folds:
        for name, pred in fold.predictions.items():
            overall = metrics.item_scores(pred, fold.actual, fold.weight)
            for metric, value in overall.items():
                rows.append(
                    {
                        "model": name,
                        "cutoff": fold.first_day,
                        "horizon": 0,
                        "metric": metric,
                        "value": value,
                    }
                )
                summary.setdefault(name, {}).setdefault(metric, []).append(value)
            by_day = metrics.nwrmsle_by_day(pred, fold.actual, fold.weight)
            rows += [
                {
                    "model": name,
                    "cutoff": fold.first_day,
                    "horizon": d + 1,
                    "metric": "nwrmsle",
                    "value": float(v),
                }
                for d, v in enumerate(by_day)
            ]
    means = {m: {k: float(np.mean(v)) for k, v in s.items()} for m, s in summary.items()}
    return rows, means


def _daily_loss(fold: ItemFold, name: str) -> np.ndarray:
    w = fold.weight[:, None]
    loss: np.ndarray = np.sum(w * (fold.predictions[name] - fold.actual) ** 2, axis=0) / np.sum(w)
    return loss


def _item_significance(
    folds: list[ItemFold], best: str, settings: ForecastSettings
) -> dict[str, Any]:
    latest = folds[-1]
    out: dict[str, Any] = {}
    for name in latest.predictions:
        if name == best:
            continue
        loss_best = np.concatenate([_daily_loss(f, best) for f in folds])
        loss_other = np.concatenate([_daily_loss(f, name) for f in folds])
        out[name] = {
            "diebold_mariano": diebold_mariano(loss_best, loss_other, max_lag=6),
            "bootstrap": bootstrap_nwrmsle_difference(
                latest.predictions[best],
                latest.predictions[name],
                latest.actual,
                latest.weight,
                resamples=settings.bootstrap_resamples,
                seed=settings.random_seed,
            ),
        }
    return out


def _seasonal_naive_scale(frame: pl.DataFrame, first_day: date, days: int) -> pl.DataFrame:
    history = frame.filter(
        (pl.col("ds") < first_day) & (pl.col("ds") >= first_day - timedelta(days=days))
    ).sort("unique_id", "ds")
    return history.group_by("unique_id").agg(
        (pl.col("y") - pl.col("y").shift(7)).abs().mean().alias("scale")
    )


def _family_scores(
    backtest: pl.DataFrame, hierarchy: FamilyHierarchy, settings: ForecastSettings
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    frame = pl.from_pandas(hierarchy.frame[["unique_id", "ds", "y"]]).with_columns(
        pl.col("ds").cast(pl.Date)
    )
    bottom_ids = set(hierarchy.tags[BOTTOM_LEVEL])  # type: ignore[call-overload]
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, list[float]]] = {}
    for (first_day,), fold in backtest.group_by("first_day", maintain_order=True):
        scale = _seasonal_naive_scale(frame, first_day, settings.family_history_days)
        scored = fold.join(frame, on=["unique_id", "ds"]).join(scale, on="unique_id")
        for level, ids in (("store_family", bottom_ids), ("all_levels", None)):
            part = scored if ids is None else scored.filter(pl.col("unique_id").is_in(list(ids)))
            for (model,), group in part.group_by("model_name", maintain_order=True):
                g = group.sort("unique_id", "ds")
                n_series = g["unique_id"].n_unique()
                pred = g["p50"].to_numpy().reshape(n_series, -1)
                actual = g["y"].to_numpy().reshape(n_series, -1)
                scales = (
                    g.group_by("unique_id", maintain_order=True)
                    .agg(pl.col("scale").first())["scale"]
                    .to_numpy()
                )
                values = {
                    "rmsle": metrics.rmsle(pred, actual),
                    "wape": metrics.wape(pred, actual),
                    "mase": metrics.mase(pred, actual, scales),
                    "bias": metrics.bias(pred, actual),
                }
                if (
                    level == "store_family"
                    and g["p10"].is_not_null().all()
                    and not g["p10"].is_nan().any()
                ):
                    values["coverage_80"] = metrics.coverage(
                        g["p10"].to_numpy(), g["p90"].to_numpy(), g["y"].to_numpy()
                    )
                for metric, value in values.items():
                    name = f"{level}:{metric}"
                    rows.append(
                        {
                            "model": str(model),
                            "cutoff": first_day,
                            "horizon": 0,
                            "metric": name,
                            "value": value,
                        }
                    )
                    summary.setdefault(str(model), {}).setdefault(name, []).append(value)
    means = {m: {k: float(np.mean(v)) for k, v in s.items()} for m, s in summary.items()}
    return rows, means


def _day_column(first_day: date, horizon: int, repeats: int) -> pl.Series:
    """Forecast dates for `repeats` series laid out series-major, as a Date column."""
    days = pl.date_range(first_day, first_day + timedelta(days=horizon - 1), "1d", eager=True)
    return pl.Series(np.tile(days.to_numpy(), repeats)).cast(pl.Date)


def _bottom_up(panel: Panel, fold: ItemFold) -> pl.DataFrame:
    """Sum item-level ensemble forecasts (in units) up to store-family series."""
    horizon = fold.actual.shape[1]
    units = np.expm1(fold.predictions[ENSEMBLE])
    keys = panel.keys.select(
        pl.format("all/{}/{}", pl.col("store_nbr"), pl.col("family")).alias("unique_id")
    )
    long = pl.DataFrame(
        {
            "unique_id": np.repeat(keys["unique_id"].to_numpy(), horizon),
            "ds": _day_column(fold.first_day, horizon, panel.n_series),
            "p50": units.ravel(),
        }
    )
    return (
        long.group_by("unique_id", "ds")
        .agg(pl.col("p50").sum())
        .with_columns(
            pl.lit(BOTTOM_UP).alias("model_name"),
            pl.lit(None, dtype=pl.Float64).alias("p10"),
            pl.lit(None, dtype=pl.Float64).alias("p90"),
            pl.lit(fold.first_day).alias("first_day"),
        )
    )


# ----------------------------------------------------------------- artifacts


def _item_frame(panel: Panel, first_day: date, columns: dict[str, np.ndarray]) -> pl.DataFrame:
    horizon = next(iter(columns.values())).shape[1]
    base = {
        "store_nbr": np.repeat(panel.keys["store_nbr"].to_numpy(), horizon),
        "item_nbr": np.repeat(panel.keys["item_nbr"].to_numpy(), horizon),
        "target_date": _day_column(first_day, horizon, panel.n_series),
    }
    return pl.DataFrame(base | {k: v.ravel() for k, v in columns.items()})


def run_training() -> dict[str, Any]:
    storage = config.storage_settings()
    settings = config.forecast_settings()
    out_dir = storage.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open_warehouse(storage) as wh:
        last_observed = wh.scalar("SELECT max(date) FROM sales")
        panel = build_panel(
            wh, last_observed, settings.item_history_days, settings.forecast_horizon
        )
        holidays = wh.frame("SELECT * FROM holidays")
        weather = wh.frame("SELECT * FROM weather") if wh.has(Table.WEATHER) else None
        hierarchy = load_hierarchy(
            wh, last_observed, settings.family_history_days, settings.forecast_horizon
        )
    ctx = build_context(panel, holidays, weather)
    logger.info("panel ready", extra={"series": panel.n_series, "days": panel.sales.shape[1]})

    # Item level: backtest, calibrate intervals on the older fold, check them on the newer.
    item = ItemLevelTrainer(panel, ctx, settings)
    tuned_params = item.tune()
    logger.info("tuned hyperparameters", extra={"params": tuned_params})
    settings = item.settings
    folds = item.backtest()
    item_rows, item_summary = _item_scores(folds)
    calibrate_on, check_on = folds[0], folds[-1]
    calibration = fit_residual_quantiles(
        calibrate_on.predictions[ENSEMBLE],
        calibrate_on.actual,
        calibrate_on.velocity,
        settings.interval_quantiles,
        settings.velocity_buckets,
    )
    lo, hi = calibration.interval(check_on.predictions[ENSEMBLE], check_on.velocity)
    interval_coverage = metrics.coverage(lo, hi, check_on.actual)
    best_item = min(item_summary, key=lambda m: item_summary[m]["nwrmsle"])
    significance = _item_significance(folds, best_item, settings)

    release_calibration = fit_residual_quantiles(
        check_on.predictions[ENSEMBLE],
        check_on.actual,
        check_on.velocity,
        settings.interval_quantiles,
        settings.velocity_buckets,
    )
    release = item.release(release_calibration)

    # Family level, plus the item ensemble summed up for comparison.
    family_parts = []
    for fold in folds:
        fc = pl.concat(
            [
                forecast_family(hierarchy, fold.first_day, settings),
                forecast_foundation(hierarchy, fold.first_day, settings, out_dir),
            ],
            how="vertical_relaxed",
        ).with_columns(pl.lit(fold.first_day).alias("first_day"))
        family_parts += [fc, _bottom_up(panel, fold).select(fc.columns)]
    family_backtest = pl.concat(family_parts, how="vertical_relaxed")
    family_rows, family_summary = _family_scores(family_backtest, hierarchy, settings)
    family_release = pl.concat(
        [
            forecast_family(hierarchy, item.release_day, settings),
            forecast_foundation(hierarchy, item.release_day, settings, out_dir),
        ],
        how="vertical_relaxed",
    )

    # Artifacts consumed by the publishing step.
    backtest_frames = []
    for fold in folds:
        p_lo, p_hi = calibration.interval(fold.predictions[ENSEMBLE], fold.velocity)
        backtest_frames.append(
            _item_frame(
                panel,
                fold.first_day,
                {
                    "actual": np.expm1(fold.actual),
                    "p10": np.expm1(p_lo),
                    "p50": np.expm1(fold.predictions[ENSEMBLE]),
                    "p90": np.expm1(p_hi),
                    "naive": np.expm1(fold.predictions["moving_average"]),
                },
            ).with_columns(pl.lit(fold.first_day).alias("first_day"))
        )
    pl.concat(backtest_frames).write_parquet(out_dir / "item_backtest.parquet")
    _item_frame(
        panel,
        release.first_day,
        {
            "p10": np.expm1(release.lower),
            "p50": np.expm1(release.point),
            "p90": np.expm1(release.upper),
            "onpromotion": release.promo > 0,
        },
    ).write_parquet(out_dir / "item_release.parquet")
    family_backtest.write_parquet(out_dir / "family_backtest.parquet")
    family_release.write_parquet(out_dir / "family_release.parquet")

    run_ids = _persist_runs(
        item_rows=item_rows,
        item_summary=item_summary,
        family_rows=family_rows,
        family_summary=family_summary,
        panel=panel,
        settings=settings,
        release=release,
    )
    report = {
        "forecast_origin": str(panel.last_observed),
        "release_first_day": str(release.first_day),
        "series": panel.n_series,
        "folds": [str(f.first_day) for f in folds],
        "item_scores": item_summary,
        "family_scores": family_summary,
        "best_item_model": best_item,
        "blend_weights": {str(f.first_day): f.blend_weights for f in folds}
        | {"release": release.blend_weights},
        "interval_coverage": {
            "target": settings.interval_quantiles[1] - settings.interval_quantiles[0],
            "observed": interval_coverage,
        },
        "significance_vs_best": significance,
        "tuned_hyperparameters": tuned_params,
        "feature_importance": dict(list(release.feature_importance.items())[:25]),
        "run_ids": run_ids,
    }
    (out_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def _persist_runs(
    *,
    item_rows: list[dict[str, Any]],
    item_summary: dict[str, dict[str, float]],
    family_rows: list[dict[str, Any]],
    family_summary: dict[str, dict[str, float]],
    panel: Panel,
    settings: ForecastSettings,
    release: ItemRelease,
) -> dict[str, str]:
    factory = session_factory(create_db_engine(config.database_settings()))
    run_ids: dict[str, str] = {}
    params = settings.model_dump(mode="json")
    with transaction(factory) as session:
        repo = ModelRunRepository(session)
        for level, rows, summary in (
            ("item", item_rows, item_summary),
            ("family", family_rows, family_summary),
        ):
            frame = pl.DataFrame(rows)
            for model, model_metrics in summary.items():
                run_id: uuid.UUID = repo.record(
                    model_name=model,
                    level=level,
                    params=params
                    | ({"blend_weights": release.blend_weights} if model == ENSEMBLE else {}),
                    metrics=model_metrics,
                    train_start=panel.start,
                    train_end=panel.last_observed,
                    scores=frame.filter(pl.col("model") == model).drop("model"),
                )
                run_ids[f"{level}:{model}"] = str(run_id)
    return run_ids
