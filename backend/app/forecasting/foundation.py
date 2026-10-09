"""Chronos-2 foundation model, used zero-shot and after fine-tuning on Favorita.

Both variants see the same store-family history and the planned promotions for
the forecast days (Chronos-2 accepts known future covariates), so they compare
directly with the N-BEATS and TFT models trained from scratch.
"""

import logging
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from app.config import ForecastSettings
from app.forecasting.family import FamilyHierarchy

logger = logging.getLogger(__name__)

ZERO_SHOT = "chronos2_zero_shot"
FINE_TUNED = "chronos2_finetuned"
_QUANTILES = [0.1, 0.5, 0.9]


def _split(
    data: FamilyHierarchy, first_day: date, settings: ForecastSettings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = pd.Timestamp(first_day)
    start = first - pd.Timedelta(days=settings.chronos_context_days)
    end = first + pd.Timedelta(days=settings.forecast_horizon)
    frame = data.frame
    history = frame[(frame["ds"] >= start) & (frame["ds"] < first)][
        ["unique_id", "ds", "y", "promo"]
    ]
    future = frame[(frame["ds"] >= first) & (frame["ds"] < end)][["unique_id", "ds", "promo"]]
    return history, future


def _training_inputs(history: pd.DataFrame) -> list[dict[str, Any]]:
    """One input per series; `promo` is marked as known for future steps."""
    inputs = []
    for _, series in history.sort_values("ds").groupby("unique_id", sort=False):
        inputs.append(
            {
                "target": series["y"].to_numpy(dtype=np.float32),
                "past_covariates": {"promo": series["promo"].to_numpy(dtype=np.float32)},
                "future_covariates": {"promo": None},
            }
        )
    return inputs


def _predict(
    pipeline: Any, history: pd.DataFrame, future: pd.DataFrame, h: int, name: str
) -> pl.DataFrame:
    forecast = pipeline.predict_df(
        history,
        future_df=future,
        id_column="unique_id",
        timestamp_column="ds",
        target="y",
        prediction_length=h,
        quantile_levels=_QUANTILES,
    )
    return pl.from_pandas(
        pd.DataFrame(
            {
                "unique_id": forecast["unique_id"],
                "ds": forecast["ds"],
                "model_name": name,
                "p10": forecast["0.1"],
                "p50": forecast["0.5"],
                "p90": forecast["0.9"],
            }
        )
    ).with_columns(
        pl.col("ds").cast(pl.Date),
        *(pl.col(c).cast(pl.Float64).clip(lower_bound=0) for c in ("p10", "p50", "p90")),
    )


def forecast_foundation(
    data: FamilyHierarchy, first_day: date, settings: ForecastSettings, work_dir: Path
) -> pl.DataFrame:
    import torch
    from chronos import Chronos2Pipeline

    torch.set_num_threads(settings.n_jobs)
    torch.manual_seed(settings.random_seed)
    h = settings.forecast_horizon
    history, future = _split(data, first_day, settings)
    base = Chronos2Pipeline.from_pretrained(settings.chronos_model_id, device_map="cpu")
    zero_shot = _predict(base, history, future, h, ZERO_SHOT)
    logger.info("chronos zero-shot done", extra={"first_day": first_day})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = base.fit(
            _training_inputs(history),
            prediction_length=h,
            learning_rate=settings.chronos_finetune_learning_rate,
            num_steps=settings.chronos_finetune_steps,
            batch_size=settings.chronos_batch_size,
            context_length=settings.chronos_context_days - h,
            output_dir=work_dir / f"chronos-{first_day}",
            remove_printer_callback=True,
        )
    fine_tuned = _predict(tuned, history, future, h, FINE_TUNED)
    logger.info("chronos fine-tuned done", extra={"first_day": first_day})
    return pl.concat([zero_shot, fine_tuned], how="vertical_relaxed")
