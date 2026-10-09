"""Statistical analysis pipeline: every finding lands in Postgres and a JSON artifact."""

import json
import logging
from collections.abc import Callable
from datetime import timedelta

import numpy as np

from app import config
from app.analysis import aggregates, decomposition, effects, stationarity
from app.analysis.decomposition import Finding, date_bounds
from app.storage.catalog import Table
from app.storage.database import create_db_engine, session_factory, transaction
from app.storage.repositories import AnalysisRepository
from app.storage.warehouse import open_warehouse

logger = logging.getLogger(__name__)


def run_analysis() -> dict[str, list[Finding]]:
    storage = config.storage_settings()
    cfg = config.analysis_settings()

    with open_warehouse(storage) as wh:
        national = aggregates.national_daily(wh)
        last_day = date_bounds(national)[1]
        families = aggregates.family_daily(
            wh,
            last_day - timedelta(days=cfg.analysis_family_window_days),
        )
        holidays = wh.frame("SELECT * FROM holidays")
        oil = wh.frame("SELECT date, price FROM oil")
        transactions = aggregates.store_daily_transactions(wh)
        weather = wh.frame("SELECT * FROM weather") if wh.has(Table.WEATHER) else None
        promo_panel = aggregates.dense_promo_panel(
            wh,
            last_day - timedelta(days=cfg.analysis_promo_window_days),
            last_day,
            min_coverage=cfg.analysis_promo_min_coverage,
            series_per_family=cfg.analysis_promo_series_per_family,
            seed=cfg.analysis_seed,
        )

    log_sales = np.log1p(decomposition.complete_daily(national, "unit_sales"))
    jobs: dict[str, Callable[[], list[Finding]]] = {
        "seasonality": lambda: decomposition.national_seasonality(national),
        "family_seasonality": lambda: decomposition.family_weekly_strength(families),
        "stationarity": lambda: stationarity.stationarity_tests(log_sales, cfg.analysis_alpha),
        "calendar_effects": lambda: effects.calendar_effects(
            national, holidays, cfg.analysis_hac_lags
        ),
        "promotion_lift": lambda: effects.promotion_lift(
            promo_panel, cfg.analysis_fe_iterations, cfg.analysis_promo_min_series
        ),
        "oil_granger": lambda: effects.oil_granger(national, oil, cfg.analysis_granger_max_lag),
    }
    if weather is not None:
        jobs["rain_effect"] = lambda: effects.rain_effect(
            transactions, weather, cfg.analysis_heavy_rain_mm, cfg.analysis_fe_iterations
        )

    results: dict[str, list[Finding]] = {}
    for name, job in jobs.items():
        results[name] = job()
        logger.info("analysis complete", extra={"analysis": name, "findings": len(results[name])})

    (storage.curated_dir / "_analysis.json").write_text(
        json.dumps({k: dict(v) for k, v in results.items()}, indent=2), encoding="utf-8"
    )
    factory = session_factory(create_db_engine(config.database_settings()))
    with transaction(factory) as session:
        repo = AnalysisRepository(session)
        for name, findings in results.items():
            repo.save(name, findings)
    return results
