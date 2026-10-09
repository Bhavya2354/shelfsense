"""Ingestion pipeline: competition data first, then signals keyed off its date range."""

import logging
from enum import StrEnum

from app import config
from app.ingestion.favorita import FavoritaSource
from app.ingestion.oil import FredOilSource
from app.ingestion.quality import CheckResult, run_quality_gate
from app.ingestion.weather import OpenMeteoWeatherSource
from app.storage.warehouse import open_warehouse

logger = logging.getLogger(__name__)


class Source(StrEnum):
    FAVORITA = "favorita"
    OIL = "oil"
    WEATHER = "weather"


def run_ingestion(sources: set[Source], *, force: bool = False) -> list[CheckResult]:
    storage = config.storage_settings()
    http = config.http_settings()

    if Source.FAVORITA in sources:
        FavoritaSource(config.kaggle_settings(), storage).run(force=force)

    with open_warehouse(storage) as wh:
        start, end = wh.frame(
            "SELECT min(date) AS s, (SELECT max(date) FROM test) AS e FROM sales"
        ).row(0)
        cities = wh.frame("SELECT DISTINCT city, state FROM stores ORDER BY city")

    if Source.OIL in sources:
        FredOilSource(config.fred_settings(), http, storage).run(start, end)
    if Source.WEATHER in sources:
        OpenMeteoWeatherSource(config.open_meteo_settings(), http, storage).run(cities, start, end)

    with open_warehouse(storage) as wh:
        return run_quality_gate(wh, storage.curated_dir / "_quality.json")
