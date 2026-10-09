"""Daily crude oil spot price from the FRED API (Ecuador's economy tracks oil)."""

import logging
from datetime import date

import polars as pl

from app.config import FredSettings, HttpSettings, StorageSettings
from app.errors import DataValidationError
from app.ingestion.http_client import ApiClient
from app.storage.catalog import Table, record_table, table_path

logger = logging.getLogger(__name__)


class FredOilSource:
    def __init__(self, fred: FredSettings, http: HttpSettings, storage: StorageSettings) -> None:
        self._fred = fred
        self._http = http
        self._storage = storage

    def run(self, start: date, end: date) -> None:
        with ApiClient(str(self._fred.fred_base_url).rstrip("/") + "/", self._http) as client:
            payload = client.get_json(
                "series/observations",
                {
                    "series_id": self._fred.fred_oil_series_id,
                    "api_key": self._fred.fred_api_key.get_secret_value(),
                    "file_type": "json",
                    "observation_start": start.isoformat(),
                    "observation_end": end.isoformat(),
                },
            )
        observations = payload.get("observations", [])
        if not observations:
            raise DataValidationError("FRED returned no oil observations")
        # FRED marks market holidays with "." instead of a number.
        frame = pl.DataFrame(
            {
                "date": [obs["date"] for obs in observations],
                "price": [
                    None if obs["value"] == "." else float(obs["value"]) for obs in observations
                ],
            }
        ).with_columns(pl.col("date").str.to_date())
        frame.write_parquet(table_path(self._storage, Table.OIL), compression="zstd")
        record_table(self._storage, Table.OIL, rows=frame.height, source="fred")
        logger.info("wrote table", extra={"table": Table.OIL, "rows": frame.height})
