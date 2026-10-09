"""Daily historical weather for every store city from the Open-Meteo archive."""

import logging
from datetime import date

import polars as pl

from app.config import HttpSettings, OpenMeteoSettings, StorageSettings
from app.errors import DataValidationError
from app.ingestion.http_client import ApiClient
from app.storage.catalog import Table, record_table, table_path

logger = logging.getLogger(__name__)


class OpenMeteoWeatherSource:
    """Geocodes each store city, then pulls one daily archive series per city."""

    def __init__(
        self, meteo: OpenMeteoSettings, http: HttpSettings, storage: StorageSettings
    ) -> None:
        self._meteo = meteo
        self._http = http
        self._storage = storage

    def run(self, cities: pl.DataFrame, start: date, end: date) -> None:
        """`cities` holds one row per (city, state) taken from the stores table."""
        locations = self._geocode(cities)
        locations.write_parquet(table_path(self._storage, Table.STORE_LOCATIONS))
        record_table(
            self._storage, Table.STORE_LOCATIONS, rows=locations.height, source="open-meteo"
        )

        frames = []
        with ApiClient(str(self._meteo.open_meteo_archive_url), self._http) as client:
            for loc in locations.iter_rows(named=True):
                payload = client.get_json(
                    "",
                    {
                        "latitude": loc["latitude"],
                        "longitude": loc["longitude"],
                        "start_date": start.isoformat(),
                        "end_date": end.isoformat(),
                        "daily": ",".join(self._meteo.weather_daily_variables),
                        "timezone": loc["timezone"],
                    },
                )
                daily = pl.DataFrame(payload["daily"]).rename({"time": "date"})
                frames.append(
                    daily.with_columns(
                        pl.col("date").str.to_date(), pl.lit(loc["city"]).alias("city")
                    )
                )
        weather = pl.concat(frames).select("city", "date", *self._meteo.weather_daily_variables)
        weather.write_parquet(table_path(self._storage, Table.WEATHER), compression="zstd")
        record_table(self._storage, Table.WEATHER, rows=weather.height, source="open-meteo")
        logger.info("wrote table", extra={"table": Table.WEATHER, "rows": weather.height})

    def _geocode(self, cities: pl.DataFrame) -> pl.DataFrame:
        rows = []
        with ApiClient(str(self._meteo.open_meteo_geocoding_url), self._http) as client:
            for city, state in cities.select("city", "state").iter_rows():
                payload = client.get_json(
                    "",
                    {
                        "name": city,
                        "countryCode": self._meteo.weather_country_code,
                        "count": 10,
                        "language": "es",
                    },
                )
                rows.append(self._pick(city, state, payload.get("results", [])))
        return pl.DataFrame(rows)

    @staticmethod
    def _pick(city: str, state: str, results: list[dict[str, object]]) -> dict[str, object]:
        """Prefer a match in the store's own province; the same name can exist elsewhere."""
        if not results:
            raise DataValidationError(f"no geocoding result for {city}")

        def normalise(text: object) -> str:
            return str(text).lower().replace("santo domingo de los tsachilas", "santo domingo")

        in_state = [r for r in results if normalise(state) in normalise(r.get("admin1", ""))]
        best = (in_state or results)[0]
        return {
            "city": city,
            "state": state,
            "latitude": best["latitude"],
            "longitude": best["longitude"],
            "elevation": best.get("elevation"),
            "timezone": best.get("timezone"),
            "matched_province": bool(in_state),
        }
