"""Typed settings, read from the environment (and `.env` in local development).

Each component loads only the section it needs, so a process fails fast on the
settings it actually depends on. Environment-specific values (paths, URLs,
credentials) have no defaults; only operational tunables do.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.errors import ConfigurationError


class _Section(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )


class RuntimeSettings(_Section):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"


class StorageSettings(_Section):
    data_dir: Path = Field(description="Root folder for raw, curated and model data.")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / "curated"


class HttpSettings(_Section):
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    http_max_retries: int = Field(default=5, ge=0)
    http_backoff_seconds: float = Field(default=1.0, gt=0)


class KaggleSettings(_Section):
    kaggle_competition: str = Field(description="Competition slug to download.")


class FredSettings(_Section):
    fred_api_key: SecretStr
    fred_base_url: HttpUrl
    fred_oil_series_id: str


class OpenMeteoSettings(_Section):
    open_meteo_archive_url: HttpUrl
    open_meteo_geocoding_url: HttpUrl
    weather_country_code: str = Field(min_length=2, max_length=2)
    weather_daily_variables: tuple[str, ...] = (
        "temperature_2m_mean",
        "temperature_2m_max",
        "precipitation_sum",
        "precipitation_hours",
    )


class AnalysisSettings(_Section):
    analysis_alpha: float = Field(default=0.05, gt=0, lt=1)
    analysis_hac_lags: int = Field(default=14, ge=1)
    analysis_fe_iterations: int = Field(default=20, ge=1)
    analysis_family_window_days: int = Field(default=730, ge=60)
    analysis_promo_window_days: int = Field(default=182, ge=28)
    analysis_promo_min_coverage: float = Field(default=0.9, gt=0, le=1)
    analysis_promo_series_per_family: int = Field(default=500, ge=10)
    analysis_promo_min_series: int = Field(default=20, ge=2)
    analysis_heavy_rain_mm: float = Field(default=10.0, gt=0)
    analysis_granger_max_lag: int = Field(default=4, ge=1)
    analysis_seed: int = 7


class DatabaseSettings(_Section):
    """Read-write connection used by the pipeline and migrations."""

    database_url: SecretStr
    database_pool_size: int = Field(default=5, ge=1)
    database_statement_timeout_ms: int = Field(default=300_000, ge=1_000)


def _load[S: _Section](section: type[S]) -> S:
    try:
        return section()
    except ValidationError as exc:
        fields = sorted({str(err["loc"][0]).upper() for err in exc.errors()})
        message = f"{section.__name__}: missing or invalid settings: {', '.join(fields)}"
        raise ConfigurationError(message) from None


@lru_cache
def runtime_settings() -> RuntimeSettings:
    return _load(RuntimeSettings)


@lru_cache
def storage_settings() -> StorageSettings:
    return _load(StorageSettings)


@lru_cache
def http_settings() -> HttpSettings:
    return _load(HttpSettings)


@lru_cache
def kaggle_settings() -> KaggleSettings:
    return _load(KaggleSettings)


@lru_cache
def fred_settings() -> FredSettings:
    return _load(FredSettings)


@lru_cache
def open_meteo_settings() -> OpenMeteoSettings:
    return _load(OpenMeteoSettings)


@lru_cache
def database_settings() -> DatabaseSettings:
    return _load(DatabaseSettings)


@lru_cache
def analysis_settings() -> AnalysisSettings:
    return _load(AnalysisSettings)
