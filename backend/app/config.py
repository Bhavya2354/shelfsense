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

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"


class HttpSettings(_Section):
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    http_max_retries: int = Field(default=5, ge=0)
    http_backoff_seconds: float = Field(default=1.0, gt=0)
    download_min_bytes_per_second: int = Field(default=150_000, ge=1)
    download_stall_window_seconds: float = Field(default=30.0, gt=0)
    download_max_reconnects: int = Field(default=200, ge=1)


class FavoritaSettings(_Section):
    favorita_archive_url: HttpUrl = Field(
        description="Zip of the Favorita competition CSVs, each zipped individually."
    )


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


class ForecastSettings(_Section):
    forecast_horizon: int = Field(default=16, ge=1)
    item_history_days: int = Field(default=280, ge=120)
    item_train_windows: int = Field(default=8, ge=1)
    window_stride_days: int = Field(default=7, ge=1)
    backtest_folds: int = Field(default=2, ge=1)
    perishable_weight: float = Field(default=1.25, gt=0)
    interval_quantiles: tuple[float, float] = (0.1, 0.9)
    velocity_buckets: int = Field(default=5, ge=1)
    lgbm_learning_rate: float = Field(default=0.05, gt=0)
    lgbm_num_leaves: int = Field(default=127, ge=2)
    lgbm_min_data_in_leaf: int = Field(default=200, ge=1)
    lgbm_feature_fraction: float = Field(default=0.8, gt=0, le=1)
    lgbm_bagging_fraction: float = Field(default=0.8, gt=0, le=1)
    lgbm_max_rounds: int = Field(default=1500, ge=10)
    lgbm_early_stopping_rounds: int = Field(default=75, ge=5)
    mlp_hidden_sizes: tuple[int, ...] = (512, 256, 128)
    mlp_dropout: float = Field(default=0.2, ge=0, lt=1)
    mlp_epochs: int = Field(default=12, ge=1)
    mlp_batch_size: int = Field(default=4096, ge=64)
    mlp_learning_rate: float = Field(default=2e-3, gt=0)
    family_history_days: int = Field(default=730, ge=120)
    family_input_size: int = Field(default=112, ge=14)
    family_max_steps: int = Field(default=800, ge=10)
    item_models: tuple[str, ...] = (
        "moving_average",
        "weekday_average",
        "lightgbm",
        "embedding_mlp",
    )
    lgbm_lambda_l2: float = Field(default=0.0, ge=0)
    tuning_lightgbm_trials: int = Field(default=20, ge=0)
    tuning_mlp_trials: int = Field(default=8, ge=0)
    tuning_max_rounds: int = Field(default=500, ge=10)
    tuning_mlp_epochs: int = Field(default=4, ge=1)
    tuning_row_fraction: float = Field(default=0.5, gt=0, le=1)
    chronos_model_id: str = Field(description="Hugging Face id of the Chronos-2 checkpoint.")
    chronos_context_days: int = Field(default=365, ge=28)
    chronos_finetune_steps: int = Field(default=300, ge=1)
    chronos_finetune_learning_rate: float = Field(default=1e-5, gt=0)
    chronos_batch_size: int = Field(default=64, ge=1)
    bootstrap_resamples: int = Field(default=300, ge=50)
    random_seed: int = 7
    n_jobs: int = Field(default=-1)


class InventorySettings(_Section):
    """Costs are fractions of an item's shelf price (Favorita publishes no prices)."""

    underage_cost: float = Field(default=0.30, gt=0, description="Margin lost per unit short.")
    overage_cost_perishable: float = Field(default=0.50, gt=0, description="Write-off per unit.")
    overage_cost_shelf_stable: float = Field(default=0.03, gt=0, description="Holding per cycle.")
    cover_days_perishable: int = Field(default=1, ge=1)
    cover_days_shelf_stable: int = Field(default=7, ge=1)
    residual_samples_per_bucket: int = Field(default=400, ge=50)


class PublishSettings(_Section):
    publish_top_series: int = Field(default=25_000, ge=100)
    publish_history_days: int = Field(default=56, ge=7)
    publish_family_history_days: int = Field(default=120, ge=14)
    publish_keep_releases: int = Field(default=2, ge=1)


class ApiSettings(_Section):
    """The API connects as a read-only role; it never needs write access."""

    api_database_url: SecretStr
    api_cors_origins: list[str] = Field(description="Exact origins allowed to call the API.")
    api_pool_size: int = Field(default=5, ge=1)
    api_cache_ttl_seconds: int = Field(default=300, ge=0)
    api_cache_max_entries: int = Field(default=2_000, ge=10)
    api_default_page_size: int = Field(default=50, ge=1)
    api_max_page_size: int = Field(default=200, ge=1)
    api_statement_timeout_ms: int = Field(default=5_000, ge=100)


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
def favorita_settings() -> FavoritaSettings:
    return _load(FavoritaSettings)


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


@lru_cache
def forecast_settings() -> ForecastSettings:
    return _load(ForecastSettings)


@lru_cache
def inventory_settings() -> InventorySettings:
    return _load(InventorySettings)


@lru_cache
def publish_settings() -> PublishSettings:
    return _load(PublishSettings)


@lru_cache
def api_settings() -> ApiSettings:
    return _load(ApiSettings)
