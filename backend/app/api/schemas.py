"""Response models: the API's public contract."""

import uuid
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class Schema(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)


class Health(Schema):
    status: str
    database: str
    release_id: uuid.UUID | None
    forecast_origin: date | None


class ReleaseInfo(Schema):
    id: uuid.UUID
    forecast_origin: date
    created_at: datetime
    notes: str | None


class Store(Schema):
    store_nbr: int
    city: str
    state: str
    store_type: str
    cluster: int
    latitude: float | None
    longitude: float | None
    forecast_units: float | None = None


class SeriesSummary(Schema):
    store_nbr: int
    item_nbr: int
    family: str
    perishable: bool
    forecast_units: float
    recent_units: float


class Page[T](Schema):
    items: list[T]
    total: int
    page: int
    page_size: int


class DailyPoint(Schema):
    day: date
    units: float
    onpromotion: bool | None = None


class ForecastPoint(Schema):
    day: date
    p10: float | None
    p50: float
    p90: float | None
    onpromotion: bool | None = None


class OrderLine(Schema):
    store_nbr: int
    item_nbr: int
    family: str | None = None
    cover_days: int
    service_level: float
    expected_demand: float
    order_quantity: float
    safety_stock: float
    expected_lost_units: float
    expected_leftover_units: float


class SeriesDetail(Schema):
    store_nbr: int
    item_nbr: int
    family: str
    item_class: int
    perishable: bool
    history: list[DailyPoint]
    forecast: list[ForecastPoint]
    order: OrderLine | None


class ModelForecast(Schema):
    model_name: str
    points: list[ForecastPoint]


class HierarchyDetail(Schema):
    store_nbr: int
    family: str
    history: list[DailyPoint]
    models: list[ModelForecast]


class PolicyResult(Schema):
    policy: str
    segment: str
    demand_units: float
    lost_units: float
    leftover_units: float
    fill_rate: float
    total_cost: float


class ModelRun(Schema):
    id: uuid.UUID
    model_name: str
    level: str
    metrics: dict[str, Any]
    created_at: datetime


class Score(Schema):
    cutoff: date
    horizon: int
    metric: str
    value: float


class Finding(Schema):
    subject: str
    payload: dict[str, Any]


class Analysis(Schema):
    analysis: str
    findings: list[Finding]


class QualityCheck(Schema):
    check_name: str
    table_name: str
    severity: str
    violations: int
    checked_at: datetime


class Overview(Schema):
    release: ReleaseInfo
    series: int
    stores: int
    forecast_units_next_7d: float
    actual_units_last_7d: float
    best_item_model: str | None
    item_scores: dict[str, dict[str, float]]
    policy_totals: list[PolicyResult]
