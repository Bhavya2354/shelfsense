"""Relational schema for everything the API serves.

The pipeline writes a complete, immutable *release* (forecasts, order plans,
scores) and then flips one row to make it current, so readers never see a
half-written release.
"""

import uuid
from datetime import date, datetime
from typing import Any, ClassVar

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONB}


class Store(Base):
    __tablename__ = "stores"

    store_nbr: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    city: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(64))
    store_type: Mapped[str] = mapped_column(String(4))
    cluster: Mapped[int] = mapped_column(SmallInteger)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)


class Item(Base):
    __tablename__ = "items"

    item_nbr: Mapped[int] = mapped_column(Integer, primary_key=True)
    family: Mapped[str] = mapped_column(String(64), index=True)
    item_class: Mapped[int] = mapped_column(SmallInteger)
    perishable: Mapped[bool] = mapped_column(Boolean)


class ModelRun(Base):
    """One trained model evaluated by rolling-origin backtest."""

    __tablename__ = "model_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(64))
    level: Mapped[str] = mapped_column(String(16))
    params: Mapped[dict[str, Any]] = mapped_column(default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    train_start: Mapped[date]
    train_end: Mapped[date]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BacktestScore(Base):
    __tablename__ = "backtest_scores"

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), primary_key=True
    )
    cutoff: Mapped[date] = mapped_column(primary_key=True)
    horizon: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    """Days ahead of the cutoff; 0 means the score over the whole horizon."""
    metric: Mapped[str] = mapped_column(String(32), primary_key=True)
    value: Mapped[float]


class Release(Base):
    """A consistent snapshot of served data; exactly one is current."""

    __tablename__ = "releases"

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    forecast_origin: Mapped[date]
    """Last day of observed sales; forecasts start the day after."""
    item_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_runs.id"))
    notes: Mapped[str | None] = mapped_column(Text)
    is_current: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(
            "uq_releases_current", "is_current", unique=True, postgresql_where=text("is_current")
        ),
    )


class ItemForecast(Base):
    __tablename__ = "item_forecasts"

    release_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("releases.id", ondelete="CASCADE"), primary_key=True
    )
    store_nbr: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    item_nbr: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_date: Mapped[date] = mapped_column(primary_key=True)
    p10: Mapped[float]
    p50: Mapped[float]
    p90: Mapped[float]
    onpromotion: Mapped[bool]


class FamilyForecast(Base):
    """Store x family forecasts from every family-level model, for side-by-side comparison."""

    __tablename__ = "family_forecasts"

    release_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("releases.id", ondelete="CASCADE"), primary_key=True
    )
    store_nbr: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    family: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    target_date: Mapped[date] = mapped_column(primary_key=True)
    p10: Mapped[float | None]
    p50: Mapped[float]
    p90: Mapped[float | None]


class SalesHistory(Base):
    """Recent daily sales for the store-items a release serves."""

    __tablename__ = "sales_history"

    release_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("releases.id", ondelete="CASCADE"), primary_key=True
    )
    store_nbr: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    item_nbr: Mapped[int] = mapped_column(Integer, primary_key=True)
    sale_date: Mapped[date] = mapped_column(Date, primary_key=True)
    unit_sales: Mapped[float]
    onpromotion: Mapped[bool]


class FamilySalesHistory(Base):
    __tablename__ = "family_sales_history"

    release_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("releases.id", ondelete="CASCADE"), primary_key=True
    )
    store_nbr: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    family: Mapped[str] = mapped_column(String(64), primary_key=True)
    sale_date: Mapped[date] = mapped_column(Date, primary_key=True)
    unit_sales: Mapped[float]


class OrderPlan(Base):
    """Recommended order for one store-item for the first planning day."""

    __tablename__ = "order_plans"

    release_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("releases.id", ondelete="CASCADE"), primary_key=True
    )
    store_nbr: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    item_nbr: Mapped[int] = mapped_column(Integer, primary_key=True)
    cover_days: Mapped[int] = mapped_column(SmallInteger)
    service_level: Mapped[float]
    expected_demand: Mapped[float]
    order_quantity: Mapped[float]
    safety_stock: Mapped[float]
    expected_lost_units: Mapped[float]
    expected_leftover_units: Mapped[float]


class PolicyEvaluation(Base):
    """Backtested outcome of a replenishment policy against actual demand."""

    __tablename__ = "policy_evaluations"

    release_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("releases.id", ondelete="CASCADE"), primary_key=True
    )
    policy: Mapped[str] = mapped_column(String(32), primary_key=True)
    segment: Mapped[str] = mapped_column(String(64), primary_key=True)
    demand_units: Mapped[float]
    lost_units: Mapped[float]
    leftover_units: Mapped[float]
    fill_rate: Mapped[float]
    total_cost: Mapped[float]


class AnalysisResult(Base):
    """Statistical findings (decomposition, tests, effect sizes) keyed by analysis and subject."""

    __tablename__ = "analysis_results"

    analysis: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject: Mapped[str] = mapped_column(String(128), primary_key=True)
    payload: Mapped[dict[str, Any]]
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class QualityCheckRecord(Base):
    __tablename__ = "quality_checks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    check_name: Mapped[str] = mapped_column(String(64))
    table_name: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(16))
    violations: Mapped[int] = mapped_column(BigInteger)
