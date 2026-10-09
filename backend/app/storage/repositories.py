"""Write-side repositories used by the pipeline. The API has its own read-only queries."""

import math
import uuid
from collections.abc import Iterable
from datetime import date
from typing import Any, cast

import polars as pl
from sqlalchemy import Table, delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.storage.database import copy_frame
from app.storage.models import (
    AnalysisResult,
    BacktestScore,
    Base,
    Item,
    ModelRun,
    QualityCheckRecord,
    Release,
    Store,
)


def _upsert(session: Session, model: type[Base], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table = cast(Table, model.__table__)
    stmt = insert(table).values(rows)
    keys = [c.name for c in table.primary_key.columns]
    updates = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in keys}
    session.execute(stmt.on_conflict_do_update(index_elements=keys, set_=updates))


class ReferenceDataRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def sync_stores(self, stores: pl.DataFrame) -> None:
        _upsert(self._session, Store, stores.to_dicts())

    def sync_items(self, items: pl.DataFrame) -> None:
        for chunk in items.iter_slices(5_000):
            _upsert(self._session, Item, chunk.to_dicts())


class ModelRunRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record(
        self,
        *,
        model_name: str,
        level: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        train_start: date,
        train_end: date,
        scores: pl.DataFrame,
    ) -> uuid.UUID:
        """Store a run with its scores (columns: cutoff, horizon, metric, value)."""
        run = ModelRun(
            model_name=model_name,
            level=level,
            params=params,
            # JSON has no Infinity/NaN; a non-finite score is stored as null.
            metrics={k: v if math.isfinite(v) else None for k, v in metrics.items()},
            train_start=train_start,
            train_end=train_end,
        )
        self._session.add(run)
        self._session.flush()
        copy_frame(
            self._session,
            BacktestScore.__tablename__,
            scores.select("cutoff", "horizon", "metric", "value").with_columns(
                pl.lit(str(run.id)).alias("run_id")
            ),
        )
        return run.id


class ReleaseRepository:
    """Builds a release in full, then makes it current in a single statement pair."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self, *, forecast_origin: date, item_run_id: uuid.UUID | None, notes: str
    ) -> uuid.UUID:
        release = Release(forecast_origin=forecast_origin, item_run_id=item_run_id, notes=notes)
        self._session.add(release)
        self._session.flush()
        return release.id

    def load(self, release_id: uuid.UUID, table: str, frame: pl.DataFrame) -> int:
        return copy_frame(
            self._session, table, frame.with_columns(pl.lit(str(release_id)).alias("release_id"))
        )

    def publish(self, release_id: uuid.UUID) -> None:
        self._session.execute(update(Release).where(Release.is_current).values(is_current=False))
        self._session.execute(
            update(Release).where(Release.id == release_id).values(is_current=True)
        )

    def prune(self, keep: int) -> int:
        """Delete all but the newest `keep` releases; child rows cascade."""
        stale = (
            select(Release.id)
            .where(~Release.is_current)
            .order_by(Release.created_at.desc())
            .offset(max(keep - 1, 0))
        )
        result = self._session.execute(delete(Release).where(Release.id.in_(stale)))
        return int(result.rowcount)  # type: ignore[attr-defined]


class AnalysisRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, analysis: str, results: Iterable[tuple[str, dict[str, Any]]]) -> None:
        rows = [{"analysis": analysis, "subject": s, "payload": p} for s, p in results]
        self._session.execute(delete(AnalysisResult).where(AnalysisResult.analysis == analysis))
        if rows:
            self._session.execute(insert(AnalysisResult).values(rows))


class QualityRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record(self, results: Iterable[dict[str, Any]]) -> None:
        self._session.add_all(QualityCheckRecord(**r) for r in results)
