"""Read-side queries for the API (the Repository the routes depend on).

Every release-scoped query filters on the release id the request resolved, so
one request always reads one consistent release.
"""

import uuid
from datetime import date, timedelta
from typing import Any

from sqlalchemy import Select, and_, func, select
from sqlalchemy.orm import Session

from app.storage.models import (
    AnalysisResult,
    BacktestScore,
    Base,
    FamilyForecast,
    FamilySalesHistory,
    Item,
    ItemForecast,
    ModelRun,
    OrderPlan,
    PolicyEvaluation,
    QualityCheckRecord,
    Release,
    SalesHistory,
    Store,
)


def _column_names(model: type[Base]) -> list[str]:
    return [attr.key for attr in model.__mapper__.column_attrs]


class ReadQueries:
    def __init__(self, session: Session) -> None:
        self._s = session

    # ------------------------------------------------------------ releases

    def current_release(self) -> Release | None:
        return self._s.scalars(select(Release).where(Release.is_current)).first()

    def ping(self) -> bool:
        return self._s.execute(select(1)).scalar() == 1

    # ------------------------------------------------------------- catalog

    def stores(self, release_id: uuid.UUID) -> list[dict[str, Any]]:
        totals = (
            select(ItemForecast.store_nbr, func.sum(ItemForecast.p50).label("forecast_units"))
            .where(ItemForecast.release_id == release_id)
            .group_by(ItemForecast.store_nbr)
            .subquery()
        )
        rows = self._s.execute(
            select(Store, totals.c.forecast_units)
            .outerjoin(totals, totals.c.store_nbr == Store.store_nbr)
            .order_by(Store.store_nbr)
        )
        return [
            {
                **{c: getattr(store, c) for c in _column_names(Store)},
                "forecast_units": units,
            }
            for store, units in rows
        ]

    def families(self) -> list[str]:
        return list(self._s.scalars(select(Item.family).distinct().order_by(Item.family)))

    def series_page(
        self,
        release_id: uuid.UUID,
        *,
        store_nbr: int | None,
        family: str | None,
        item_nbr: int | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        forecast = (
            select(
                ItemForecast.store_nbr,
                ItemForecast.item_nbr,
                func.sum(ItemForecast.p50).label("forecast_units"),
            )
            .where(ItemForecast.release_id == release_id)
            .group_by(ItemForecast.store_nbr, ItemForecast.item_nbr)
            .subquery()
        )
        recent = (
            select(
                SalesHistory.store_nbr,
                SalesHistory.item_nbr,
                func.sum(SalesHistory.unit_sales).label("recent_units"),
            )
            .where(SalesHistory.release_id == release_id)
            .group_by(SalesHistory.store_nbr, SalesHistory.item_nbr)
            .subquery()
        )
        query: Select[Any] = (
            select(
                forecast.c.store_nbr,
                forecast.c.item_nbr,
                Item.family,
                Item.perishable,
                forecast.c.forecast_units,
                func.coalesce(recent.c.recent_units, 0).label("recent_units"),
            )
            .join(Item, Item.item_nbr == forecast.c.item_nbr)
            .outerjoin(
                recent,
                and_(
                    recent.c.store_nbr == forecast.c.store_nbr,
                    recent.c.item_nbr == forecast.c.item_nbr,
                ),
            )
        )
        if store_nbr is not None:
            query = query.where(forecast.c.store_nbr == store_nbr)
        if family:
            query = query.where(Item.family == family)
        if item_nbr is not None:
            query = query.where(forecast.c.item_nbr == item_nbr)
        total = self._s.execute(select(func.count()).select_from(query.subquery())).scalar_one()
        rows = self._s.execute(
            query.order_by(
                forecast.c.forecast_units.desc(), forecast.c.store_nbr, forecast.c.item_nbr
            )
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return [dict(r._mapping) for r in rows], total

    # ------------------------------------------------------------ forecasts

    def series_detail(
        self, release_id: uuid.UUID, store_nbr: int, item_nbr: int
    ) -> dict[str, Any] | None:
        item = self._s.get(Item, item_nbr)
        if item is None:
            return None
        forecast = self._s.execute(
            select(
                ItemForecast.target_date.label("day"),
                ItemForecast.p10,
                ItemForecast.p50,
                ItemForecast.p90,
                ItemForecast.onpromotion,
            )
            .where(
                ItemForecast.release_id == release_id,
                ItemForecast.store_nbr == store_nbr,
                ItemForecast.item_nbr == item_nbr,
            )
            .order_by(ItemForecast.target_date)
        ).all()
        if not forecast:
            return None
        history = self._s.execute(
            select(
                SalesHistory.sale_date.label("day"),
                SalesHistory.unit_sales.label("units"),
                SalesHistory.onpromotion,
            )
            .where(
                SalesHistory.release_id == release_id,
                SalesHistory.store_nbr == store_nbr,
                SalesHistory.item_nbr == item_nbr,
            )
            .order_by(SalesHistory.sale_date)
        ).all()
        order = self._s.get(OrderPlan, (release_id, store_nbr, item_nbr))
        return {
            "store_nbr": store_nbr,
            "item_nbr": item_nbr,
            "family": item.family,
            "item_class": item.item_class,
            "perishable": item.perishable,
            "history": [dict(r._mapping) for r in history],
            "forecast": [dict(r._mapping) for r in forecast],
            "order": order,
        }

    def hierarchy_detail(
        self, release_id: uuid.UUID, store_nbr: int, family: str, models: list[str] | None
    ) -> dict[str, Any] | None:
        query = select(FamilyForecast).where(
            FamilyForecast.release_id == release_id,
            FamilyForecast.store_nbr == store_nbr,
            FamilyForecast.family == family,
        )
        if models:
            query = query.where(FamilyForecast.model_name.in_(models))
        rows = self._s.scalars(
            query.order_by(FamilyForecast.model_name, FamilyForecast.target_date)
        ).all()
        if not rows:
            return None
        by_model: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_model.setdefault(r.model_name, []).append(
                {"day": r.target_date, "p10": r.p10, "p50": r.p50, "p90": r.p90}
            )
        history = self._s.execute(
            select(
                FamilySalesHistory.sale_date.label("day"),
                FamilySalesHistory.unit_sales.label("units"),
            )
            .where(
                FamilySalesHistory.release_id == release_id,
                FamilySalesHistory.store_nbr == store_nbr,
                FamilySalesHistory.family == family,
            )
            .order_by(FamilySalesHistory.sale_date)
        ).all()
        return {
            "store_nbr": store_nbr,
            "family": family,
            "history": [dict(r._mapping) for r in history],
            "models": [{"model_name": m, "points": p} for m, p in by_model.items()],
        }

    def national_totals(
        self, release_id: uuid.UUID, origin: date, days: int
    ) -> tuple[float, float]:
        forecast = self._s.execute(
            select(func.coalesce(func.sum(ItemForecast.p50), 0)).where(
                ItemForecast.release_id == release_id,
                ItemForecast.target_date <= origin + timedelta(days=days),
            )
        ).scalar_one()
        actual = self._s.execute(
            select(func.coalesce(func.sum(SalesHistory.unit_sales), 0)).where(
                SalesHistory.release_id == release_id,
                SalesHistory.sale_date > origin - timedelta(days=days),
            )
        ).scalar_one()
        return float(forecast), float(actual)

    def series_count(self, release_id: uuid.UUID) -> int:
        return self._s.execute(
            select(func.count()).select_from(
                select(ItemForecast.store_nbr, ItemForecast.item_nbr)
                .where(ItemForecast.release_id == release_id)
                .distinct()
                .subquery()
            )
        ).scalar_one()

    # ------------------------------------------------------------- planning

    def orders_page(
        self,
        release_id: uuid.UUID,
        *,
        store_nbr: int | None,
        family: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        query = (
            select(OrderPlan, Item.family)
            .join(Item, Item.item_nbr == OrderPlan.item_nbr)
            .where(OrderPlan.release_id == release_id)
        )
        if store_nbr is not None:
            query = query.where(OrderPlan.store_nbr == store_nbr)
        if family:
            query = query.where(Item.family == family)
        total = self._s.execute(select(func.count()).select_from(query.subquery())).scalar_one()
        rows = self._s.execute(
            query.order_by(OrderPlan.order_quantity.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return [
            {**{c: getattr(plan, c) for c in _column_names(OrderPlan)}, "family": fam}
            for plan, fam in rows
        ], total

    def policies(self, release_id: uuid.UUID, segment_prefix: str | None) -> list[PolicyEvaluation]:
        query = select(PolicyEvaluation).where(PolicyEvaluation.release_id == release_id)
        if segment_prefix:
            query = query.where(PolicyEvaluation.segment.startswith(segment_prefix))
        return list(
            self._s.scalars(query.order_by(PolicyEvaluation.segment, PolicyEvaluation.policy))
        )

    # ------------------------------------------------------- models & data

    def model_runs(self, level: str | None) -> list[ModelRun]:
        latest = (
            select(
                ModelRun.model_name,
                ModelRun.level,
                func.max(ModelRun.created_at).label("created_at"),
            )
            .group_by(ModelRun.model_name, ModelRun.level)
            .subquery()
        )
        query = select(ModelRun).join(
            latest,
            and_(
                latest.c.model_name == ModelRun.model_name,
                latest.c.level == ModelRun.level,
                latest.c.created_at == ModelRun.created_at,
            ),
        )
        if level:
            query = query.where(ModelRun.level == level)
        return list(self._s.scalars(query.order_by(ModelRun.level, ModelRun.model_name)))

    def run_scores(self, run_id: uuid.UUID) -> list[BacktestScore]:
        return list(
            self._s.scalars(
                select(BacktestScore)
                .where(BacktestScore.run_id == run_id)
                .order_by(BacktestScore.cutoff, BacktestScore.metric, BacktestScore.horizon)
            )
        )

    def analyses(self) -> list[AnalysisResult]:
        return list(
            self._s.scalars(
                select(AnalysisResult).order_by(AnalysisResult.analysis, AnalysisResult.subject)
            )
        )

    def latest_quality(self) -> list[QualityCheckRecord]:
        newest = select(func.max(QualityCheckRecord.checked_at)).scalar_subquery()
        return list(
            self._s.scalars(
                select(QualityCheckRecord)
                .where(QualityCheckRecord.checked_at == newest)
                .order_by(QualityCheckRecord.severity, QualityCheckRecord.check_name)
            )
        )
