import uuid
from itertools import groupby

from fastapi import APIRouter, Request, Response

from app.api import schemas
from app.api.dependencies import CurrentRelease, Queries, Settings, release_response

router = APIRouter(tags=["insights"])

_OVERVIEW_DAYS = 7
_SUMMARY_SEGMENTS = ("all", "perishable", "shelf_stable")


@router.get("/overview", response_model=schemas.Overview)
def overview(
    request: Request, release: CurrentRelease, queries: Queries, settings: Settings
) -> Response:
    def compute() -> schemas.Overview:
        forecast_7d, actual_7d = queries.national_totals(
            release.id, release.forecast_origin, _OVERVIEW_DAYS
        )
        runs = queries.model_runs("item")
        scores = {r.model_name: {k: float(v) for k, v in r.metrics.items()} for r in runs}
        best = min(scores, key=lambda m: scores[m].get("nwrmsle", float("inf"))) if scores else None
        policies = [
            schemas.PolicyResult.model_validate(p)
            for p in queries.policies(release.id, None)
            if p.segment in _SUMMARY_SEGMENTS
        ]
        return schemas.Overview(
            release=schemas.ReleaseInfo.model_validate(release),
            series=queries.series_count(release.id),
            stores=len(queries.stores(release.id)),
            forecast_units_next_7d=forecast_7d,
            actual_units_last_7d=actual_7d,
            best_item_model=best,
            item_scores=scores,
            policy_totals=policies,
        )

    return release_response(request, release, compute, max_age=settings.api_cache_ttl_seconds)


@router.get("/models", response_model=list[schemas.ModelRun])
def models(
    request: Request,
    release: CurrentRelease,
    queries: Queries,
    settings: Settings,
    level: str | None = None,
) -> Response:
    return release_response(
        request,
        release,
        lambda: [schemas.ModelRun.model_validate(r) for r in queries.model_runs(level)],
        max_age=settings.api_cache_ttl_seconds,
    )


@router.get("/models/{run_id}/scores", response_model=list[schemas.Score])
def run_scores(
    run_id: uuid.UUID,
    request: Request,
    release: CurrentRelease,
    queries: Queries,
    settings: Settings,
) -> Response:
    return release_response(
        request,
        release,
        lambda: [schemas.Score.model_validate(s) for s in queries.run_scores(run_id)],
        max_age=settings.api_cache_ttl_seconds,
    )


@router.get("/analyses", response_model=list[schemas.Analysis])
def analyses(
    request: Request, release: CurrentRelease, queries: Queries, settings: Settings
) -> Response:
    def compute() -> list[schemas.Analysis]:
        rows = queries.analyses()
        return [
            schemas.Analysis(
                analysis=name,
                findings=[schemas.Finding(subject=r.subject, payload=r.payload) for r in group],
            )
            for name, group in groupby(rows, key=lambda r: r.analysis)
        ]

    return release_response(request, release, compute, max_age=settings.api_cache_ttl_seconds)


@router.get("/quality", response_model=list[schemas.QualityCheck])
def quality(
    request: Request, release: CurrentRelease, queries: Queries, settings: Settings
) -> Response:
    return release_response(
        request,
        release,
        lambda: [schemas.QualityCheck.model_validate(q) for q in queries.latest_quality()],
        max_age=settings.api_cache_ttl_seconds,
    )
