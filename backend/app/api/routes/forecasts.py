from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.api import schemas
from app.api.dependencies import CurrentRelease, Paging, Queries, Settings, release_response

router = APIRouter(tags=["forecasts"])


@router.get("/stores", response_model=list[schemas.Store])
def stores(
    request: Request, release: CurrentRelease, queries: Queries, settings: Settings
) -> Response:
    return release_response(
        request,
        release,
        lambda: [schemas.Store(**s) for s in queries.stores(release.id)],
        max_age=settings.api_cache_ttl_seconds,
    )


@router.get("/families", response_model=list[str])
def families(
    request: Request, release: CurrentRelease, queries: Queries, settings: Settings
) -> Response:
    return release_response(
        request, release, queries.families, max_age=settings.api_cache_ttl_seconds
    )


@router.get("/series", response_model=schemas.Page[schemas.SeriesSummary])
def series(
    request: Request,
    release: CurrentRelease,
    queries: Queries,
    settings: Settings,
    paging: Paging,
    store_nbr: int | None = None,
    family: str | None = None,
    item_nbr: int | None = None,
) -> Response:
    def compute() -> schemas.Page[schemas.SeriesSummary]:
        rows, total = queries.series_page(
            release.id,
            store_nbr=store_nbr,
            family=family,
            item_nbr=item_nbr,
            page=paging.page,
            page_size=paging.page_size,
        )
        return schemas.Page[schemas.SeriesSummary](
            items=[schemas.SeriesSummary(**r) for r in rows],
            total=total,
            page=paging.page,
            page_size=paging.page_size,
        )

    return release_response(request, release, compute, max_age=settings.api_cache_ttl_seconds)


@router.get("/series/{store_nbr}/{item_nbr}", response_model=schemas.SeriesDetail)
def series_detail(
    store_nbr: int,
    item_nbr: int,
    request: Request,
    release: CurrentRelease,
    queries: Queries,
    settings: Settings,
) -> Response:
    detail = queries.series_detail(release.id, store_nbr, item_nbr)
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "series not in the current release")
    return release_response(
        request,
        release,
        lambda: schemas.SeriesDetail(**detail),
        max_age=settings.api_cache_ttl_seconds,
    )


@router.get("/hierarchy/{store_nbr}/{family}", response_model=schemas.HierarchyDetail)
def hierarchy(
    store_nbr: int,
    family: str,
    request: Request,
    release: CurrentRelease,
    queries: Queries,
    settings: Settings,
    models: Annotated[list[str] | None, Query()] = None,
) -> Response:
    """Store 0 is the national total; family `ALL` is the store total."""
    detail = queries.hierarchy_detail(release.id, store_nbr, family, models)
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no forecasts for that store and family")
    return release_response(
        request,
        release,
        lambda: schemas.HierarchyDetail(**detail),
        max_age=settings.api_cache_ttl_seconds,
    )
