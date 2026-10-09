from fastapi import APIRouter, Request, Response

from app.api import schemas
from app.api.dependencies import CurrentRelease, Paging, Queries, Settings, release_response

router = APIRouter(tags=["planning"])


@router.get("/orders", response_model=schemas.Page[schemas.OrderLine])
def orders(
    request: Request,
    release: CurrentRelease,
    queries: Queries,
    settings: Settings,
    paging: Paging,
    store_nbr: int | None = None,
    family: str | None = None,
) -> Response:
    def compute() -> schemas.Page[schemas.OrderLine]:
        rows, total = queries.orders_page(
            release.id,
            store_nbr=store_nbr,
            family=family,
            page=paging.page,
            page_size=paging.page_size,
        )
        return schemas.Page[schemas.OrderLine](
            items=[schemas.OrderLine(**r) for r in rows],
            total=total,
            page=paging.page,
            page_size=paging.page_size,
        )

    return release_response(request, release, compute, max_age=settings.api_cache_ttl_seconds)


@router.get("/policies", response_model=list[schemas.PolicyResult])
def policies(
    request: Request,
    release: CurrentRelease,
    queries: Queries,
    settings: Settings,
    segment: str | None = None,
) -> Response:
    """Backtested policy outcomes; `segment` filters by prefix (e.g. `perishable`)."""
    return release_response(
        request,
        release,
        lambda: [
            schemas.PolicyResult.model_validate(p) for p in queries.policies(release.id, segment)
        ],
        max_age=settings.api_cache_ttl_seconds,
    )
