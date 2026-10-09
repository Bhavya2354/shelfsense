"""Request-scoped dependencies and the release-aware response helper."""

import hashlib
from collections.abc import Callable, Iterator
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker

from app.api.cache import TTLCache
from app.api.queries import ReadQueries
from app.config import ApiSettings
from app.storage.models import Release


def get_settings(request: Request) -> ApiSettings:
    settings: ApiSettings = request.app.state.settings
    return settings


def get_session(request: Request) -> Iterator[Session]:
    factory: sessionmaker[Session] = request.app.state.sessions
    with factory() as session:
        yield session


def get_queries(session: Annotated[Session, Depends(get_session)]) -> ReadQueries:
    return ReadQueries(session)


Queries = Annotated[ReadQueries, Depends(get_queries)]
Settings = Annotated[ApiSettings, Depends(get_settings)]


def get_release(queries: Queries) -> Release:
    release = queries.current_release()
    if release is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "no forecast release published yet"
        )
    return release


CurrentRelease = Annotated[Release, Depends(get_release)]


class PageParams:
    def __init__(self, settings: Settings, page: int = 1, page_size: int | None = None) -> None:
        if page < 1:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "page starts at 1")
        size = page_size or settings.api_default_page_size
        if not 1 <= size <= settings.api_max_page_size:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"page_size must be between 1 and {settings.api_max_page_size}",
            )
        self.page = page
        self.page_size = size


Paging = Annotated[PageParams, Depends()]


def release_response(
    request: Request, release: Release, compute: Callable[[], Any], *, max_age: int
) -> Response:
    """Serve release-scoped data with an ETag; unchanged releases answer 304."""
    key = (str(release.id), request.url.path, str(sorted(request.query_params.multi_items())))
    etag = 'W/"' + hashlib.sha256(repr(key).encode()).hexdigest()[:32] + '"'
    headers = {"ETag": etag, "Cache-Control": f"public, max-age={max_age}"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    cache: TTLCache = request.app.state.cache
    payload = cache.get_or_compute(key, lambda: jsonable_encoder(compute()))
    return JSONResponse(payload, headers=headers)
