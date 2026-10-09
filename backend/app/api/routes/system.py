from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api import schemas
from app.api.dependencies import Queries

router = APIRouter(tags=["system"])


@router.get("/health", response_model=schemas.Health)
def health(queries: Queries) -> JSONResponse | schemas.Health:
    try:
        queries.ping()
        release = queries.current_release()
    except SQLAlchemyError:
        body = schemas.Health(
            status="degraded", database="unreachable", release_id=None, forecast_origin=None
        )
        return JSONResponse(
            body.model_dump(mode="json"), status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    return schemas.Health(
        status="ok",
        database="ok",
        release_id=release.id if release else None,
        forecast_origin=release.forecast_origin if release else None,
    )
