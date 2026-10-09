"""ASGI application factory: `uvicorn --factory app.api.main:create_app`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app import __version__, config
from app.api.cache import TTLCache
from app.api.middleware import RequestContextMiddleware
from app.api.routes import forecasts, insights, planning, system
from app.config import ApiSettings, DatabaseSettings
from app.observability import configure_logging
from app.storage.database import create_db_engine, session_factory

API_PREFIX = "/api"


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or config.api_settings()
    configure_logging(config.runtime_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_db_engine(
            DatabaseSettings(
                database_url=settings.api_database_url,
                database_pool_size=settings.api_pool_size,
                database_statement_timeout_ms=settings.api_statement_timeout_ms,
            )
        )
        app.state.settings = settings
        app.state.sessions = session_factory(engine)
        app.state.cache = TTLCache(settings.api_cache_ttl_seconds, settings.api_cache_max_entries)
        yield
        engine.dispose()

    app = FastAPI(
        title="ShelfSense API",
        version=__version__,
        description="Demand forecasts, order plans and model diagnostics for grocery retail.",
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
        redoc_url=None,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_methods=["GET"],
        allow_headers=["If-None-Match", "X-Request-ID"],
        expose_headers=["ETag", "X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(RequestContextMiddleware)
    for router in (system.router, forecasts.router, planning.router, insights.router):
        app.include_router(router, prefix=API_PREFIX)
    return app
