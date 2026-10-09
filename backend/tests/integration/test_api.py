"""API and repository tests against a real Postgres (set TEST_DATABASE_URL)."""

import os
import uuid
from collections.abc import Iterator
from datetime import date, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, select

from app.api.main import create_app
from app.config import ApiSettings, DatabaseSettings
from app.storage.database import create_db_engine, session_factory, transaction
from app.storage.models import Base, ItemForecast, Release
from app.storage.repositories import ReferenceDataRepository, ReleaseRepository

pytestmark = pytest.mark.integration
DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
ORIGIN = date(2017, 8, 15)


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    if not DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    engine = create_db_engine(DatabaseSettings(database_url=SecretStr(DATABASE_URL)))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


def _publish(engine: Engine, units: float) -> uuid.UUID:
    factory = session_factory(engine)
    days = [ORIGIN + timedelta(days=d) for d in range(1, 17)]
    with transaction(factory) as session:
        ref = ReferenceDataRepository(session)
        ref.sync_stores(
            pl.DataFrame(
                {
                    "store_nbr": [1, 2],
                    "city": ["QUITO", "CUENCA"],
                    "state": ["P", "A"],
                    "store_type": ["A", "B"],
                    "cluster": [1, 2],
                    "latitude": [None, None],
                    "longitude": [None, None],
                }
            )
        )
        ref.sync_items(
            pl.DataFrame(
                {
                    "item_nbr": [10, 11],
                    "family": ["DAIRY", "BREAD"],
                    "item_class": [1, 2],
                    "perishable": [True, False],
                }
            )
        )
        releases = ReleaseRepository(session)
        release_id = releases.create(forecast_origin=ORIGIN, item_run_id=None, notes="test")
        forecasts = pl.DataFrame(
            [
                (s, i, d, units * 0.5, units, units * 1.5, False)
                for s in (1, 2)
                for i in (10, 11)
                for d in days
            ],
            schema=["store_nbr", "item_nbr", "target_date", "p10", "p50", "p90", "onpromotion"],
            orient="row",
        )
        releases.load(release_id, ItemForecast.__tablename__, forecasts)
        releases.publish(release_id)
    return release_id


@pytest.fixture(scope="module")
def client(engine: Engine) -> Iterator[TestClient]:
    _publish(engine, units=4.0)
    settings = ApiSettings(
        api_database_url=SecretStr(DATABASE_URL or ""), api_cors_origins=["https://app.test"]
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_reports_the_current_release(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["forecast_origin"] == ORIGIN.isoformat()


def test_series_paging_and_validation(client: TestClient) -> None:
    page = client.get("/api/series", params={"page_size": 3}).json()
    assert page["total"] == 4
    assert len(page["items"]) == 3
    assert client.get("/api/series", params={"page_size": 10_000}).status_code == 422
    assert client.get("/api/series", params={"family": "DAIRY"}).json()["total"] == 2


def test_series_detail_and_not_found(client: TestClient) -> None:
    detail = client.get("/api/series/1/10").json()
    assert len(detail["forecast"]) == 16
    assert detail["family"] == "DAIRY"
    assert client.get("/api/series/1/999").status_code == 404


def test_etag_revalidation_returns_304(client: TestClient) -> None:
    first = client.get("/api/stores")
    etag = first.headers["etag"]
    again = client.get("/api/stores", headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert first.headers["x-content-type-options"] == "nosniff"
    assert "x-request-id" in first.headers


def test_cors_allows_only_configured_origins(client: TestClient) -> None:
    allowed = client.get("/api/health", headers={"Origin": "https://app.test"})
    blocked = client.get("/api/health", headers={"Origin": "https://evil.test"})
    assert allowed.headers.get("access-control-allow-origin") == "https://app.test"
    assert "access-control-allow-origin" not in blocked.headers


def test_publishing_a_new_release_swaps_atomically_and_prunes(
    engine: Engine, client: TestClient
) -> None:
    newer = _publish(engine, units=9.0)
    factory = session_factory(engine)
    with factory() as session:
        current = session.scalars(select(Release).where(Release.is_current)).all()
        assert [r.id for r in current] == [newer]
    stores = client.get("/api/series/1/10").json()
    assert stores["forecast"][0]["p50"] == 9.0  # new release, new cache key
    with transaction(factory) as session:
        assert ReleaseRepository(session).prune(keep=1) >= 1
