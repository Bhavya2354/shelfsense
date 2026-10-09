"""Engine and session factory, plus a COPY-based bulk loader for large frames."""

import io
from collections.abc import Iterator
from contextlib import contextmanager

import polars as pl
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import DatabaseSettings


def _driver_url(url: str) -> str:
    """Accept plain `postgresql://` URLs (as hosting providers hand them out)."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


def create_db_engine(settings: DatabaseSettings) -> Engine:
    return create_engine(
        _driver_url(settings.database_url.get_secret_value()),
        pool_size=settings.database_pool_size,
        pool_pre_ping=True,
        connect_args={"options": f"-c statement_timeout={settings.database_statement_timeout_ms}"},
    )


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def transaction(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as session, session.begin():
        yield session


def copy_frame(session: Session, table: str, frame: pl.DataFrame) -> int:
    """Stream a frame into `table` with COPY; far faster than INSERT for millions of rows."""
    if frame.is_empty():
        return 0
    buffer = io.BytesIO()
    frame.write_csv(buffer, include_header=False, null_value="")
    columns = ", ".join(frame.columns)
    raw = session.connection().connection.driver_connection
    with (
        raw.cursor() as cur,  # type: ignore[union-attr]
        cur.copy(f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '')") as copy,
    ):
        copy.write(buffer.getvalue())
    return frame.height
