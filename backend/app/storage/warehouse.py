"""Read-only analytical access to the curated Parquet tables through DuckDB."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import duckdb
import polars as pl

from app.config import StorageSettings
from app.storage.catalog import Table, table_path


class Warehouse:
    """Exposes each curated Parquet file as a DuckDB view named after its table."""

    def __init__(self, settings: StorageSettings, *, memory_limit: str | None = None) -> None:
        self._settings = settings
        self._con = duckdb.connect()
        if memory_limit:
            self._con.execute(f"SET memory_limit = '{memory_limit}'")
        self.refresh()

    def refresh(self) -> None:
        for table in Table:
            path = table_path(self._settings, table)
            if path.exists():
                self._con.execute(
                    # Identifier comes from the Table enum, never from input.
                    f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet(?)",  # noqa: S608
                    [str(path)],
                )

    def has(self, table: Table) -> bool:
        return table_path(self._settings, table).exists()

    def frame(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        return self._con.execute(sql, params or []).pl()

    def scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        row = self._con.execute(sql, params or []).fetchone()
        return None if row is None else row[0]

    def close(self) -> None:
        self._con.close()


@contextmanager
def open_warehouse(settings: StorageSettings, **kwargs: Any) -> Iterator[Warehouse]:
    warehouse = Warehouse(settings, **kwargs)
    try:
        yield warehouse
    finally:
        warehouse.close()
