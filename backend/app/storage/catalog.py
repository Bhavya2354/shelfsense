"""Names and locations of every curated dataset the pipeline produces."""

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.config import StorageSettings


class Table(StrEnum):
    SALES = "sales"
    TEST = "test"
    ITEMS = "items"
    STORES = "stores"
    TRANSACTIONS = "transactions"
    HOLIDAYS = "holidays"
    OIL_KAGGLE = "oil_kaggle"
    OIL = "oil"
    STORE_LOCATIONS = "store_locations"
    WEATHER = "weather"


def table_path(settings: StorageSettings, table: Table) -> Path:
    return settings.curated_dir / f"{table}.parquet"


def _manifest_path(settings: StorageSettings) -> Path:
    return settings.curated_dir / "_manifest.json"


def read_manifest(settings: StorageSettings) -> dict[str, dict[str, Any]]:
    path = _manifest_path(settings)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record_table(settings: StorageSettings, table: Table, *, rows: int, source: str) -> None:
    """Note a freshly written table so reruns can skip it and reports can cite it."""
    manifest = read_manifest(settings)
    manifest[table] = {
        "rows": rows,
        "source": source,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _manifest_path(settings).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def is_current(settings: StorageSettings, table: Table) -> bool:
    return table_path(settings, table).exists() and table in read_manifest(settings)
