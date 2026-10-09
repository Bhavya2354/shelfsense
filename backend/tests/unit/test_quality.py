from datetime import date
from pathlib import Path

import polars as pl
import pytest

from app.config import StorageSettings
from app.errors import DataValidationError
from app.ingestion.quality import run_quality_gate
from app.storage.catalog import Table, table_path
from app.storage.warehouse import open_warehouse


def _write(settings: StorageSettings, table: Table, frame: pl.DataFrame) -> None:
    settings.curated_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(table_path(settings, table))


@pytest.fixture
def storage(tmp_path: Path) -> StorageSettings:
    settings = StorageSettings(data_dir=tmp_path)
    _write(settings, Table.STORES, pl.DataFrame({"store_nbr": [1, 2], "city": ["Quito", "Cuenca"]}))
    _write(settings, Table.ITEMS, pl.DataFrame({"item_nbr": [10, 11]}))
    return settings


def test_clean_data_passes_with_warnings_reported(storage: StorageSettings) -> None:
    _write(
        storage,
        Table.SALES,
        pl.DataFrame(
            {
                "date": [date(2017, 1, 1), date(2017, 1, 3)],
                "store_nbr": [1, 2],
                "item_nbr": [10, 11],
                "unit_sales": [3.0, -1.0],
                "onpromotion": [False, None],
            }
        ),
    )
    with open_warehouse(storage) as wh:
        results = {r.name: r for r in run_quality_gate(wh, storage.curated_dir / "q.json")}
    assert results["sales_negative_units"].violations == 1
    assert results["sales_calendar_gaps"].violations == 1
    assert results["sales_item_known"].passed


def test_unknown_items_stop_the_pipeline(storage: StorageSettings) -> None:
    _write(
        storage,
        Table.SALES,
        pl.DataFrame(
            {
                "date": [date(2017, 1, 1)],
                "store_nbr": [1],
                "item_nbr": [999],
                "unit_sales": [1.0],
                "onpromotion": [False],
            }
        ),
    )
    with (
        open_warehouse(storage) as wh,
        pytest.raises(DataValidationError, match="sales_item_known"),
    ):
        run_quality_gate(wh, storage.curated_dir / "q.json")
