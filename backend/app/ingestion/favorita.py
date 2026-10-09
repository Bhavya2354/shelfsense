"""Corporación Favorita competition data: download, unpack and convert to typed Parquet."""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import duckdb

from app.config import KaggleSettings, StorageSettings
from app.errors import ExternalServiceError
from app.storage.catalog import Table, is_current, record_table, table_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CsvSpec:
    """How one competition CSV maps onto a curated table."""

    table: Table
    file_name: str
    columns: dict[str, str]
    select: str = "*"


# Column types are declared up front so a 125M-row file never relies on type sniffing.
SPECS: tuple[CsvSpec, ...] = (
    CsvSpec(
        Table.STORES,
        "stores.csv",
        {
            "store_nbr": "SMALLINT",
            "city": "VARCHAR",
            "state": "VARCHAR",
            "type": "VARCHAR",
            "cluster": "SMALLINT",
        },
        "store_nbr, city, state, type AS store_type, cluster",
    ),
    CsvSpec(
        Table.ITEMS,
        "items.csv",
        {"item_nbr": "INTEGER", "family": "VARCHAR", "class": "SMALLINT", "perishable": "TINYINT"},
        "item_nbr, family, class AS item_class, perishable = 1 AS perishable",
    ),
    CsvSpec(
        Table.HOLIDAYS,
        "holidays_events.csv",
        {
            "date": "DATE",
            "type": "VARCHAR",
            "locale": "VARCHAR",
            "locale_name": "VARCHAR",
            "description": "VARCHAR",
            "transferred": "BOOLEAN",
        },
        "date, type AS holiday_type, locale, locale_name, description, transferred",
    ),
    CsvSpec(
        Table.OIL_KAGGLE,
        "oil.csv",
        {"date": "DATE", "dcoilwtico": "DOUBLE"},
        "date, dcoilwtico AS price",
    ),
    CsvSpec(
        Table.TRANSACTIONS,
        "transactions.csv",
        {"date": "DATE", "store_nbr": "SMALLINT", "transactions": "INTEGER"},
    ),
    CsvSpec(
        Table.TEST,
        "test.csv",
        {
            "id": "BIGINT",
            "date": "DATE",
            "store_nbr": "SMALLINT",
            "item_nbr": "INTEGER",
            "onpromotion": "BOOLEAN",
        },
    ),
    CsvSpec(
        Table.SALES,
        "train.csv",
        {
            "id": "BIGINT",
            "date": "DATE",
            "store_nbr": "SMALLINT",
            "item_nbr": "INTEGER",
            "unit_sales": "DOUBLE",
            "onpromotion": "BOOLEAN",
        },
        "date, store_nbr, item_nbr, CAST(unit_sales AS FLOAT) AS unit_sales, onpromotion",
    ),
)


class FavoritaSource:
    """Downloads the competition bundle once, then rebuilds only missing tables."""

    def __init__(self, kaggle: KaggleSettings, storage: StorageSettings) -> None:
        self._kaggle = kaggle
        self._storage = storage
        self._raw_dir = storage.raw_dir / "favorita"

    def run(self, *, force: bool = False) -> None:
        pending = [s for s in SPECS if force or not is_current(self._storage, s.table)]
        if not pending:
            logger.info("favorita tables already current")
            return
        self._storage.curated_dir.mkdir(parents=True, exist_ok=True)
        for spec in pending:
            csv_path = self._ensure_csv(spec.file_name)
            rows = self._to_parquet(spec, csv_path)
            record_table(self._storage, spec.table, rows=rows, source=f"kaggle:{spec.file_name}")
            logger.info("wrote table", extra={"table": spec.table, "rows": rows})

    def _ensure_csv(self, file_name: str) -> Path:
        csv_path = self._raw_dir / file_name
        if csv_path.exists():
            return csv_path
        archive = self._raw_dir / f"{file_name}.7z"
        if not archive.exists():
            self._download_bundle()
        import py7zr

        logger.info("extracting", extra={"archive": archive.name})
        with py7zr.SevenZipFile(archive) as seven:
            seven.extractall(self._raw_dir)
        return csv_path

    def _download_bundle(self) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        self._raw_dir.mkdir(parents=True, exist_ok=True)
        slug = self._kaggle.kaggle_competition
        logger.info("downloading competition bundle", extra={"competition": slug})
        api = KaggleApi()
        api.authenticate()
        try:
            api.competition_download_files(slug, path=str(self._raw_dir), quiet=True)
        except Exception as exc:
            raise ExternalServiceError(f"kaggle download failed: {exc}") from exc
        bundle = self._raw_dir / f"{slug}.zip"
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(self._raw_dir)
        bundle.unlink()

    def _to_parquet(self, spec: CsvSpec, csv_path: Path) -> int:
        target = table_path(self._storage, spec.table)
        tmp = target.with_suffix(".parquet.tmp")
        con = duckdb.connect()
        try:
            # The projection and target path come from the static SPECS table above.
            copy_sql = f"""
                COPY (
                    SELECT {spec.select}
                    FROM read_csv(?, header = true, columns = ?, nullstr = '')
                ) TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
            """  # noqa: S608
            con.execute(copy_sql, [str(csv_path), spec.columns])
            count = con.execute("SELECT count(*) FROM read_parquet(?)", [str(tmp)]).fetchone()
            rows = int(count[0]) if count else 0
        finally:
            con.close()
        tmp.replace(target)
        return rows
