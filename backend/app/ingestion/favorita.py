"""Corporación Favorita competition data: download, unpack and convert to typed Parquet."""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import duckdb

from app.config import FavoritaSettings, HttpSettings, StorageSettings
from app.errors import DataValidationError
from app.ingestion.http_client import download_file
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
    """Downloads the competition archive once, then rebuilds only missing tables.

    The archive holds one zipped CSV per table (`train.csv.zip`, ...).
    """

    def __init__(
        self, favorita: FavoritaSettings, http: HttpSettings, storage: StorageSettings
    ) -> None:
        self._favorita = favorita
        self._http = http
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
            record_table(self._storage, spec.table, rows=rows, source=f"favorita:{spec.file_name}")
            logger.info("wrote table", extra={"table": spec.table, "rows": rows})

    def _ensure_csv(self, file_name: str) -> Path:
        csv_path = self._raw_dir / file_name
        if csv_path.exists():
            return csv_path
        inner = self._raw_dir / f"{file_name}.zip"
        if not inner.exists():
            self._download_archive()
        logger.info("extracting", extra={"archive": inner.name})
        with zipfile.ZipFile(inner) as zf:
            zf.extract(file_name, self._raw_dir)
        return csv_path

    def _download_archive(self) -> None:
        archive = self._raw_dir / "favorita.zip"
        if not archive.exists():
            logger.info("downloading competition archive")
            download_file(str(self._favorita.favorita_archive_url), archive, self._http)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if m.endswith(".csv.zip")]
            if not members:
                raise DataValidationError("archive contains no zipped CSV files")
            for member in members:
                target = self._raw_dir / Path(member).name
                target.write_bytes(zf.read(member))

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
