"""Data quality gate run after ingestion; errors stop the pipeline, warnings are reported."""

import json
import logging
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from app.errors import DataValidationError
from app.storage.catalog import Table
from app.storage.warehouse import Warehouse

logger = logging.getLogger(__name__)


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Check:
    name: str
    table: Table
    severity: Severity
    sql: str
    """Returns a single number: the count of offending rows (0 means the check passes)."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    table: str
    severity: str
    violations: int

    @property
    def passed(self) -> bool:
        return self.violations == 0


CHECKS: tuple[Check, ...] = (
    Check(
        "stores_key_unique",
        Table.STORES,
        Severity.ERROR,
        "SELECT count(*) - count(DISTINCT store_nbr) FROM stores",
    ),
    Check(
        "items_key_unique",
        Table.ITEMS,
        Severity.ERROR,
        "SELECT count(*) - count(DISTINCT item_nbr) FROM items",
    ),
    Check(
        "sales_keys_not_null",
        Table.SALES,
        Severity.ERROR,
        "SELECT count(*) FROM sales WHERE date IS NULL OR store_nbr IS NULL OR item_nbr IS NULL",
    ),
    Check(
        "sales_store_known",
        Table.SALES,
        Severity.ERROR,
        "SELECT count(DISTINCT store_nbr) FROM sales ANTI JOIN stores USING (store_nbr)",
    ),
    Check(
        "sales_item_known",
        Table.SALES,
        Severity.ERROR,
        "SELECT count(DISTINCT item_nbr) FROM sales ANTI JOIN items USING (item_nbr)",
    ),
    Check(
        "sales_duplicate_rows",
        Table.SALES,
        Severity.ERROR,
        """SELECT count(*) FROM (SELECT date, store_nbr, item_nbr FROM sales
             GROUP BY ALL HAVING count(*) > 1)""",
    ),
    Check(
        "test_item_known",
        Table.TEST,
        Severity.ERROR,
        "SELECT count(DISTINCT item_nbr) FROM test ANTI JOIN items USING (item_nbr)",
    ),
    Check(
        "test_follows_train",
        Table.TEST,
        Severity.ERROR,
        "SELECT count(*) FROM test WHERE date <= (SELECT max(date) FROM sales)",
    ),
    Check(
        "sales_negative_units",
        Table.SALES,
        Severity.WARNING,
        "SELECT count(*) FROM sales WHERE unit_sales < 0",
    ),
    Check(
        "sales_promo_unknown",
        Table.SALES,
        Severity.WARNING,
        "SELECT count(*) FROM sales WHERE onpromotion IS NULL",
    ),
    Check(
        "sales_calendar_gaps",
        Table.SALES,
        Severity.WARNING,
        """SELECT count(*) FROM range(
                 (SELECT min(date) FROM sales), (SELECT max(date) FROM sales) + 1, INTERVAL 1 DAY
             ) AS r(d) ANTI JOIN (SELECT DISTINCT date FROM sales) s ON s.date = r.d""",
    ),
    Check(
        "oil_missing_days",
        Table.OIL,
        Severity.WARNING,
        "SELECT count(*) FROM oil WHERE price IS NULL",
    ),
    Check(
        "oil_sources_disagree",
        Table.OIL,
        Severity.WARNING,
        """SELECT count(*) FROM oil JOIN oil_kaggle k USING (date)
             WHERE abs(oil.price - k.price) > 0.01""",
    ),
    Check(
        "weather_city_coverage",
        Table.WEATHER,
        Severity.ERROR,
        "SELECT count(DISTINCT city) FROM stores ANTI JOIN weather USING (city)",
    ),
    Check(
        "weather_missing_values",
        Table.WEATHER,
        Severity.WARNING,
        "SELECT count(*) FROM weather WHERE temperature_2m_mean IS NULL",
    ),
)


def run_quality_gate(warehouse: Warehouse, report_path: Path) -> list[CheckResult]:
    results = [
        CheckResult(c.name, c.table, c.severity, int(warehouse.scalar(c.sql)))
        for c in CHECKS
        if warehouse.has(c.table)
    ]
    report_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    for r in results:
        level = logging.INFO if r.passed else logging.WARNING
        logger.log(level, "quality check", extra={"check": r.name, "violations": r.violations})
    failed = [r.name for r in results if r.severity == Severity.ERROR and not r.passed]
    if failed:
        raise DataValidationError(f"quality gate failed: {', '.join(failed)}")
    return results
