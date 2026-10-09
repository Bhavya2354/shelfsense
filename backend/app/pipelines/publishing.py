"""Publishing pipeline: training artifacts -> one immutable release in Postgres.

The release is written in full inside a transaction and only then marked
current, so the API never serves a mix of old and new rows.
"""

import json
import logging
import uuid
from datetime import date, timedelta
from typing import Any

import polars as pl

from app import config
from app.inventory.policies import calibrate, evaluate_policies, plan_orders
from app.storage.catalog import Table
from app.storage.database import create_db_engine, session_factory, transaction
from app.storage.models import (
    FamilyForecast,
    FamilySalesHistory,
    ItemForecast,
    OrderPlan,
    PolicyEvaluation,
    SalesHistory,
)
from app.storage.repositories import QualityRepository, ReferenceDataRepository, ReleaseRepository
from app.storage.warehouse import open_warehouse

logger = logging.getLogger(__name__)

NATIONAL_STORE = 0
ALL_FAMILIES = "ALL"


def _split_unique_id(frame: pl.DataFrame) -> pl.DataFrame:
    """'all' / 'all/12' / 'all/12/DAIRY' -> (store_nbr, family) with 0 / ALL for totals."""
    parts = pl.col("unique_id").str.split("/")
    return frame.with_columns(
        parts.list.get(1, null_on_oob=True)
        .cast(pl.Int16)
        .fill_null(NATIONAL_STORE)
        .alias("store_nbr"),
        parts.list.get(2, null_on_oob=True).fill_null(ALL_FAMILIES).alias("family"),
    ).drop("unique_id")


def run_publishing() -> dict[str, Any]:
    storage = config.storage_settings()
    publish = config.publish_settings()
    inventory = config.inventory_settings()
    forecast = config.forecast_settings()
    artifacts = storage.artifacts_dir
    report = json.loads((artifacts / "training_report.json").read_text(encoding="utf-8"))
    origin = date.fromisoformat(report["forecast_origin"])
    first_day = origin + timedelta(days=1)

    item_release = pl.read_parquet(artifacts / "item_release.parquet")
    item_backtest = pl.read_parquet(artifacts / "item_backtest.parquet")
    family_release = pl.read_parquet(artifacts / "family_release.parquet")

    with open_warehouse(storage) as wh:
        stores = wh.frame(
            """
            SELECT s.store_nbr, s.city, s.state, s.store_type, s.cluster, l.latitude, l.longitude
            FROM stores s LEFT JOIN store_locations l USING (city, state)
            """
            if wh.has(Table.STORE_LOCATIONS)
            else "SELECT *, NULL::DOUBLE AS latitude, NULL::DOUBLE AS longitude FROM stores"
        )
        items = wh.frame("SELECT item_nbr, family, item_class, perishable FROM items")
        # Serve the store-items that matter most: highest recent unit sales.
        top = wh.frame(
            """
            SELECT store_nbr, item_nbr FROM sales WHERE date > ?::DATE - ?::INTEGER
            GROUP BY ALL ORDER BY sum(greatest(unit_sales, 0)) DESC LIMIT ?
            """,
            [origin, publish.publish_history_days, publish.publish_top_series],
        )
        history = wh.frame(
            """
            SELECT store_nbr, item_nbr, date AS sale_date, greatest(unit_sales, 0) AS unit_sales,
                   coalesce(onpromotion, false) AS onpromotion
            FROM sales WHERE date > ?::DATE - ?::INTEGER
            """,
            [origin, publish.publish_history_days],
        ).join(top, on=["store_nbr", "item_nbr"])
        family_history = wh.frame(
            """
            WITH daily AS (
                SELECT s.store_nbr, i.family, s.date AS sale_date,
                       sum(greatest(s.unit_sales, 0)) AS unit_sales
                FROM sales s JOIN items i USING (item_nbr)
                WHERE s.date > ?::DATE - ?::INTEGER GROUP BY ALL
            )
            SELECT * FROM daily
            UNION ALL SELECT store_nbr, 'ALL', sale_date, sum(unit_sales) FROM daily GROUP BY ALL
            UNION ALL SELECT 0, 'ALL', sale_date, sum(unit_sales) FROM daily GROUP BY ALL
            """,
            [origin, publish.publish_family_history_days],
        )
    quality_path = storage.curated_dir / "_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else []

    # Inventory: calibrate on the older fold, evaluate on the newer, plan on the release.
    fold_days = sorted(item_backtest["first_day"].unique().to_list())
    older = item_backtest.filter(pl.col("first_day") == fold_days[0])
    newer = item_backtest.filter(pl.col("first_day") == fold_days[-1])
    seed = forecast.random_seed
    buckets = forecast.velocity_buckets
    policy_results = evaluate_policies(
        newer, items, calibrate(older, items, inventory, buckets, seed), inventory
    )
    release_frame = item_release.join(top, on=["store_nbr", "item_nbr"]).with_columns(
        pl.lit(first_day).alias("first_day")
    )
    orders = plan_orders(
        release_frame, items, calibrate(newer, items, inventory, buckets, seed), inventory
    )

    family_rows = _split_unique_id(family_release.rename({"ds": "target_date"}))
    factory = session_factory(create_db_engine(config.database_settings()))
    with transaction(factory) as session:
        ReferenceDataRepository(session).sync_stores(stores)
        ReferenceDataRepository(session).sync_items(items)
        QualityRepository(session).record(
            {
                "check_name": q["name"],
                "table_name": q["table"],
                "severity": q["severity"],
                "violations": q["violations"],
            }
            for q in quality
        )
        releases = ReleaseRepository(session)
        run_id = report["run_ids"].get("item:ensemble")
        release_id = releases.create(
            forecast_origin=origin,
            item_run_id=uuid.UUID(run_id) if run_id else None,
            notes=f"backtest folds {', '.join(report['folds'])}",
        )
        loaded = {
            ItemForecast.__tablename__: releases.load(
                release_id,
                ItemForecast.__tablename__,
                release_frame.select(
                    "store_nbr", "item_nbr", "target_date", "p10", "p50", "p90", "onpromotion"
                ),
            ),
            FamilyForecast.__tablename__: releases.load(
                release_id,
                FamilyForecast.__tablename__,
                family_rows.select(
                    "store_nbr", "family", "model_name", "target_date", "p10", "p50", "p90"
                ),
            ),
            SalesHistory.__tablename__: releases.load(
                release_id, SalesHistory.__tablename__, history
            ),
            FamilySalesHistory.__tablename__: releases.load(
                release_id, FamilySalesHistory.__tablename__, family_history
            ),
            OrderPlan.__tablename__: releases.load(release_id, OrderPlan.__tablename__, orders),
            PolicyEvaluation.__tablename__: releases.load(
                release_id, PolicyEvaluation.__tablename__, policy_results
            ),
        }
        releases.publish(release_id)
        pruned = releases.prune(publish.publish_keep_releases)
    logger.info(
        "release published", extra={"release_id": str(release_id), "rows": loaded, "pruned": pruned}
    )
    return {"release_id": str(release_id), "rows": loaded, "pruned": pruned}
