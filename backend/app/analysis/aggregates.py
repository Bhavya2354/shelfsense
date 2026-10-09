"""Aggregated series the statistical analyses run on, pulled from the warehouse."""

from datetime import date

import polars as pl

from app.storage.warehouse import Warehouse


def national_daily(wh: Warehouse) -> pl.DataFrame:
    """Chain-wide units per day; returns are netted out by clipping at zero."""
    return wh.frame(
        """
        SELECT s.date,
               sum(greatest(s.unit_sales, 0)) AS unit_sales,
               any_value(t.transactions) AS transactions
        FROM sales s
        LEFT JOIN (SELECT date, sum(transactions) AS transactions FROM transactions GROUP BY date) t
            USING (date)
        GROUP BY s.date ORDER BY s.date
        """
    )


def family_daily(wh: Warehouse, start: date) -> pl.DataFrame:
    return wh.frame(
        """
        SELECT s.date, i.family, sum(greatest(s.unit_sales, 0)) AS unit_sales
        FROM sales s JOIN items i USING (item_nbr)
        WHERE s.date >= ?
        GROUP BY ALL ORDER BY i.family, s.date
        """,
        [start],
    )


def store_daily_transactions(wh: Warehouse) -> pl.DataFrame:
    return wh.frame(
        """
        SELECT t.date, t.store_nbr, st.city, t.transactions
        FROM transactions t JOIN stores st USING (store_nbr)
        WHERE t.transactions > 0
        """
    )


def dense_promo_panel(
    wh: Warehouse,
    start: date,
    end: date,
    *,
    min_coverage: float,
    series_per_family: int,
    seed: int,
) -> pl.DataFrame:
    """Store-item days for series that sell on almost every day of the window.

    The sales table only records days with a sale, so promotion status on zero
    days is unknown. Restricting to dense series keeps that selection bias small.
    """
    days = (end - start).days + 1
    return wh.frame(
        """
        WITH dense AS (
            SELECT store_nbr, item_nbr, i.family
            FROM sales s JOIN items i USING (item_nbr)
            WHERE s.date BETWEEN ? AND ? AND s.onpromotion IS NOT NULL
            GROUP BY ALL
            HAVING count(*) >= ? * ?
        ),
        sampled AS (
            SELECT * FROM dense
            QUALIFY row_number() OVER (
                PARTITION BY family ORDER BY hash(store_nbr, item_nbr, ?)
            ) <= ?
        )
        SELECT d.family, s.store_nbr, s.item_nbr, s.date,
               ln(1 + greatest(s.unit_sales, 0)) AS log_sales,
               s.onpromotion::INTEGER AS promo
        FROM sales s JOIN sampled d USING (store_nbr, item_nbr)
        WHERE s.date BETWEEN ? AND ?
        """,
        [start, end, min_coverage, days, seed, series_per_family, start, end],
    )
