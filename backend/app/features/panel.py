"""Dense store-item x day matrices that every item-level model is built from."""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from app.storage.warehouse import Warehouse


@dataclass(frozen=True)
class Panel:
    """Aligned arrays: row i of every matrix is the series described by `keys` row i.

    `sales` holds log1p(units) with days without a record set to zero (the sales
    table only stores days with a sale). `promo` extends `horizon` days past the
    last observed day because promotions are planned, and so known, in advance.
    """

    keys: pl.DataFrame
    start: date
    last_observed: date
    horizon: int
    sales: np.ndarray
    promo: np.ndarray

    @property
    def n_series(self) -> int:
        return self.keys.height

    def day(self, when: date) -> int:
        return (when - self.start).days

    def date_at(self, index: int) -> date:
        return self.start + timedelta(days=index)


def build_panel(wh: Warehouse, last_observed: date, history_days: int, horizon: int) -> Panel:
    """Series sold in the history window plus every series scored in the future window."""
    start = last_observed - timedelta(days=history_days - 1)
    future_end = last_observed + timedelta(days=horizon)
    keys = wh.frame(
        """
        WITH series AS (
            SELECT DISTINCT store_nbr, item_nbr FROM sales WHERE date BETWEEN ? AND ?
            UNION
            SELECT DISTINCT store_nbr, item_nbr FROM test WHERE date BETWEEN ? AND ?
        )
        SELECT s.store_nbr, s.item_nbr, i.family, i.item_class, i.perishable,
               st.city, st.state, st.store_type, st.cluster
        FROM series s
        JOIN items i USING (item_nbr)
        JOIN stores st USING (store_nbr)
        ORDER BY s.store_nbr, s.item_nbr
        """,
        [start, last_observed, start, future_end],
    ).with_row_index("row")

    observed_days = history_days
    sales = np.zeros((keys.height, observed_days), dtype=np.float32)
    promo = np.zeros((keys.height, observed_days + horizon), dtype=np.float32)

    lookup = keys.select("store_nbr", "item_nbr", "row")
    observed = wh.frame(
        """
        SELECT store_nbr, item_nbr, date_diff('day', ?, date) AS d,
               ln(1 + greatest(unit_sales, 0)) AS y, coalesce(onpromotion, false) AS p
        FROM sales WHERE date BETWEEN ? AND ?
        """,
        [start, start, last_observed],
    ).join(lookup, on=["store_nbr", "item_nbr"])
    rows, days = observed["row"].to_numpy(), observed["d"].to_numpy()
    sales[rows, days] = observed["y"].to_numpy()
    promo[rows, days] = observed["p"].to_numpy()

    future = wh.frame(
        """
        SELECT store_nbr, item_nbr, date_diff('day', ?, date) AS d,
               coalesce(onpromotion, false) AS p
        FROM test WHERE date BETWEEN ? AND ?
        """,
        [start, last_observed + timedelta(days=1), future_end],
    ).join(lookup, on=["store_nbr", "item_nbr"])
    promo[future["row"].to_numpy(), future["d"].to_numpy()] = future["p"].to_numpy()

    return Panel(
        keys=keys.drop("row"),
        start=start,
        last_observed=last_observed,
        horizon=horizon,
        sales=sales,
        promo=promo,
    )
