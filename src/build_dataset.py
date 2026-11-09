"""Build one analysis row per customer, describing their FIRST order and whether they came back.

Design
------
- Population: customers whose first order was *delivered* and was placed between
  2017-01-01 and 2018-02-28, so every customer has at least 180 days of follow-up
  before the data ends (late Aug 2018).
- Exposure: first order arrived after the estimated delivery date ("late"), plus days late.
- Outcome: the customer placed another order AFTER the first one was delivered and
  within 180 days of the first purchase. Orders placed before delivery (often a split
  basket) do not count: the customer had not yet experienced the delivery.
- Controls: state, product category, basket value, freight share, items, purchase month.

Output: data/processed/first_orders.parquet (+ .csv)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

WINDOW_DAYS = 180
COHORT_START, COHORT_END = "2017-01-01", "2018-02-28"


def load() -> dict[str, pd.DataFrame]:
    ts = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
          "order_delivered_customer_date", "order_estimated_delivery_date"]
    return {
        "orders": pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=ts),
        "customers": pd.read_csv(RAW / "olist_customers_dataset.csv"),
        "items": pd.read_csv(RAW / "olist_order_items_dataset.csv"),
        "products": pd.read_csv(RAW / "olist_products_dataset.csv"),
        "cats": pd.read_csv(RAW / "product_category_name_translation.csv"),
        "reviews": pd.read_csv(RAW / "olist_order_reviews_dataset.csv",
                               parse_dates=["review_creation_date"]),
    }


def build(t: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    o = t["orders"].merge(t["customers"], on="customer_id", how="left")
    dq = {"orders": len(o), "customers": o.customer_unique_id.nunique()}

    # Order-level basket facts.
    it = (t["items"].merge(t["products"][["product_id", "product_category_name"]], on="product_id", how="left")
          .merge(t["cats"], on="product_category_name", how="left"))
    it["category"] = it.product_category_name_english.fillna("unknown")
    basket = it.groupby("order_id").agg(
        items=("order_item_id", "count"), price=("price", "sum"), freight=("freight_value", "sum"),
        sellers=("seller_id", "nunique"),
        category=("category", lambda s: s.value_counts().index[0]))
    o = o.merge(basket, on="order_id", how="left")

    # One review per order (a few orders have 2–3; keep the latest).
    rv = (t["reviews"].sort_values("review_creation_date")
          .drop_duplicates("order_id", keep="last")[["order_id", "review_score"]])
    o = o.merge(rv, on="order_id", how="left")

    o = o.sort_values(["customer_unique_id", "order_purchase_timestamp", "order_id"])
    o["order_rank"] = o.groupby("customer_unique_id").cumcount() + 1
    first = o[o.order_rank == 1].copy()
    dq["first_orders"] = len(first)

    # Keep delivered first orders with a delivery date, inside the cohort window.
    first = first[first.order_status.eq("delivered") & first.order_delivered_customer_date.notna()]
    dq["first_orders_delivered"] = len(first)
    in_window = first.order_purchase_timestamp.between(COHORT_START, f"{COHORT_END} 23:59:59")
    first = first[in_window].copy()
    dq["population"] = len(first)

    # Exposure.
    first["days_late"] = (first.order_delivered_customer_date.dt.normalize()
                          - first.order_estimated_delivery_date).dt.days
    first["late"] = first.days_late > 0
    first["delivery_days"] = (first.order_delivered_customer_date - first.order_purchase_timestamp).dt.days
    first["late_bucket"] = pd.cut(first.days_late, [-np.inf, 0, 3, 7, 14, np.inf],
                                  labels=["on time", "1-3 days", "4-7 days", "8-14 days", "15+ days"])

    # Outcome: a later order placed after delivery and within the window.
    later = o[o.order_rank > 1][["customer_unique_id", "order_purchase_timestamp", "price"]]
    m = first[["customer_unique_id", "order_purchase_timestamp", "order_delivered_customer_date"]].merge(
        later, on="customer_unique_id", how="inner", suffixes=("_first", "_next"))
    ok = ((m.order_purchase_timestamp_next > m.order_delivered_customer_date) &
          (m.order_purchase_timestamp_next <= m.order_purchase_timestamp_first + pd.Timedelta(days=WINDOW_DAYS)))
    returners = m[ok].groupby("customer_unique_id").agg(next_order_value=("price", "first"))
    first = first.merge(returners, on="customer_unique_id", how="left")
    first["repeat_180d"] = first.next_order_value.notna()
    # Same-basket repeats (ordered before delivery) are excluded but counted for transparency.
    pre = m[m.order_purchase_timestamp_next <= m.order_delivered_customer_date].customer_unique_id.nunique()
    dq["customers_with_pre_delivery_second_order"] = int(pre)

    # Controls.
    first["purchase_month"] = first.order_purchase_timestamp.dt.to_period("M").astype(str)
    first["log_price"] = np.log1p(first.price)
    first["freight_share"] = first.freight / (first.price + first.freight)
    top_cat = first.category.value_counts().index[:15]
    first["category_grp"] = first.category.where(first.category.isin(top_cat), "other")
    top_state = first.customer_state.value_counts().index[:10]
    first["state_grp"] = first.customer_state.where(first.customer_state.isin(top_state), "other")

    cols = ["customer_unique_id", "order_id", "order_purchase_timestamp", "purchase_month",
            "customer_state", "state_grp", "category", "category_grp", "items", "sellers", "price",
            "freight", "log_price", "freight_share", "delivery_days", "days_late", "late",
            "late_bucket", "review_score", "repeat_180d", "next_order_value"]
    return first[cols].reset_index(drop=True), dq


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, dq = build(load())
    df.to_csv(OUT / "first_orders.csv", index=False)
    for k, v in dq.items():
        print(f"{k:>42}: {v:,}")
    print(f"{'late share':>42}: {df.late.mean():.1%}")
    print(f"{'repeat within 180d':>42}: {df.repeat_180d.mean():.2%}")


if __name__ == "__main__":
    main()
