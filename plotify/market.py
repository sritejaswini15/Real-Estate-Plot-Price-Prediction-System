"""Descriptive market statistics and comparable listings (no model involved)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .domain import MIN_LOCALITY_COUNT, OTHER_LOCALITY


def _quantiles(values: pd.Series) -> dict[str, float]:
    return {
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
    }


def market_summary(listings: pd.DataFrame) -> dict[str, Any]:
    """Per-locality price-per-sq-yd statistics.

    Localities with fewer than ``MIN_LOCALITY_COUNT`` listings are folded into
    "Other". A median of one or two listings would look more precise than it
    is.
    """
    df = listings.copy()
    counts = df["locality"].value_counts()
    frequent = set(counts[counts >= MIN_LOCALITY_COUNT].index)
    df["bucket"] = df["locality"].where(df["locality"].isin(frequent), OTHER_LOCALITY)
    rows = []
    for bucket, part in df.groupby("bucket"):
        rate = _quantiles(part["price_per_sqyd"])
        rows.append({
            "locality": str(bucket),
            "listings": int(len(part)),
            "distinct_plots": int(part["group_id"].nunique()),
            "rate_p25": rate["p25"],
            "rate_median": rate["median"],
            "rate_p75": rate["p75"],
            "median_area_sqyd": float(part["area_sqyd"].median()),
            "median_price_inr": float(part["price_inr"].median()),
            "members": sorted(part["locality"].unique().tolist()) if bucket == OTHER_LOCALITY else [],
        })
    rows.sort(key=lambda r: (r["locality"] == OTHER_LOCALITY, -r["rate_median"]))
    overall = _quantiles(df["price_per_sqyd"])
    return {
        "localities": rows,
        "overall": {
            "listings": int(len(df)),
            "distinct_plots": int(df["group_id"].nunique()),
            "rate_p25": overall["p25"],
            "rate_median": overall["median"],
            "rate_p75": overall["p75"],
            "median_area_sqyd": float(df["area_sqyd"].median()),
            "area_min": float(df["area_sqyd"].min()),
            "area_max": float(df["area_sqyd"].max()),
        },
    }


def find_comparables(listings: pd.DataFrame, *, area_sqyd: float, locality: str, facing: str | None,
                     road_width_ft: float | None, limit: int = 5) -> list[dict[str, Any]]:
    """Return the listings most similar to a plot, one per cross-posted group.

    The distance is hand-weighted and easy to explain. Plot size on a log
    scale counts most, then locality, then road width and facing. It is not
    learned from the data.
    """
    if listings.empty:
        return []
    df = listings.copy()
    distance = np.abs(np.log(df["area_sqyd"].astype(float)) - math.log(area_sqyd)) / math.log(1.5)
    distance = distance + np.where(df["locality"] == locality, 0.0, 1.0)
    if road_width_ft is not None:
        road = pd.to_numeric(df["road_width_ft"], errors="coerce").astype(float)
        distance = distance + np.where(road.isna(), 0.25, np.minimum(np.abs(road - road_width_ft) / 40.0, 1.0) * 0.5)
    if facing is not None:
        distance = distance + np.where(df["facing"].astype(object) == facing, 0.0, 0.25)
    df["distance"] = distance
    best = df.sort_values(["distance", "source_row"]).drop_duplicates("group_id", keep="first").head(limit)
    columns = ["source_row", "title", "locality", "project", "area_sqyd", "price_inr", "price_per_sqyd",
               "facing", "open_sides", "road_width_ft", "distance"]
    return [_clean_record(r) for r in best[columns].to_dict("records")]


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in record.items():
        if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
            clean[key] = None
        elif isinstance(value, np.generic):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean
