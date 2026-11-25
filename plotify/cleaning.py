"""Parse and validate raw plot listings into a typed modelling table.

The source is an export of residential-plot listings, as scraped from
99acres.com. Every ``parse_*`` function turns one messy field into a
typed value or ``None``. :func:`build_dataset` applies them row by row and
records every row it drops and why, so the cleaning step can be audited.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .domain import (
    DEFAULT_REGION,
    FACINGS,
    MAX_AREA_SQYD,
    MAX_OPEN_SIDES,
    MAX_RATE_INR_PER_SQYD,
    MAX_ROAD_WIDTH_FT,
    MIN_AREA_SQYD,
    MIN_RATE_INR_PER_SQYD,
    MIN_ROAD_WIDTH_FT,
)


class SchemaError(ValueError):
    """The raw table is missing columns the pipeline needs."""


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "n/a", "na", "nan", "none", "-"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_PRICE_UNITS = {
    "": 1.0,
    "cr": 1e7,
    "crore": 1e7,
    "crores": 1e7,
    "l": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
    "lakh": 1e5,
    "lakhs": 1e5,
}
_NUMBER_WITH_UNIT = re.compile(r"([0-9]*\.?[0-9]+)\s*([a-z.\s]*)")


def parse_price_inr(value: Any) -> float | None:
    """Parse prices such as ``"85 Lac"``, ``"1.2 Crore"``, ``"₹45,00,000"``.

    Returns rupees, or ``None`` when the value is missing, a range, or not a
    positive amount.
    """
    if _is_missing(value):
        return None
    if _is_number(value):
        return float(value) if value > 0 and math.isfinite(value) else None
    text = str(value).strip().lower().replace(",", "")
    text = re.sub(r"^(₹|rs\.?|inr)\s*", "", text)
    match = _NUMBER_WITH_UNIT.fullmatch(text)
    if not match:
        return None
    unit = match.group(2).replace(".", "").strip()
    multiplier = _PRICE_UNITS.get(unit)
    if multiplier is None:
        return None
    amount = float(match.group(1)) * multiplier
    return amount if amount > 0 else None


_AREA_UNITS_IN_TEXT = {
    "": 1.0,  # the export records plot areas in square yards
    "sqyd": 1.0,
    "sqyds": 1.0,
    "sqyard": 1.0,
    "sqyards": 1.0,
    "sqyrd": 1.0,
    "sqyrds": 1.0,
    "yards": 1.0,
    "sqft": 1.0 / 9.0,
    "sqfeet": 1.0 / 9.0,
    "sqm": 1.0 / 0.83612736,
    "sqmeter": 1.0 / 0.83612736,
    "sqmeters": 1.0 / 0.83612736,
    "sqmetre": 1.0 / 0.83612736,
    "sqmetres": 1.0 / 0.83612736,
    "guntha": 121.0,
    "gunthas": 121.0,
    "gunta": 121.0,
    "guntas": 121.0,
    "acre": 4840.0,
    "acres": 4840.0,
}


def parse_area_sqyd(value: Any) -> float | None:
    """Parse a plot area to square yards (``200``, ``"1800 sq.ft."``, ``"2 acres"``)."""
    if _is_missing(value):
        return None
    if _is_number(value):
        return float(value) if value > 0 and math.isfinite(value) else None
    text = str(value).strip().lower().replace(",", "")
    match = _NUMBER_WITH_UNIT.fullmatch(text)
    if not match:
        return None
    unit = re.sub(r"[^a-z]", "", match.group(2))
    factor = _AREA_UNITS_IN_TEXT.get(unit)
    if factor is None:
        return None
    area = float(match.group(1)) * factor
    return area if area > 0 else None


def parse_road_width_ft(value: Any) -> float | None:
    """Parse road widths such as ``"30.00 ft"`` or ``"9 m"`` to feet."""
    if _is_missing(value):
        return None
    if _is_number(value):
        width = float(value)
    else:
        text = str(value).strip().lower()
        match = _NUMBER_WITH_UNIT.fullmatch(text)
        if not match:
            return None
        unit = re.sub(r"[^a-z]", "", match.group(2))
        if unit in {"", "ft", "feet", "foot"}:
            width = float(match.group(1))
        elif unit in {"m", "meter", "meters", "metre", "metres", "mtr", "mtrs"}:
            width = float(match.group(1)) * 3.28084
        else:
            return None
    if not math.isfinite(width) or not MIN_ROAD_WIDTH_FT <= width <= MAX_ROAD_WIDTH_FT:
        return None
    return round(width, 1)


def parse_open_sides(value: Any) -> int | None:
    """Parse the number of open sides (1-4).

    The scraper sometimes read a neighbouring cell instead, giving values like
    ``"Immediate"`` or ``"No"``; those become ``None``.
    """
    if _is_missing(value):
        return None
    try:
        number = float(str(value).strip()) if not _is_number(value) else float(value)
    except ValueError:
        return None
    if not number.is_integer() or not 1 <= number <= MAX_OPEN_SIDES:
        return None
    return int(number)


_FACING_KEYS = {re.sub(r"[^a-z]", "", f.lower()): f for f in FACINGS}
_FACING_KEYS.update({"n": "North", "ne": "North-East", "e": "East", "se": "South-East",
                     "s": "South", "sw": "South-West", "w": "West", "nw": "North-West"})


def normalise_facing(value: Any) -> str | None:
    """Map ``"north east"``, ``"NE"`` or ``"North-East"`` to ``"North-East"``."""
    if _is_missing(value):
        return None
    key = re.sub(r"[^a-z]", "", str(value).lower())
    key = key.removesuffix("facing")
    return _FACING_KEYS.get(key)


def _tidy(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,")


def split_location(value: Any, region: str = DEFAULT_REGION) -> tuple[str, str | None]:
    """Split a 99acres location into ``(locality, project)``.

    The location field looks like ``"Aparna Avani\\nIsnapur, Patancheru"``: an
    optional project or layout name on the first line, then the locality and
    the wider area (``region``). The locality is the most specific part of the
    last line. Listings that name only the region, or nothing, map to the
    region itself.
    """
    if _is_missing(value):
        return region, None
    lines = [_tidy(line) for line in str(value).splitlines() if _tidy(line)]
    if not lines:
        return region, None
    parts = [_tidy(part) for part in lines[-1].split(",") if _tidy(part)]
    locality = region
    if parts and parts[0].lower() != region.lower():
        locality = " ".join(w[:1].upper() + w[1:].lower() for w in parts[0].split())
    project = lines[0] if len(lines) > 1 else None
    return locality, project


# Raw header (normalised to lower-case alphanumerics) -> canonical field.
_HEADER_ALIASES = {
    "title": "title",
    "price": "price",
    "location": "location",
    "plotarea": "area",
    "area": "area",
    "facing": "facing",
    "opensides": "open_sides",
    "widthroad": "road_width",
    "roadwidth": "road_width",
}
REQUIRED_FIELDS = ("price", "area", "location")
OPTIONAL_FIELDS = ("title", "facing", "open_sides", "road_width")


def resolve_columns(columns: list[str]) -> dict[str, str]:
    """Map canonical field names to the raw table's column names."""
    mapping: dict[str, str] = {}
    for column in columns:
        field = _HEADER_ALIASES.get(re.sub(r"[^a-z]", "", str(column).lower()))
        if field and field not in mapping:
            mapping[field] = column
    missing = [f for f in REQUIRED_FIELDS if f not in mapping]
    if missing:
        raise SchemaError(
            f"Raw data is missing required column(s) {missing}; found {list(columns)}. "
            "Expected headers like Price, Plot_Area, Location, facing, open_sides, Width_road."
        )
    return mapping


@dataclass
class CleanResult:
    data: pd.DataFrame
    report: dict[str, Any]


CLEAN_COLUMNS = [
    "source_row",
    "title",
    "location",
    "locality",
    "project",
    "in_project",
    "price_inr",
    "area_sqyd",
    "price_per_sqyd",
    "facing",
    "open_sides",
    "road_width_ft",
    "group_id",
]


def build_dataset(raw: pd.DataFrame, region: str = DEFAULT_REGION) -> CleanResult:
    """Clean a raw listing table whose listings lie in ``region``.

    Rules, in order:

    1. Price and area must parse; area must be within the sanity bounds and
       price per sq yd within ₹2,000-₹2,50,000. Failing rows are dropped and
       logged with their spreadsheet row number.
    2. Unparseable optional fields (facing, open sides, road width) become
       missing; the model imputes them instead of dropping the row.
    3. Exact duplicates (same title, location, price, area and attributes) are
       dropped.
    4. Listings with the same price and area share a ``group_id``. Agents often
       cross-post one plot under several localities, so cross-validation and
       the test split keep each group on one side to avoid leakage.
    """
    columns = resolve_columns(list(raw.columns))
    records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    invalid_optional: Counter[str] = Counter()

    def get(row: pd.Series, field: str) -> Any:
        return row[columns[field]] if field in columns else None

    for position, (_, row) in enumerate(raw.iterrows()):
        source_row = position + 2  # spreadsheet row number, header is row 1
        price = parse_price_inr(get(row, "price"))
        area = parse_area_sqyd(get(row, "area"))
        reason = None
        if price is None:
            reason = f"price not parseable ({get(row, 'price')!r})"
        elif area is None:
            reason = f"plot area not parseable ({get(row, 'area')!r})"
        elif not MIN_AREA_SQYD <= area <= MAX_AREA_SQYD:
            reason = f"plot area {area:g} sq yd outside {MIN_AREA_SQYD:g}-{MAX_AREA_SQYD:g} (likely a unit error)"
        elif not MIN_RATE_INR_PER_SQYD <= price / area <= MAX_RATE_INR_PER_SQYD:
            reason = f"price per sq yd ₹{price / area:,.0f} outside plausible range (likely a unit error)"
        if reason:
            dropped.append({"source_row": source_row, "reason": reason})
            continue

        facing_raw, sides_raw, road_raw = get(row, "facing"), get(row, "open_sides"), get(row, "road_width")
        facing = normalise_facing(facing_raw)
        open_sides = parse_open_sides(sides_raw)
        road_width = parse_road_width_ft(road_raw)
        for name, raw_value, parsed in (("facing", facing_raw, facing), ("open_sides", sides_raw, open_sides),
                                        ("road_width_ft", road_raw, road_width)):
            if parsed is None and not _is_missing(raw_value):
                invalid_optional[name] += 1

        location_raw = get(row, "location")
        locality, project = split_location(location_raw, region)
        title = get(row, "title")
        records.append({
            "source_row": source_row,
            "title": _tidy(str(title)) if not _is_missing(title) else "",
            "location": _tidy(" ".join(str(location_raw).splitlines())) if not _is_missing(location_raw) else "",
            "locality": locality,
            "project": project,
            "in_project": project is not None,
            "price_inr": round(price, 2),
            "area_sqyd": round(area, 2),
            "price_per_sqyd": round(price / area, 2),
            "facing": facing,
            "open_sides": open_sides,
            "road_width_ft": road_width,
        })

    data = pd.DataFrame.from_records(records, columns=[c for c in CLEAN_COLUMNS if c != "group_id"])
    key = ["title", "location", "price_inr", "area_sqyd", "facing", "open_sides", "road_width_ft"]
    duplicate_mask = data.astype({c: "object" for c in key}).duplicated(subset=key, keep="first")
    duplicates = data.loc[duplicate_mask, "source_row"].tolist()
    data = data.loc[~duplicate_mask].reset_index(drop=True)
    data["open_sides"] = data["open_sides"].astype("Int64")
    data["road_width_ft"] = data["road_width_ft"].astype("Float64")
    data["group_id"] = data.groupby(["price_inr", "area_sqyd"], sort=True).ngroup()

    report = {
        "region": region,
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(data)),
        "dropped_invalid": dropped,
        "dropped_duplicate_source_rows": [int(r) for r in duplicates],
        "unparseable_optional_values": dict(invalid_optional),
        "missing_after_cleaning": {
            column: int(data[column].isna().sum()) for column in ("facing", "open_sides", "road_width_ft")
        },
        "listing_groups": int(data["group_id"].nunique()),
        "cross_posted_groups": int((data.groupby("group_id").size() > 1).sum()),
        "localities": {k: int(v) for k, v in data["locality"].value_counts().items()},
    }
    return CleanResult(data=data, report=report)


def load_raw(path) -> pd.DataFrame:
    """Read a raw listing export (``.xlsx``/``.xls`` or ``.csv``)."""
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file {path!r}; use .xlsx or .csv")
