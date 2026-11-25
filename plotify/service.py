"""Input validation and price estimation shared by the web UI, API and CLI."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cleaning import normalise_facing
from .domain import (
    AREA_UNITS,
    DEFAULT_REGION,
    MAX_AREA_SQYD,
    MAX_OPEN_SIDES,
    MAX_ROAD_WIDTH_FT,
    MIN_AREA_SQYD,
    MIN_ROAD_WIDTH_FT,
    OTHER_LOCALITY,
)
from .domain import FACINGS
from .formatting import INPUT_LABELS, describe_inputs, format_number
from .market import find_comparables

FIELDS = ("area", "area_unit", "locality", "facing", "open_sides", "road_width_ft", "in_project")
_UNKNOWN_WORDS = {"", "unknown", "not sure", "notsure", "any", "none", "null"}
_TRUE_WORDS = {"1", "true", "yes", "on", "y"}
_FALSE_WORDS = {"0", "false", "no", "off", "n", ""}


class QueryError(ValueError):
    """One or more input fields are invalid; ``errors`` maps field -> message."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


@dataclass(frozen=True)
class PlotQuery:
    area: float
    area_unit: str
    area_sqyd: float
    locality: str
    facing: str | None
    open_sides: int | None
    road_width_ft: float | None
    in_project: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in _UNKNOWN_WORDS)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value).strip().replace(",", ""))
        except ValueError:
            return None
    return number if math.isfinite(number) else None


def parse_query(data: Mapping[str, Any], *, localities: list[str], rare_localities: list[str] = (),
                default_locality: str = DEFAULT_REGION, strict: bool = False) -> PlotQuery:
    """Validate raw input (JSON or query-string values) into a :class:`PlotQuery`.

    With ``strict`` (the JSON API), unknown fields are rejected. A misspelt
    field name must not be silently ignored: the original project's predictor
    did exactly that with ``land_area`` and ``road_width``.
    """
    errors: dict[str, str] = {}
    if strict:
        for key in data:
            if key not in FIELDS:
                errors[str(key)[:40]] = f"Unknown field. Allowed fields: {', '.join(FIELDS)}."

    unit = "sqyd" if _blank(data.get("area_unit")) else str(data.get("area_unit")).strip().lower()
    if unit not in AREA_UNITS:
        errors["area_unit"] = f"Choose one of: {', '.join(AREA_UNITS)}."

    area = area_sqyd = None
    if _blank(data.get("area")):
        errors["area"] = "Enter the plot area."
    else:
        area = _number(data.get("area"))
        if area is None or area <= 0:
            errors["area"] = "Plot area must be a positive number."
        elif unit in AREA_UNITS:
            area_sqyd = area * AREA_UNITS[unit][1]
            if not MIN_AREA_SQYD <= area_sqyd <= MAX_AREA_SQYD:
                errors["area"] = (f"Plot area must be between {format_number(MIN_AREA_SQYD)} and "
                                  f"{format_number(MAX_AREA_SQYD)} sq yd (you entered "
                                  f"{format_number(area_sqyd, 1)} sq yd).")

    known = {name.lower(): name for name in [*localities, *rare_localities, OTHER_LOCALITY]}
    raw_locality = data.get("locality")
    locality = default_locality
    if not _blank(raw_locality):
        locality = known.get(str(raw_locality).strip().lower())
        if locality is None:
            errors["locality"] = f"Unknown locality. Choose one of: {', '.join(localities)}, or {OTHER_LOCALITY}."

    facing = None
    if not _blank(data.get("facing")):
        facing = normalise_facing(data.get("facing"))
        if facing is None:
            errors["facing"] = "Facing must be a compass direction such as East or North-East, or left blank."

    open_sides = None
    if not _blank(data.get("open_sides")):
        number = _number(data.get("open_sides"))
        if number is None or not number.is_integer() or not 1 <= number <= MAX_OPEN_SIDES:
            errors["open_sides"] = f"Open sides must be a whole number from 1 to {MAX_OPEN_SIDES}, or left blank."
        else:
            open_sides = int(number)

    road_width = None
    if not _blank(data.get("road_width_ft")):
        road_width = _number(data.get("road_width_ft"))
        if road_width is None or not MIN_ROAD_WIDTH_FT <= road_width <= MAX_ROAD_WIDTH_FT:
            errors["road_width_ft"] = (f"Road width must be between {MIN_ROAD_WIDTH_FT:g} and "
                                       f"{MAX_ROAD_WIDTH_FT:g} ft, or left blank.")
            road_width = None

    raw_project = data.get("in_project")
    in_project = False
    if isinstance(raw_project, bool):
        in_project = raw_project
    elif raw_project is not None:
        word = str(raw_project).strip().lower()
        if word in _TRUE_WORDS:
            in_project = True
        elif word not in _FALSE_WORDS:
            errors["in_project"] = "in_project must be true or false."

    if errors:
        raise QueryError(errors)
    return PlotQuery(area=float(area), area_unit=unit, area_sqyd=float(area_sqyd), locality=locality,
                     facing=facing, open_sides=open_sides, road_width_ft=road_width, in_project=in_project)


@dataclass
class Estimate:
    query: PlotQuery
    price_inr: float
    low_inr: float
    high_inr: float
    interval_level: float
    price_per_sqyd: float
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    comparables: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": {
                "price_inr": self.price_inr,
                "low_inr": self.low_inr,
                "high_inr": self.high_inr,
                "interval_level": self.interval_level,
                "price_per_sqyd_inr": self.price_per_sqyd,
            },
            "input": self.query.to_dict(),
            "warnings": self.warnings,
            "notes": self.notes,
            "comparables": self.comparables,
        }


def _round_to(value: float, step: float) -> float:
    return float(round(value / step) * step)


class PlotPriceService:
    """Wraps a loaded model bundle; safe to share across request threads."""

    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle
        self.estimator = bundle["estimator"]
        self.schema = bundle["schema"]
        self.interval = bundle["interval"]
        self.model_inputs: list[str] = bundle["model_inputs"]
        self.listings = pd.DataFrame(bundle["listings"])
        self.region: str = self.schema.get("region", DEFAULT_REGION)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": self.bundle["metadata"]["model_version"],
            "trained_at": self.bundle["metadata"]["trained_at"],
            "model": self.bundle["model_name"],
            "model_label": self.bundle["model_label"],
            "region": self.region,
            "listings": self.bundle["metadata"]["rows"],
        }

    def options(self) -> dict[str, Any]:
        return {
            "area_units": {key: label for key, (label, _) in AREA_UNITS.items()},
            "region": self.region,
            "default_locality": self.region,
            "localities": self.schema["localities"],
            "rare_localities": self.schema["rare_localities"],
            "other_locality": OTHER_LOCALITY,
            "facings": list(FACINGS),
            "open_sides": list(range(1, MAX_OPEN_SIDES + 1)),
            "limits": {
                "area_sqyd": [MIN_AREA_SQYD, MAX_AREA_SQYD],
                "road_width_ft": [MIN_ROAD_WIDTH_FT, MAX_ROAD_WIDTH_FT],
            },
            "model_inputs": self.model_inputs,
        }

    def parse(self, data: Mapping[str, Any], *, strict: bool = False) -> PlotQuery:
        return parse_query(data, localities=self.schema["localities"],
                           rare_localities=self.schema["rare_localities"], default_locality=self.region,
                           strict=strict)

    def estimate(self, query: PlotQuery, *, comparables: int = 5) -> Estimate:
        row = pd.DataFrame([{
            "area_sqyd": query.area_sqyd,
            "locality": query.locality,
            "facing": query.facing,
            "open_sides": np.nan if query.open_sides is None else float(query.open_sides),
            "road_width_ft": np.nan if query.road_width_ft is None else float(query.road_width_ft),
            "in_project": query.in_project,
        }])
        price = float(self.estimator.predict(row)[0])
        low = price * math.exp(self.interval["log_low"])
        high = price * math.exp(self.interval["log_high"])
        return Estimate(
            query=query,
            price_inr=_round_to(price, 1000),
            low_inr=_round_to(low, 1000),
            high_inr=_round_to(high, 1000),
            interval_level=float(self.interval["level"]),
            price_per_sqyd=_round_to(price / query.area_sqyd, 10),
            warnings=self._warnings(query),
            notes=self._notes(query),
            comparables=find_comparables(self.listings, area_sqyd=query.area_sqyd, locality=query.locality,
                                         facing=query.facing, road_width_ft=query.road_width_ft,
                                         limit=comparables),
        )

    def _warnings(self, query: PlotQuery) -> list[str]:
        warnings = []
        area = self.schema["area_sqyd"]
        span = f"{format_number(area['min'])}–{format_number(area['max'])} sq yd"
        if query.area_sqyd < area["min"] or query.area_sqyd > area["max"]:
            size = "small" if query.area_sqyd < area["min"] else "large"
            warnings.append(f"No listing in the data is this {size} (listings span {span}), so this "
                            "estimate is an extrapolation.")
        elif query.area_sqyd < area["p05"] or query.area_sqyd > area["p95"]:
            size = "small" if query.area_sqyd < area["p05"] else "large"
            warnings.append(f"Very few listings are this {size}; treat the estimate as rough.")
        road = self.schema.get("road_width_ft")
        if ("road_width_ft" in self.model_inputs and road and query.road_width_ft is not None
                and not road["min"] <= query.road_width_ft <= road["max"]):
            warnings.append(f"Listed road widths span {road['min']:g}–{road['max']:g} ft; the model has "
                            "not seen this width.")
        if query.locality == OTHER_LOCALITY or query.locality in self.schema["rare_localities"]:
            place = "this locality" if query.locality == OTHER_LOCALITY else query.locality
            warnings.append(f"There are few or no listings from {place}, so the estimate uses the pooled "
                            "rate for less common localities.")
        missing = [INPUT_LABELS[name] for name in ("facing", "open_sides", "road_width_ft")
                   if name in self.model_inputs and getattr(query, name) is None]
        if missing:
            warnings.append(f"Not given: {', '.join(missing)}. The model fills these in the way it "
                            "filled missing values in training.")
        return warnings

    def _notes(self, query: PlotQuery) -> list[str]:
        unused = [name for name in ("facing", "open_sides", "road_width_ft", "in_project")
                  if name not in self.model_inputs]
        if not unused:
            return []
        return [
            f"The estimate uses {describe_inputs(self.model_inputs)} only. In cross-validation, adding "
            f"{describe_inputs(unused)} did not measurably improve accuracy on these listings, so those "
            "inputs only choose which comparable listings are shown."
        ]
