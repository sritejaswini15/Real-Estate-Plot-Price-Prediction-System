"""Domain constants shared by cleaning, modelling, the API and the web UI."""

from __future__ import annotations

FACINGS: tuple[str, ...] = (
    "North",
    "North-East",
    "East",
    "South-East",
    "South",
    "South-West",
    "West",
    "North-West",
)

# Area units a user may enter, as (label, square yards per unit).
# The listings themselves are recorded in square yards, the usual unit for
# plots in Hyderabad.
AREA_UNITS: dict[str, tuple[str, float]] = {
    "sqyd": ("sq yd", 1.0),
    "sqft": ("sq ft", 1.0 / 9.0),
    "sqm": ("sq m", 1.0 / 0.83612736),
    "guntha": ("guntha", 121.0),
    "acre": ("acre", 4840.0),
}

# The area the bundled dataset covers. Listings that name no sub-locality
# fall under it. For another area's data, train with `--region "Name"`.
# The name is saved in the model bundle, and the UI reads it from there.
DEFAULT_REGION = "Patancheru"
OTHER_LOCALITY = "Other"

# Sanity bounds for cleaning and for input validation. They are deliberately
# generous: they reject unit mix-ups (e.g. a "22.11 sq yd" villa plot priced
# at ₹80 lakh) and typos, not unusual but genuine listings.
MIN_AREA_SQYD = 50.0
MAX_AREA_SQYD = 100_000.0
MIN_RATE_INR_PER_SQYD = 2_000.0
MAX_RATE_INR_PER_SQYD = 250_000.0
MIN_ROAD_WIDTH_FT = 5.0
MAX_ROAD_WIDTH_FT = 300.0
MAX_OPEN_SIDES = 4

# Localities with fewer listings than this are pooled as "infrequent" by the
# model and are not offered as their own choice in the UI.
MIN_LOCALITY_COUNT = 3
