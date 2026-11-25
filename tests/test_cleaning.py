import math

import pandas as pd
import pytest

from plotify.cleaning import (
    SchemaError,
    build_dataset,
    load_raw,
    normalise_facing,
    parse_area_sqyd,
    parse_open_sides,
    parse_price_inr,
    parse_road_width_ft,
    resolve_columns,
    split_location,
)
from plotify.training import DEFAULT_DATA

from .conftest import make_raw


@pytest.mark.parametrize("raw, expected", [
    ("85 Lac", 8_500_000), ("1.2 Crore", 12_000_000), ("₹ 61.56 Lac", 6_156_000), ("45,00,000", 4_500_000),
    (8_500_000, 8_500_000), ("1.5 Cr", 15_000_000), ("50 lakhs", 5_000_000), ("2 crores", 20_000_000),
    ("Rs. 30 Lac", 3_000_000), ("105 Crore", 1_050_000_000),
    ("N/A", None), ("45 - 50 Lac", None), ("Price on request", None), (0, None), (-5, None), (None, None),
    (float("nan"), None), (True, None), ("12 bananas", None),
])
def test_parse_price(raw, expected):
    result = parse_price_inr(raw)
    assert result == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize("raw, expected", [
    (200, 200), ("1800 sq.ft.", 200), ("2 acres", 9680), ("1 guntha", 121), ("100 sq m", 119.599),
    ("300 sq. yards", 300), ("abc", None), (0, None), ("", None),
])
def test_parse_area(raw, expected):
    result = parse_area_sqyd(raw)
    assert result == (pytest.approx(expected, rel=1e-4) if expected is not None else None)


@pytest.mark.parametrize("raw, expected", [
    ("30.00 ft", 30.0), ("100.0 ft", 100.0), (40, 40.0), ("9 m", 29.5), ("2 ft", None), ("Immediate", None),
    (None, None), ("400 ft", None),
])
def test_parse_road_width(raw, expected):
    assert parse_road_width_ft(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    (2, 2), ("2", 2), (2.0, 2), (4, 4), ("Immediate", None), ("No", None), (5, None), (1.5, None), (0, None),
    (None, None), (float("nan"), None),
])
def test_parse_open_sides(raw, expected):
    assert parse_open_sides(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("East", "East"), ("north east", "North-East"), ("NE", "North-East"), ("North-West", "North-West"),
    ("east facing", "East"), ("SOUTH-WEST", "South-West"), ("Sideways", None), (None, None), ("", None),
])
def test_normalise_facing(raw, expected):
    assert normalise_facing(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Aparna Avani\nIsnapur, Patancheru", ("Isnapur", "Aparna Avani")),
    ("Isnapur, Patancheru", ("Isnapur", None)),
    ("Patancheru", ("Patancheru", None)),
    ("CK Golden Doors\nPatancheru", ("Patancheru", "CK Golden Doors")),
    ("abhinandan avenue, Chitkul\nIsnapur, Patancheru", ("Isnapur", "abhinandan avenue, Chitkul")),
    ("Alwin Colony,Patancheruvu", ("Alwin Colony", None)),
    ("Chitkul", ("Chitkul", None)),
    ("sreeram nagar colony, Patancheru", ("Sreeram Nagar Colony", None)),
    ("Patancheru, Hyderabad", ("Patancheru", None)),
    (None, ("Patancheru", None)),
])
def test_split_location(raw, expected):
    assert split_location(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Kothur, Shamshabad", ("Kothur", None)),
    ("Shamshabad, Hyderabad", ("Shamshabad", None)),
    ("shamshabad", ("Shamshabad", None)),
    ("Green Meadows\nKothur, Shamshabad", ("Kothur", "Green Meadows")),
    (None, ("Shamshabad", None)),
])
def test_split_location_for_another_region(raw, expected):
    assert split_location(raw, region="Shamshabad") == expected


def test_resolve_columns_accepts_scraper_and_export_headers():
    export = resolve_columns(["Title", "Price", "Location", "Plot_Area", "facing", "open_sides", "Width_road"])
    scraper = resolve_columns(["Title", "Price", "Location", "Plot Area", "Facing", "Open Sides"])
    assert export["area"] == "Plot_Area" and export["road_width"] == "Width_road"
    assert scraper["area"] == "Plot Area" and scraper["open_sides"] == "Open Sides"
    assert "road_width" not in scraper


def test_resolve_columns_reports_missing_required():
    with pytest.raises(SchemaError, match="price"):
        resolve_columns(["Title", "Location", "Plot_Area"])


def test_build_dataset_rules():
    raw = make_raw()
    result = build_dataset(raw)
    data, report = result.data, result.report

    # The 22.11 sq yd unit error is dropped with its spreadsheet row number and reason.
    assert [d["source_row"] for d in report["dropped_invalid"]] == [len(raw)]
    assert "unit error" in report["dropped_invalid"][0]["reason"]
    # The exact duplicate (last row) is removed; its twin is kept.
    assert report["dropped_duplicate_source_rows"] == [len(raw) + 1]
    assert report["clean_rows"] == len(data) == len(raw) - 2
    # "Immediate" in open sides is a scraper misread: missing, and counted.
    assert report["unparseable_optional_values"]["open_sides"] > 0
    assert data["open_sides"].dropna().isin([1, 2]).all()
    # Rows with a project line are flagged; the Chitkul cross-post shares its twin's group.
    assert data.loc[data["project"].notna(), "in_project"].all()
    chitkul = data[data["locality"] == "Chitkul"].iloc[0]
    twins = data[(data["price_inr"] == chitkul["price_inr"]) & (data["area_sqyd"] == chitkul["area_sqyd"])]
    assert len(twins) >= 2 and twins["group_id"].nunique() == 1
    assert (data["price_per_sqyd"] - data["price_inr"] / data["area_sqyd"]).abs().max() < 0.01


def test_build_dataset_keeps_rows_with_missing_optional_fields():
    raw = pd.DataFrame({"Price": ["50 Lac", "60 Lac"], "Plot_Area": [200, 150], "Location": ["Isnapur, Patancheru", None]})
    data = build_dataset(raw).data
    assert len(data) == 2
    assert data["facing"].isna().all() and data["road_width_ft"].isna().all()
    assert data.loc[1, "locality"] == "Patancheru"


@pytest.mark.skipif(not DEFAULT_DATA.is_file(), reason="real data file not present")
def test_real_dataset_contract():
    """Pins the cleaning outcome on the shipped 99acres export (see README)."""
    result = build_dataset(load_raw(DEFAULT_DATA))
    report = result.report
    assert report["raw_rows"] == 100
    assert report["clean_rows"] == 94
    assert [d["source_row"] for d in report["dropped_invalid"]] == [68]
    assert len(report["dropped_duplicate_source_rows"]) == 5
    assert report["cross_posted_groups"] == 7
    assert math.isclose(result.data["price_inr"].min(), 1_400_000)
