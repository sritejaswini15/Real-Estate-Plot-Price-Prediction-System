import json
import shutil

import pytest

from plotify.bundle import BundleError, load_bundle, manifest_path
from plotify.domain import OTHER_LOCALITY
from plotify.formatting import describe_inputs, format_inr, format_inr_short, format_multiplier, format_number
from plotify.service import QueryError, parse_query

LOCALITIES = ["Isnapur", "Muthangi", "Patancheru"]


def parse(data, **kw):
    return parse_query(data, localities=LOCALITIES, rare_localities=["Chitkul"], **kw)


def test_parse_minimal_query_uses_defaults():
    q = parse({"area": "200"})
    assert (q.area_sqyd, q.area_unit, q.locality, q.facing, q.open_sides, q.road_width_ft, q.in_project) == \
        (200.0, "sqyd", "Patancheru", None, None, None, False)


@pytest.mark.parametrize("area, unit, sqyd", [("1800", "sqft", 200.0), (1, "guntha", 121.0),
                                              ("0.5", "acre", 2420.0), ("1,200", "sqyd", 1200.0)])
def test_parse_area_units(area, unit, sqyd):
    assert parse({"area": area, "area_unit": unit}).area_sqyd == pytest.approx(sqyd)


def test_parse_normalises_optional_fields():
    q = parse({"area": 150, "locality": "isnapur", "facing": "north east", "open_sides": "2",
               "road_width_ft": "30", "in_project": "on"})
    assert (q.locality, q.facing, q.open_sides, q.road_width_ft, q.in_project) == ("Isnapur", "North-East", 2, 30.0, True)
    assert parse({"area": 150, "locality": "Chitkul"}).locality == "Chitkul"
    assert parse({"area": 150, "locality": "other"}).locality == OTHER_LOCALITY
    assert parse({"area": 150, "facing": "", "open_sides": "", "road_width_ft": ""}).facing is None


@pytest.mark.parametrize("data, field", [
    ({}, "area"), ({"area": "abc"}, "area"), ({"area": -3}, "area"), ({"area": True}, "area"),
    ({"area": 10}, "area"), ({"area": 200, "area_unit": "hectare"}, "area_unit"),
    ({"area": 200, "locality": "Atlantis"}, "locality"), ({"area": 200, "facing": "Up"}, "facing"),
    ({"area": 200, "open_sides": 7}, "open_sides"), ({"area": 200, "open_sides": 1.5}, "open_sides"),
    ({"area": 200, "road_width_ft": 1000}, "road_width_ft"), ({"area": 200, "in_project": "maybe"}, "in_project"),
    ({"area": float("inf")}, "area"),
])
def test_parse_rejects_bad_values(data, field):
    with pytest.raises(QueryError) as info:
        parse(data)
    assert field in info.value.errors


def test_strict_mode_rejects_unknown_fields():
    # The original predictor silently ignored misnamed inputs such as "road_width".
    with pytest.raises(QueryError) as info:
        parse({"area": 200, "road_width": 30}, strict=True)
    assert "road_width" in info.value.errors
    assert parse({"area": 200, "utm_source": "x"}).area_sqyd == 200  # lenient for web forms


def test_estimate_is_consistent(service):
    small = service.estimate(service.parse({"area": 150, "locality": "Isnapur"}))
    large = service.estimate(service.parse({"area": 450, "locality": "Isnapur"}))
    assert 0 < small.price_inr < large.price_inr
    for est in (small, large):
        assert est.low_inr < est.price_inr < est.high_inr
        assert est.price_per_sqyd == pytest.approx(est.price_inr / est.query.area_sqyd, rel=0.01)
        assert est.price_inr % 1000 == 0
        assert 0 < len(est.comparables) <= 5
        groups = [c["source_row"] for c in est.comparables]
        assert len(groups) == len(set(groups))


def test_estimate_reflects_locality_rates(service):
    """The synthetic data has Patighanpur at 45k/sq yd and Patancheru at 30k/sq yd."""
    high = service.estimate(service.parse({"area": 200, "locality": "Patighanpur"}))
    low = service.estimate(service.parse({"area": 200, "locality": "Patancheru"}))
    assert high.price_per_sqyd > low.price_per_sqyd * 1.2


def test_estimate_warnings(service):
    huge = service.estimate(service.parse({"area": 50_000}))
    assert any("extrapolation" in w for w in huge.warnings)
    other = service.estimate(service.parse({"area": 200, "locality": OTHER_LOCALITY}))
    assert any("pooled" in w for w in other.warnings)
    typical = service.estimate(service.parse({"area": 200, "locality": "Isnapur", "facing": "East",
                                              "open_sides": 2, "road_width_ft": 40}))
    assert typical.warnings == []


def test_notes_explain_unused_inputs(service):
    est = service.estimate(service.parse({"area": 200}))
    if "facing" in service.model_inputs:
        assert est.notes == []
    else:
        assert "did not measurably improve accuracy" in est.notes[0]


def test_estimate_to_dict_is_json_serialisable(service):
    payload = service.estimate(service.parse({"area": 200, "facing": "East"})).to_dict()
    json.dumps(payload)
    assert set(payload) == {"estimate", "input", "warnings", "notes", "comparables"}


def test_bundle_integrity_checks(trained, tmp_path):
    path = tmp_path / "model.joblib"
    shutil.copy(trained["model_path"], path)
    with pytest.raises(BundleError, match="Manifest"):
        load_bundle(path)
    shutil.copy(manifest_path(trained["model_path"]), manifest_path(path))
    assert load_bundle(path)["model_name"]
    with open(path, "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(BundleError, match="checksum"):
        load_bundle(path)
    with pytest.raises(BundleError, match="python -m plotify train"):
        load_bundle(tmp_path / "missing.joblib")


@pytest.mark.parametrize("amount, full, short", [
    (5_840_000, "₹58,40,000", "₹58.4 lakh"), (10_500_000, "₹1,05,00,000", "₹1.05 crore"),
    (1_050_000_000, "₹1,05,00,00,000", "₹105 crore"), (38_000, "₹38,000", "₹38,000"),
    (999, "₹999", "₹999"), (99_999_000, "₹9,99,99,000", "₹10 crore"), (-1500, "−₹1,500", "−₹1,500"),
])
def test_rupee_formatting(amount, full, short):
    assert format_inr(amount) == full
    assert format_inr_short(amount) == short


def test_other_formatting():
    assert format_number(1234567.891, 2) == "12,34,567.89"
    assert format_inr(None) == "—"
    assert describe_inputs(["area_sqyd", "locality"]) == "plot area and locality"
    assert describe_inputs(["area_sqyd", "locality", "facing"]) == "plot area, locality and facing"
    assert format_multiplier(0.0) == "+0%"
    assert format_multiplier(-0.6931) == "−50%"
