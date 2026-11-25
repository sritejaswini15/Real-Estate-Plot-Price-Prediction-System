import json

import pytest

from plotify import create_app
from plotify.config import Settings
from plotify.security import CONTENT_SECURITY_POLICY

VALID = {"area": 200, "area_unit": "sqyd", "locality": "Isnapur", "facing": "East", "open_sides": 2,
         "road_width_ft": 30, "in_project": True}


@pytest.mark.parametrize("path", ["/", "/market", "/model", "/api"])
def test_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.mimetype == "text/html"
    body = response.get_data(as_text=True)
    assert "<main" in body and "Plotify" in body


def test_estimate_page(client):
    response = client.get("/?area=200&area_unit=sqyd&locality=Isnapur&facing=East&road_width_ft=30")
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Estimated asking price" in body and "₹" in body
    assert "Most similar listings" in body
    assert 'class="viz"' in body  # the strip chart rendered


def test_estimate_page_shows_field_errors(client):
    response = client.get("/?area=abc&open_sides=9")
    body = response.get_data(as_text=True)
    assert response.status_code == 400
    assert 'aria-invalid="true"' in body
    assert "Plot area must be a positive number." in body
    assert "Open sides must be a whole number" in body


def test_form_keeps_defaults_for_absent_fields(client):
    body = client.get("/?area=abc").get_data(as_text=True)
    assert '<option value="Patancheru" selected>' in body
    assert '<option value="sqyd" selected>' in body


def test_user_input_is_escaped(client):
    response = client.get("/?area=<script>alert(1)</script>&locality=<b>x</b>")
    body = response.get_data(as_text=True)
    assert response.status_code == 400
    assert "<script>alert" not in body and "<b>x</b>" not in body


def test_security_headers(client):
    response = client.get("/")
    assert response.headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
    assert "script-src" not in CONTENT_SECURITY_POLICY  # default-src 'none' blocks all scripts
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "same-origin"
    assert "Strict-Transport-Security" not in response.headers
    assert len(response.headers["X-Request-ID"]) == 32


def test_request_id_is_echoed_when_reasonable(client):
    assert client.get("/", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"
    assert client.get("/", headers={"X-Request-ID": "x" * 200}).headers["X-Request-ID"] != "x" * 200


def test_static_assets(client):
    css = client.get("/static/app.css")
    assert css.status_code == 200 and css.mimetype == "text/css"
    assert client.get("/static/favicon.svg").status_code == 200


def test_html_404(client):
    response = client.get("/no-such-page")
    assert response.status_code == 404 and response.mimetype == "text/html"
    assert "Back to the estimator" in response.get_data(as_text=True)


def test_api_predict(client):
    response = client.post("/api/v1/predict", json=VALID)
    assert response.status_code == 200
    data = response.get_json()
    est = data["estimate"]
    assert est["low_inr"] < est["price_inr"] < est["high_inr"]
    assert data["input"]["area_sqyd"] == 200
    assert data["model"]["model_version"]
    assert response.headers["Cache-Control"] == "no-store"


def test_api_validation_errors(client):
    response = client.post("/api/v1/predict", json={"area": -1, "road_width": 30})
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert error["code"] == "validation_error"
    assert set(error["fields"]) == {"area", "road_width"}


@pytest.mark.parametrize("kwargs, status", [
    ({"data": "area=200", "content_type": "application/x-www-form-urlencoded"}, 415),
    ({"data": "[1, 2]", "content_type": "application/json"}, 400),
    ({"data": "{not json", "content_type": "application/json"}, 400),
    ({"data": json.dumps({"area": 200, "pad": "x" * 20_000}), "content_type": "application/json"}, 413),
])
def test_api_rejects_bad_bodies(client, kwargs, status):
    response = client.post("/api/v1/predict", **kwargs)
    assert response.status_code == status
    assert response.mimetype == "application/json"
    assert "message" in response.get_json()["error"]


def test_api_method_and_route_errors_are_json(client):
    response = client.get("/api/v1/predict")
    assert response.status_code == 405 and "POST" in response.headers["Allow"]
    assert response.get_json()["error"]["code"] == "method_not_allowed"
    missing = client.get("/api/v1/nothing")
    assert missing.status_code == 404 and missing.get_json()["error"]["code"] == "not_found"


def test_api_health_and_meta(client, service):
    health = client.get("/api/v1/health").get_json()
    assert health == {"status": "ok", "model_version": service.metadata["model_version"]}
    meta = client.get("/api/v1/meta").get_json()
    assert meta["options"]["localities"] == service.schema["localities"]
    assert meta["options"]["model_inputs"] == service.model_inputs


def test_rate_limit(trained, service):
    app = create_app(Settings(model_path=trained["model_path"], rate_limit_per_minute=2), service=service)
    client = app.test_client()
    assert client.post("/api/v1/predict", json={"area": 200}).status_code == 200
    assert client.get("/?area=200").status_code == 200
    limited = client.post("/api/v1/predict", json={"area": 200})
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    assert limited.get_json()["error"]["code"] == "too_many_requests"
    assert client.get("/?area=200").status_code == 429
    assert client.get("/").status_code == 200  # browsing without estimating is not limited


def test_unhandled_errors_do_not_leak_details(app, service, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(service, "estimate", boom)
    client = app.test_client()
    response = client.post("/api/v1/predict", json={"area": 200})
    assert response.status_code == 500
    assert "secret" not in response.get_data(as_text=True)
    html = client.get("/?area=200")
    assert html.status_code == 500 and "secret" not in html.get_data(as_text=True)


def test_another_regions_dataset_end_to_end(tmp_path):
    """Listings for a different area train and serve with no trace of the default region."""
    from plotify.bundle import load_bundle
    from plotify.service import PlotPriceService
    from plotify.training import MODEL_FILENAME, TrainConfig, train

    from .conftest import make_raw

    raw_path = tmp_path / "shamshabad.csv"
    make_raw(region="Shamshabad").to_csv(raw_path, index=False)
    train(TrainConfig(data_path=raw_path, region="Shamshabad", model_dir=tmp_path / "models",
                      report_dir=tmp_path / "reports", processed_path=None, n_repeats=1,
                      candidates=("median_rate", "locality_median_rate"), log=lambda *_: None))
    svc = PlotPriceService(load_bundle(tmp_path / "models" / MODEL_FILENAME))
    assert svc.region == "Shamshabad"
    assert "Shamshabad" in svc.schema["localities"] and "Patancheru" not in svc.schema["localities"]

    client = create_app(Settings(model_path=tmp_path / "models" / MODEL_FILENAME), service=svc).test_client()
    for path in ["/", "/market", "/model", "/api", "/?area=200"]:
        body = client.get(path).get_data(as_text=True)
        assert "Patancheru" not in body, path
    assert "What should a plot in Shamshabad cost?" in client.get("/").get_data(as_text=True)
    assert '<option value="Shamshabad" selected>' in client.get("/").get_data(as_text=True)
    assert client.post("/api/v1/predict", json={"area": 200}).get_json()["input"]["locality"] == "Shamshabad"


def test_hsts_and_proxy_settings(trained, service):
    app = create_app(Settings(model_path=trained["model_path"], hsts=True, trusted_proxies=1), service=service)
    response = app.test_client().get("/", headers={"X-Forwarded-For": "203.0.113.9"})
    assert response.headers["Strict-Transport-Security"].startswith("max-age=")
