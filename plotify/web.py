"""HTML pages: estimator, market overview, model report and API docs."""

from __future__ import annotations

import json

from flask import Blueprint, current_app, render_template, request
from werkzeug.exceptions import TooManyRequests

from .charts import locality_rate_bars, oof_scatter, rate_strip
from .service import FIELDS, PlotPriceService, QueryError

web = Blueprint("web", __name__)


def service() -> PlotPriceService:
    return current_app.extensions["plotify"]["service"]


def check_rate_limit() -> None:
    retry = current_app.extensions["plotify"]["limiter"].hit(request.remote_addr or "unknown")
    if retry is not None:
        raise TooManyRequests("Too many estimates from your address. Please wait a minute and try again.",
                              retry_after=max(int(retry) + 1, 1))


@web.get("/")
def index():
    svc = service()
    form = {"area": "", "area_unit": "sqyd", "locality": svc.region, "facing": "", "open_sides": "",
            "road_width_ft": "", "in_project": ""}
    submitted = any(name in request.args for name in FIELDS)
    if submitted:
        # Absent fields keep their defaults; an unticked checkbox is simply absent.
        form.update({name: request.args[name][:40] for name in FIELDS if name in request.args})
    estimate, errors, chart, status = None, {}, None, 200
    if submitted:
        check_rate_limit()
        try:
            estimate = svc.estimate(svc.parse(request.args))
        except QueryError as exc:
            errors, status = exc.errors, 400
        else:
            area = estimate.query.area_sqyd
            chart = rate_strip(estimate.price_per_sqyd, estimate.low_inr / area, estimate.high_inr / area,
                               estimate.comparables, estimate.interval_level)
    return render_template("index.html", form=form, options=svc.options(), estimate=estimate, errors=errors,
                           chart=chart, interval=svc.interval), status


@web.get("/market")
def market():
    svc = service()
    summary = svc.bundle["market"]
    return render_template("market.html", market=summary, chart=locality_rate_bars(summary["localities"]),
                           quality=svc.bundle["data_quality"])


@web.get("/model")
def model():
    svc = service()
    bundle = svc.bundle
    return render_template("model.html", bundle=bundle, metrics=bundle["metrics"], interval=svc.interval,
                           chart=oof_scatter(bundle["oof_points"], svc.interval), quality=bundle["data_quality"])


EXAMPLE_REQUEST = {"area": 200, "area_unit": "sqyd", "locality": "Isnapur", "facing": "East", "road_width_ft": 30}


@web.get("/api")
def api_docs():
    svc = service()
    example_request = dict(EXAMPLE_REQUEST)
    if example_request["locality"] not in svc.schema["localities"]:
        example_request["locality"] = svc.schema["localities"][0] if svc.schema["localities"] else "Other"
    response = svc.estimate(svc.parse(example_request, strict=True), comparables=1).to_dict()
    response["model"] = svc.metadata
    return render_template("api.html", options=svc.options(),
                           example_request=json.dumps(example_request),
                           example_response=json.dumps(response, indent=2, ensure_ascii=False),
                           rate_limit=current_app.extensions["plotify"]["settings"].rate_limit_per_minute)
