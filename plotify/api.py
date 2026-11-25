"""JSON API, version 1."""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

from .service import QueryError
from .web import check_rate_limit, service

api = Blueprint("api", __name__, url_prefix="/api/v1")


@api.get("/health")
def health():
    return jsonify(status="ok", model_version=service().metadata["model_version"])


@api.get("/meta")
def meta():
    svc = service()
    return jsonify(model=svc.metadata, options=svc.options(), interval_level=svc.interval["level"])


@api.post("/predict")
def predict():
    check_rate_limit()
    if not request.is_json:
        raise UnsupportedMediaType("Send a JSON object with Content-Type: application/json.")
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequest("The request body must be a JSON object.")
    svc = service()
    try:
        query = svc.parse(data, strict=True)
    except QueryError as exc:
        return jsonify(error={"code": "validation_error", "message": "Some fields are invalid.",
                              "fields": exc.errors}), 400
    payload = svc.estimate(query).to_dict()
    payload["model"] = svc.metadata
    return jsonify(payload)
