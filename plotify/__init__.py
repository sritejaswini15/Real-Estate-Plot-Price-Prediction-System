"""Plotify: asking-price estimates for residential plots, learned from scraped listings.

Serve with ``python -m plotify serve``, or point a WSGI server at
``plotify:create_app`` (e.g. ``python -m waitress --call plotify:create_app``).
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, InternalServerError
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import Settings
from .formatting import (
    describe_inputs,
    format_area,
    format_inr,
    format_inr_short,
    format_multiplier,
    format_number,
    format_pct,
)
from .security import RateLimiter, apply_security_headers
from .service import PlotPriceService

__version__ = "1.0.0"
__all__ = ["create_app", "__version__"]

log = logging.getLogger("plotify")


def _configure_logging(level: str) -> None:
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(handler)
        log.propagate = False
    log.setLevel(level)


def create_app(settings: Settings | None = None, *, service: PlotPriceService | None = None) -> Flask:
    settings = settings or Settings.from_env()
    _configure_logging(settings.log_level)
    if service is None:
        from .bundle import load_bundle

        service = PlotPriceService(load_bundle(settings.model_path))

    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=16 * 1024, SEND_FILE_MAX_AGE_DEFAULT=86400)
    app.json.sort_keys = False
    app.json.ensure_ascii = False
    app.extensions["plotify"] = {
        "service": service,
        "settings": settings,
        "limiter": RateLimiter(settings.rate_limit_per_minute),
    }
    if settings.trusted_proxies:
        n = settings.trusted_proxies
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=n, x_proto=n, x_host=n)

    css = Path(app.static_folder, "app.css").read_bytes()
    asset_version = hashlib.sha256(css).hexdigest()[:10]

    app.jinja_env.filters.update(inr=format_inr, inr_short=format_inr_short, num=format_number,
                                 area=format_area, pct=format_pct, multiplier=format_multiplier,
                                 inputs=describe_inputs)

    @app.context_processor
    def _globals():
        return {"asset_version": asset_version, "model_meta": service.metadata, "app_version": __version__,
                "region": service.region}

    from .api import api
    from .web import web

    app.register_blueprint(web)
    app.register_blueprint(api)

    @app.before_request
    def _start_timer():
        g.started = time.perf_counter()
        incoming = request.headers.get("X-Request-ID", "")
        g.request_id = incoming if 0 < len(incoming) <= 64 and incoming.isprintable() else uuid.uuid4().hex

    @app.after_request
    def _finish(response):
        apply_security_headers(response, hsts=settings.hsts)
        response.headers["X-Request-ID"] = g.get("request_id", "")
        if request.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        elapsed = (time.perf_counter() - g.get("started", time.perf_counter())) * 1000
        log.info("%s %s %s %.1fms id=%s", request.method, request.path, response.status_code, elapsed,
                 g.get("request_id", "-"))
        return response

    def _error_response(error: HTTPException):
        if request.path.startswith("/api/"):
            body = {"error": {"code": error.name.lower().replace(" ", "_"), "message": error.description,
                              "request_id": g.get("request_id")}}
            response = jsonify(body)
        else:
            response = app.make_response(render_template("error.html", error=error))
        response.status_code = error.code or 500
        for name, value in error.get_headers():
            if name.lower() != "content-type":
                response.headers[name] = value
        return response

    @app.errorhandler(HTTPException)
    def _http_error(error: HTTPException):
        return _error_response(error)

    @app.errorhandler(Exception)
    def _unhandled(error: Exception):
        if isinstance(error, HTTPException):
            return _error_response(error)
        log.exception("Unhandled error id=%s", g.get("request_id", "-"))
        return _error_response(InternalServerError("Something went wrong on our side. Please try again."))

    return app
