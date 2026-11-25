"""Command line: ``python -m plotify {train,predict,serve,eda}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_MODEL_PATH


def _utf8_output() -> None:
    # Rupee signs and other symbols must not crash output piped on Windows (cp1252).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            stream.reconfigure(encoding="utf-8", errors="replace")


def _train(args: argparse.Namespace) -> int:
    from .training import TestSetAlreadyUsed, TrainConfig, train

    config = TrainConfig(data_path=args.data, region=args.region, model_dir=args.model_dir,
                         n_repeats=args.repeats, evaluate_test=args.evaluate_test,
                         allow_retest=args.allow_retest)
    try:
        train(config)
    except TestSetAlreadyUsed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _predict(args: argparse.Namespace) -> int:
    from .bundle import load_bundle
    from .formatting import format_area, format_inr, format_inr_short
    from .service import PlotPriceService, QueryError

    service = PlotPriceService(load_bundle(args.model))
    raw = {"area": args.area, "area_unit": args.unit, "locality": args.locality, "facing": args.facing,
           "open_sides": args.open_sides, "road_width_ft": args.road_width, "in_project": args.in_project}
    try:
        estimate = service.estimate(service.parse({k: v for k, v in raw.items() if v is not None}))
    except QueryError as exc:
        for field, message in exc.errors.items():
            print(f"error: {field}: {message}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({**estimate.to_dict(), "model": service.metadata}, indent=2, ensure_ascii=False))
        return 0
    level = round(estimate.interval_level * 100)
    print(f"Estimated asking price: {format_inr_short(estimate.price_inr)} ({format_inr(estimate.price_inr)})")
    print(f"{level}% range:             {format_inr_short(estimate.low_inr)} – {format_inr_short(estimate.high_inr)}")
    print(f"Price per sq yd:        {format_inr(estimate.price_per_sqyd)}")
    for line in [*estimate.warnings, *estimate.notes]:
        print(f"  • {line}")
    if estimate.comparables:
        print("Similar listings:")
        for c in estimate.comparables:
            print(f"  row {c['source_row']:>3}  {c['locality']:<14} {format_area(c['area_sqyd']):>14}  "
                  f"{format_inr_short(c['price_inr']):>13}  {format_inr(c['price_per_sqyd'])}/sq yd")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from waitress import serve

    from . import create_app

    app = create_app()
    print(f"Plotify on http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    serve(app, host=args.host, port=args.port, threads=args.threads, ident="plotify")
    return 0


def _eda(args: argparse.Namespace) -> int:
    from .eda import run_eda

    run_eda(args.data, args.out, region=args.region)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .domain import DEFAULT_REGION
    from .training import DEFAULT_DATA, DEFAULT_MODEL_DIR, DEFAULT_REPORT_DIR

    parser = argparse.ArgumentParser(prog="python -m plotify",
                                     description="Residential plot asking-price estimator.")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="clean the data, compare models, save the deployed model")
    t.add_argument("--data", type=Path, default=DEFAULT_DATA, help="raw .xlsx or .csv export")
    t.add_argument("--region", default=DEFAULT_REGION,
                   help=f"area the listings cover, as written in their Location field (default {DEFAULT_REGION})")
    t.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    t.add_argument("--repeats", type=int, default=10, help="CV repeats (default 10)")
    t.add_argument("--evaluate-test", action="store_true", help="score the held-out test split (logged)")
    t.add_argument("--allow-retest", action="store_true", help="permit scoring the test split again")
    t.set_defaults(func=_train)

    p = sub.add_parser("predict", help="estimate one plot's asking price")
    p.add_argument("--area", required=True, help="plot area (in --unit)")
    p.add_argument("--unit", default="sqyd", help="sqyd (default), sqft, sqm, guntha or acre")
    p.add_argument("--locality", help="e.g. Isnapur; default: the region the model was trained on")
    p.add_argument("--facing", help="e.g. East, North-East")
    p.add_argument("--open-sides", type=int)
    p.add_argument("--road-width", type=float, help="road width in feet")
    p.add_argument("--in-project", action="store_true", default=None, help="inside a named layout/venture")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--json", action="store_true", help="print the API JSON response")
    p.set_defaults(func=_predict)

    s = sub.add_parser("serve", help="run the web app with waitress")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--threads", type=int, default=4)
    s.set_defaults(func=_serve)

    e = sub.add_parser("eda", help="write exploratory figures and a summary to reports/")
    e.add_argument("--data", type=Path, default=DEFAULT_DATA)
    e.add_argument("--region", default=DEFAULT_REGION)
    e.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)
    e.set_defaults(func=_eda)
    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_output()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
