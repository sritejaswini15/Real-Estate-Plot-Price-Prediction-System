"""Training workflow: clean, split, cross-validate, select, test once, refit, save.

Protocol (fixed before any model was scored):

* A grouped 20% test split (seed ``SPLIT_SEED``) is set aside. Cross-posted
  copies of one plot share a group, so they never straddle the split.
* Candidates are compared by repeated grouped 5-fold cross-validation on the
  training split only. The simplest candidate within one standard error of
  the lowest-MAPE model wins (see ``select_one_standard_error``). The first
  CV run showed the forest's lead over the locality-median baseline was
  smaller than its standard error, and the rule was adopted then. The test
  split had not been scored.
* The test split is scored only with ``--evaluate-test``. Every scoring is
  appended to ``models/test_evaluations.json``. A second scoring needs
  ``--allow-retest`` and is shown with its count, so a test result is never
  quietly tuned against.
* The deployed model is the selected configuration refit on all clean rows.
  Its 80% range comes from out-of-fold residuals of grouped CV on those rows.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit

from .bundle import save_bundle
from .cleaning import build_dataset, load_raw
from .domain import DEFAULT_REGION, FACINGS, MIN_LOCALITY_COUNT
from .market import market_summary
from .modeling import (
    FEATURE_COLUMNS,
    MODEL_LABELS,
    TARGET_COLUMN,
    candidate_models,
    interval_coverage,
    log_residual_interval,
    model_inputs,
    out_of_fold_predictions,
    regression_metrics,
    select_one_standard_error,
    summarise_repeats,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "99acres_patancheru_properties.xlsx"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed" / "plots.csv"
MODEL_FILENAME = "plotify_model.joblib"
TEST_LOG_FILENAME = "test_evaluations.json"
SPLIT_SEED = 2026
TEST_FRACTION = 0.2
SELECTION_METRIC = "mape"
MIN_ROWS = 20


class TestSetAlreadyUsed(RuntimeError):
    """Raised when asked to score the test split again without ``allow_retest``."""

    __test__ = False  # not a pytest test class


@dataclass
class TrainConfig:
    data_path: Path = DEFAULT_DATA
    region: str = DEFAULT_REGION
    model_dir: Path = DEFAULT_MODEL_DIR
    report_dir: Path = DEFAULT_REPORT_DIR
    processed_path: Path | None = DEFAULT_PROCESSED
    n_splits: int = 5
    n_repeats: int = 10
    seed: int = 0
    candidates: tuple[str, ...] | None = None
    evaluate_test: bool = False
    allow_retest: bool = False
    interval_level: float = 0.8
    log: Any = field(default=print, repr=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_test_log(model_dir: Path) -> dict[str, Any]:
    path = Path(model_dir) / TEST_LOG_FILENAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"count": 0, "evaluations": []}


def split_train_test(data: pd.DataFrame, seed: int = SPLIT_SEED,
                     fraction: float = TEST_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train_idx, test_idx = next(splitter.split(data, groups=data["group_id"]))
    return np.sort(train_idx), np.sort(test_idx)


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    return (df[FEATURE_COLUMNS].reset_index(drop=True), df[TARGET_COLUMN].to_numpy(dtype=float),
            df["group_id"].to_numpy())


def _schema(data: pd.DataFrame, region: str) -> dict[str, Any]:
    counts = data["locality"].value_counts()
    road = pd.to_numeric(data["road_width_ft"], errors="coerce").dropna().astype(float)
    area = data["area_sqyd"].astype(float)
    return {
        "region": region,
        "features": FEATURE_COLUMNS,
        "facings": list(FACINGS),
        "localities": sorted(counts[counts >= MIN_LOCALITY_COUNT].index.tolist()),
        "rare_localities": sorted(counts[counts < MIN_LOCALITY_COUNT].index.tolist()),
        "area_sqyd": {"min": float(area.min()), "p05": float(area.quantile(0.05)),
                      "p95": float(area.quantile(0.95)), "max": float(area.max())},
        "road_width_ft": {"min": float(road.min()), "max": float(road.max())} if len(road) else None,
    }


def train(config: TrainConfig | None = None) -> dict[str, Any]:
    config = config or TrainConfig()
    log = config.log
    model_dir = Path(config.model_dir)
    data_path = Path(config.data_path)
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    # The once-only rule is per dataset: a different data file has its own test split.
    test_log = read_test_log(model_dir)
    previous = [e for e in test_log["evaluations"] if e.get("data_sha256") == data_sha256]
    if config.evaluate_test and previous and not config.allow_retest:
        raise TestSetAlreadyUsed(
            f"This dataset's test split has already been scored {len(previous)} time(s). Re-scoring after "
            "changes leaks the test set into model choice. Pass --allow-retest to do it anyway; the "
            "count is recorded and shown."
        )

    cleaned = build_dataset(load_raw(data_path), region=config.region)
    data = cleaned.data
    log(f"Cleaned {cleaned.report['raw_rows']} raw rows -> {len(data)} listings "
        f"({len(cleaned.report['dropped_invalid'])} invalid, "
        f"{len(cleaned.report['dropped_duplicate_source_rows'])} duplicates dropped)")
    if len(data) < MIN_ROWS:
        raise ValueError(f"Only {len(data)} usable listings; need at least {MIN_ROWS} to train.")
    if config.processed_path:
        Path(config.processed_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(config.processed_path, index=False)
    _write_json(Path(config.report_dir) / "data_quality.json", {"data_file": data_path.name,
                                                               "data_sha256": data_sha256, **cleaned.report})

    train_idx, test_idx = split_train_test(data)
    train_df, test_df = data.iloc[train_idx], data.iloc[test_idx]
    X_train, y_train, g_train = _xy(train_df)
    log(f"Split: {len(train_df)} training / {len(test_df)} test listings "
        f"({train_df['group_id'].nunique()} / {test_df['group_id'].nunique()} plot groups)")

    models = candidate_models(config.seed)
    if config.candidates:
        unknown = set(config.candidates) - set(models)
        if unknown:
            raise ValueError(f"Unknown candidate(s) {sorted(unknown)}; choose from {sorted(models)}")
        models = {k: v for k, v in models.items() if k in config.candidates}

    cv: dict[str, Any] = {}
    oof_train: dict[str, np.ndarray] = {}
    for name, model in models.items():
        oof = out_of_fold_predictions(model, X_train, y_train, g_train, config.n_splits, config.n_repeats,
                                      config.seed)
        oof_train[name] = oof
        cv[name] = {"label": MODEL_LABELS.get(name, name), **summarise_repeats(y_train, oof)}
        m = cv[name]["mean"]
        log(f"  CV {name:<24} MAPE {m['mape']:.1%}  median APE {m['median_ape']:.1%}  "
            f"within 20% {m['within_20pct']:.0%}  R² (log) {m['r2_log_price']:.2f}")
    selected, selection = select_one_standard_error(y_train, oof_train, g_train)
    for name, d in selection["candidates"].items():
        cv[name]["selection"] = d
    log(f"Lowest CV MAPE: {selection['best_by_mape']}; selected (simplest within one SE): {selected}")

    if config.evaluate_test:
        X_test, y_test, _ = _xy(test_df)
        evaluated = {}
        for name in dict.fromkeys([selected, "locality_median_rate", "median_rate"]):
            if name not in models:
                continue
            fitted = clone(models[name]).fit(X_train, y_train)
            pred = fitted.predict(X_test)
            interval = log_residual_interval(y_train, oof_train[name], config.interval_level)
            evaluated[name] = {"label": MODEL_LABELS.get(name, name), **regression_metrics(y_test, pred),
                               "interval_coverage": interval_coverage(y_test, pred, interval)}
        entry = {
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "evaluation_number": len(previous) + 1,
            "selected_model": selected,
            "data_sha256": data_sha256,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "split_seed": SPLIT_SEED,
            "interval_level": config.interval_level,
            "results": evaluated,
        }
        test_log = {"count": test_log["count"] + 1, "evaluations": [*test_log["evaluations"], entry]}
        _write_json(model_dir / TEST_LOG_FILENAME, test_log)
        previous.append(entry)
        r = evaluated[selected]
        log(f"TEST (evaluation #{entry['evaluation_number']}) {selected}: MAPE {r['mape']:.1%}, median APE "
            f"{r['median_ape']:.1%}, {config.interval_level:.0%} range coverage {r['interval_coverage']:.0%} "
            f"on {len(test_df)} listings")

    X_all, y_all, g_all = _xy(data)
    final_model = clone(models[selected]).fit(X_all, y_all)
    oof_all = out_of_fold_predictions(models[selected], X_all, y_all, g_all, config.n_splits,
                                      config.n_repeats, config.seed)
    interval = log_residual_interval(y_all, oof_all, config.interval_level)
    deployed_cv = summarise_repeats(y_all, oof_all)
    geo_mean_oof = np.exp(np.log(np.clip(oof_all, 1.0, None)).mean(axis=0))

    trained_at = datetime.now(timezone.utc)
    latest_test = previous[-1] if previous else None  # this dataset's scorings only
    metrics = {
        "selection_metric": SELECTION_METRIC,
        "selection_rule": "simplest candidate within one standard error (paired, per plot group) of the "
                          "lowest-MAPE candidate",
        "best_by_mape": selection["best_by_mape"],
        "selected_model": selected,
        "cv_protocol": {"n_splits": config.n_splits, "n_repeats": config.n_repeats, "grouped": True,
                        "rows": int(len(train_df)), "seed": config.seed},
        "cv": cv,
        "deployed_cv": {"rows": int(len(data)), **deployed_cv},
        "interval": interval,
        "test": {
            "count": len(previous),
            "latest": latest_test,
            "matches_deployed": bool(latest_test and latest_test["selected_model"] == selected),
        },
    }
    bundle = {
        "estimator": final_model,
        "model_name": selected,
        "model_label": MODEL_LABELS.get(selected, selected),
        "model_inputs": model_inputs(selected),
        "interval": interval,
        "schema": _schema(data, config.region),
        "listings": [{k: (None if v is pd.NA or (isinstance(v, float) and np.isnan(v)) else v)
                      for k, v in row.items()} for row in data.astype(object).to_dict("records")],
        "market": market_summary(data),
        "metrics": _jsonable(metrics),
        "oof_points": [
            {"source_row": int(s), "locality": str(loc), "area_sqyd": float(a), "actual": float(y),
             "predicted": float(p)}
            for s, loc, a, y, p in zip(data["source_row"], data["locality"], data["area_sqyd"], y_all, geo_mean_oof)
        ],
        "data_quality": _jsonable(cleaned.report),
        "metadata": {
            "model_version": f"{trained_at:%Y%m%d%H%M%S}-{data_sha256[:8]}",
            "trained_at": trained_at.isoformat(timespec="seconds"),
            "data_file": data_path.name,
            "data_sha256": data_sha256,
            "region": config.region,
            "rows": int(len(data)),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
    }
    manifest = save_bundle(bundle, model_dir / MODEL_FILENAME)
    _write_json(model_dir / "metrics.json", {"model_version": manifest["model_version"], **metrics})
    log(f"Deployed model ({selected}, refit on {len(data)} listings): CV MAPE {deployed_cv['mean']['mape']:.1%}; "
        f"{config.interval_level:.0%} range ×{np.exp(interval['log_low']):.2f}–×{np.exp(interval['log_high']):.2f}")
    log(f"Saved {model_dir / MODEL_FILENAME} (version {manifest['model_version']})")
    return {"selected": selected, "metrics": metrics, "manifest": manifest, "data_quality": cleaned.report}
