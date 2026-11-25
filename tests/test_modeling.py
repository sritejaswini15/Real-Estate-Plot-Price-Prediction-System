import json

import numpy as np
import pandas as pd
import pytest

from plotify.cleaning import build_dataset
from plotify.modeling import (
    FEATURE_COLUMNS,
    LocalityMedianRateBaseline,
    LogRateRegressor,
    MedianRateBaseline,
    candidate_models,
    interval_coverage,
    log_residual_interval,
    regression_metrics,
    repeated_group_kfold,
    select_one_standard_error,
)
from plotify.training import TEST_LOG_FILENAME, TestSetAlreadyUsed, TrainConfig, split_train_test, train

from .conftest import LOCALITY_RATES, make_raw


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    localities = rng.choice(list(LOCALITY_RATES), n)
    area = rng.choice([150.0, 200.0, 300.0, 500.0], n)
    X = pd.DataFrame({
        "area_sqyd": area,
        "locality": localities,
        "facing": rng.choice(["East", "West", None], n),
        "open_sides": rng.choice([1.0, 2.0, np.nan], n),
        "road_width_ft": rng.choice([30.0, 60.0, np.nan], n),
        "in_project": rng.random(n) < 0.3,
    })
    y = area * np.array([LOCALITY_RATES[loc] for loc in localities])
    return X, y


def test_repeated_group_kfold_never_splits_a_group():
    groups = np.repeat(np.arange(30), 3)
    for folds in repeated_group_kfold(groups, n_splits=5, n_repeats=3, seed=1):
        seen = []
        for train_idx, valid_idx in folds:
            assert not set(groups[train_idx]) & set(groups[valid_idx])
            seen.extend(valid_idx)
        assert sorted(seen) == list(range(len(groups)))


def test_log_rate_regressor_recovers_locality_rates():
    X, y = _frame()
    model = candidate_models()["ridge_log_rate"].fit(X, y)
    pred = model.predict(X)
    assert (pred > 0).all()
    assert np.max(np.abs(pred / y - 1)) < 0.05


def test_every_candidate_fits_and_predicts_with_missing_inputs():
    X, y = _frame()
    query = pd.DataFrame([{"area_sqyd": 210.0, "locality": "Unseen", "facing": None, "open_sides": np.nan,
                           "road_width_ft": np.nan, "in_project": False}])
    for name, model in candidate_models().items():
        prediction = model.fit(X, y).predict(query)
        assert np.isfinite(prediction).all(), name


def test_locality_baseline_falls_back_to_overall_median():
    X = pd.DataFrame({"area_sqyd": [100.0] * 5, "locality": ["A", "A", "A", "B", "B"]})
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0]) * 100
    model = LocalityMedianRateBaseline(min_count=3).fit(X, y)
    pred = model.predict(pd.DataFrame({"area_sqyd": [10.0, 10.0], "locality": ["A", "B"]}))
    assert pred[0] == pytest.approx(20.0)  # A's own median rate (2) × 10
    assert pred[1] == pytest.approx(30.0)  # B is rare: overall median rate (3) × 10
    assert MedianRateBaseline().fit(X, y).predict(X.iloc[:1])[0] == pytest.approx(300.0)


def test_regression_metrics():
    y = np.array([100.0, 200.0, 400.0])
    perfect = regression_metrics(y, y)
    assert perfect["mape"] == 0 and perfect["r2_log_price"] == pytest.approx(1.0)
    bad = regression_metrics(y, np.array([-50.0, 200.0, 400.0]))
    assert bad["nonpositive_predictions"] == 1
    assert bad["within_20pct"] == pytest.approx(2 / 3)


def test_interval_from_out_of_fold_residuals_has_nominal_coverage():
    rng = np.random.default_rng(3)
    y = np.exp(rng.normal(15, 1, 400))
    oof = y * np.exp(rng.normal(0, 0.3, (5, 400)))
    interval = log_residual_interval(y, oof, level=0.8)
    assert interval["log_low"] < 0 < interval["log_high"]
    assert interval_coverage(y, oof[0], interval) == pytest.approx(0.8, abs=0.06)


def test_one_standard_error_rule():
    y = np.full(200, 100.0)
    groups = np.arange(200)
    best = np.full(200, 130.0)  # every error 30%
    # Errors of 10% and 52% alternate: mean gap +1 point, but very noisy, so within one SE.
    noisy = np.where(np.arange(200) % 2 == 0, 110.0, 152.0)
    worse = np.full(200, 160.0)  # consistently 30 points worse
    selected, details = select_one_standard_error(y, {"simple": noisy[None, :], "complex": best[None, :]}, groups)
    assert details["best_by_mape"] == "complex"
    assert selected == "simple"
    assert details["candidates"]["simple"]["gap_to_best"] == pytest.approx(0.01)
    assert details["candidates"]["simple"]["within_one_se"]
    selected, _ = select_one_standard_error(y, {"simple": worse[None, :], "complex": best[None, :]}, groups)
    assert selected == "complex"


def test_one_standard_error_rule_averages_within_groups():
    y = np.full(4, 100.0)
    best = np.full(4, 125.0)  # MAPE 0.25
    other = np.array([110.0, 110.0, 110.0, 190.0])  # MAPE 0.30; rows 0-2 are one cross-posted plot
    _, details = select_one_standard_error(y, {"other": other[None, :], "best": best[None, :]},
                                           np.array([0, 0, 0, 1]))
    assert details["best_by_mape"] == "best"
    # Per-group gaps are -0.15 and +0.65, so the mean is +0.25, not the per-row mean of +0.05.
    assert details["candidates"]["other"]["gap_to_best"] == pytest.approx(0.25)


def test_test_split_keeps_groups_apart():
    data = build_dataset(make_raw()).data
    train_idx, test_idx = split_train_test(data)
    assert not set(data.loc[train_idx, "group_id"]) & set(data.loc[test_idx, "group_id"])
    assert 0.1 < len(test_idx) / len(data) < 0.35


def test_training_outputs(trained):
    root, result = trained["root"], trained["result"]
    metrics = result["metrics"]
    assert metrics["selected_model"] in metrics["cv"]
    assert set(metrics["cv"]) == {"median_rate", "locality_median_rate", "ridge_log_rate"}
    assert metrics["test"]["count"] == 0  # training alone never touches the test split
    for path in ["models/plotify_model.joblib", "models/plotify_model.manifest.json", "models/metrics.json",
                 "reports/data_quality.json", "plots.csv"]:
        assert (root / path).is_file(), path
    saved = json.loads((root / "models" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["selected_model"] == metrics["selected_model"]


def test_test_split_is_scored_once_unless_explicitly_allowed(raw_path, tmp_path):
    config = TrainConfig(data_path=raw_path, model_dir=tmp_path / "models", report_dir=tmp_path / "reports",
                         processed_path=None, n_repeats=1, candidates=("median_rate", "locality_median_rate"),
                         evaluate_test=True, log=lambda *_: None)
    first = train(config)
    assert first["metrics"]["test"]["count"] == 1
    latest = first["metrics"]["test"]["latest"]
    assert latest["selected_model"] in latest["results"]
    assert first["metrics"]["test"]["matches_deployed"]

    with pytest.raises(TestSetAlreadyUsed):
        train(config)
    log = json.loads((tmp_path / "models" / TEST_LOG_FILENAME).read_text(encoding="utf-8"))
    assert log["count"] == 1

    config.allow_retest = True
    assert train(config)["metrics"]["test"]["count"] == 2

    # A different dataset in the same model folder gets its own once-only test.
    other = tmp_path / "other.csv"
    make_raw(seed=99, region="Shamshabad").to_csv(other, index=False)
    config.data_path, config.region, config.allow_retest = other, "Shamshabad", False
    fresh = train(config)["metrics"]["test"]
    assert fresh["count"] == 1 and fresh["latest"]["evaluation_number"] == 1
    with pytest.raises(TestSetAlreadyUsed):
        train(config)


def test_feature_columns_are_the_service_contract(service):
    assert service.bundle["schema"]["features"] == FEATURE_COLUMNS
    assert isinstance(candidate_models()["ridge_log_rate"], LogRateRegressor)
