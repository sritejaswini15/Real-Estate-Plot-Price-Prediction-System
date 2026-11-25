"""Estimators, features, grouped cross-validation and metrics.

Every model takes the same canonical columns (``FEATURE_COLUMNS``) at fit and
predict time. Fit and predict share one feature function, so training and
serving cannot drift apart. The original project did not share it, and its
prediction script silently ignored plot area, open sides and road width.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .domain import MIN_LOCALITY_COUNT

FEATURE_COLUMNS = ["area_sqyd", "locality", "facing", "open_sides", "road_width_ft", "in_project"]
TARGET_COLUMN = "price_inr"
NUMERIC_FEATURES = ["area", "road_width_ft", "open_sides", "in_project"]
CATEGORICAL_FEATURES = ["facing", "locality"]


def _base_features(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "road_width_ft": pd.to_numeric(X["road_width_ft"], errors="coerce").astype(float),
        "open_sides": pd.to_numeric(X["open_sides"], errors="coerce").astype(float),
        "in_project": X["in_project"].astype(bool).astype(float),
        "facing": X["facing"].astype(object).where(X["facing"].notna(), "Unknown").astype(str),
        "locality": X["locality"].astype(object).where(X["locality"].notna(), "Unknown").astype(str),
    }, index=X.index)


def log_area_features(X: pd.DataFrame) -> pd.DataFrame:
    """Model features with plot area on a log scale (for log-rate models)."""
    out = _base_features(X)
    out.insert(0, "area", np.log(X["area_sqyd"].astype(float)))
    return out


def raw_area_features(X: pd.DataFrame) -> pd.DataFrame:
    """Model features with plot area in sq yd (for the original raw-price model)."""
    out = _base_features(X)
    out.insert(0, "area", X["area_sqyd"].astype(float))
    return out


def make_preprocessor(log_area: bool = True) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_LOCALITY_COUNT,
                                sparse_output=False)
    columns = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    features = log_area_features if log_area else raw_area_features
    return Pipeline([("features", FunctionTransformer(features)), ("columns", columns)])


class LogRateRegressor(RegressorMixin, BaseEstimator):
    """Fit ``log(price / area)`` and predict ``area * exp(prediction)``.

    Price per square yard is the quantity buyers compare. Its log is roughly
    symmetric, and predictions are always positive, unlike a linear model on
    raw price.
    """

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X: pd.DataFrame, y) -> "LogRateRegressor":
        area = X["area_sqyd"].to_numpy(dtype=float)
        target = np.log(np.asarray(y, dtype=float) / area)
        self.regressor_ = clone(self.regressor).fit(X, target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "regressor_")
        return X["area_sqyd"].to_numpy(dtype=float) * np.exp(self.regressor_.predict(X))


class MedianRateBaseline(RegressorMixin, BaseEstimator):
    """Baseline: area x the median price per sq yd of the training listings."""

    def fit(self, X: pd.DataFrame, y) -> "MedianRateBaseline":
        self.rate_ = float(np.median(np.asarray(y, dtype=float) / X["area_sqyd"].to_numpy(dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "rate_")
        return X["area_sqyd"].to_numpy(dtype=float) * self.rate_


class LocalityMedianRateBaseline(RegressorMixin, BaseEstimator):
    """Baseline: area x the median price per sq yd in the plot's locality.

    Localities with fewer than ``min_count`` training listings use the overall
    median. This is roughly what a person would do by hand from the listings.
    """

    def __init__(self, min_count: int = MIN_LOCALITY_COUNT):
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y) -> "LocalityMedianRateBaseline":
        rate = pd.Series(np.asarray(y, dtype=float) / X["area_sqyd"].to_numpy(dtype=float), index=X.index)
        groups = rate.groupby(X["locality"].astype(str))
        counts, medians = groups.size(), groups.median()
        self.rates_ = {k: float(v) for k, v in medians[counts >= self.min_count].items()}
        self.overall_rate_ = float(rate.median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "rates_")
        rates = X["locality"].astype(str).map(self.rates_).fillna(self.overall_rate_).to_numpy(dtype=float)
        return X["area_sqyd"].to_numpy(dtype=float) * rates


MODEL_LABELS = {
    "median_rate": "Baseline: overall median ₹/sq yd × area",
    "locality_median_rate": "Baseline: locality median ₹/sq yd × area",
    "linear_price": "Linear regression on raw price (the original model form)",
    "ridge_log_rate": "Ridge regression on log ₹/sq yd",
    "random_forest_log_rate": "Random forest on log ₹/sq yd",
}


def model_inputs(name: str) -> list[str]:
    """The inputs that can change a candidate's estimate."""
    return {
        "median_rate": ["area_sqyd"],
        "locality_median_rate": ["area_sqyd", "locality"],
    }.get(name, list(FEATURE_COLUMNS))


def candidate_models(seed: int = 0) -> dict[str, BaseEstimator]:
    """The models compared by cross-validation, cheapest first."""
    return {
        "median_rate": MedianRateBaseline(),
        "locality_median_rate": LocalityMedianRateBaseline(),
        "linear_price": Pipeline([("prep", make_preprocessor(log_area=False)), ("model", LinearRegression())]),
        "ridge_log_rate": LogRateRegressor(Pipeline([
            ("prep", make_preprocessor()),
            ("model", RidgeCV(alphas=np.logspace(-2, 3, 11))),
        ])),
        "random_forest_log_rate": LogRateRegressor(Pipeline([
            ("prep", make_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                            random_state=seed, n_jobs=1)),
        ])),
    }


def repeated_group_kfold(groups: np.ndarray, n_splits: int = 5, n_repeats: int = 10,
                         seed: int = 0) -> Iterator[list[tuple[np.ndarray, np.ndarray]]]:
    """Yield one list of ``(train_idx, valid_idx)`` folds per repeat.

    Rows sharing a group never appear on both sides of a fold.
    """
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    placeholder = np.zeros(len(groups))
    for repeat in range(n_repeats):
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        yield list(splitter.split(placeholder, groups=groups))


def out_of_fold_predictions(model: BaseEstimator, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                            n_splits: int = 5, n_repeats: int = 10, seed: int = 0) -> np.ndarray:
    """Return an ``(n_repeats, n_rows)`` array of out-of-fold predictions."""
    y = np.asarray(y, dtype=float)
    predictions = np.full((n_repeats, len(y)), np.nan)
    for repeat, folds in enumerate(repeated_group_kfold(groups, n_splits, n_repeats, seed)):
        for train_idx, valid_idx in folds:
            fitted = clone(model).fit(X.iloc[train_idx], y[train_idx])
            predictions[repeat, valid_idx] = fitted.predict(X.iloc[valid_idx])
    return predictions


def select_one_standard_error(y_true, oof_by_name: dict[str, np.ndarray], groups) -> tuple[str, dict]:
    """Pick the simplest candidate within one standard error of the best.

    ``oof_by_name`` must be ordered from simplest to most complex. Each
    listing's absolute percentage error is averaged over CV repeats, then
    over each plot group (cross-posted copies are not independent). A
    candidate's paired gap to the best model is compared with the standard
    error of that gap. This is the "one-SE rule" (Hastie, Tibshirani &
    Friedman, *Elements of Statistical Learning*, §7.10). With about 100
    listings it stops a complex model winning on noise.
    """
    y_true = np.asarray(y_true, dtype=float)
    groups = pd.Series(np.asarray(groups))
    ape = {name: (np.abs(oof - y_true) / y_true).mean(axis=0) for name, oof in oof_by_name.items()}
    best = min(ape, key=lambda name: ape[name].mean())
    details: dict[str, dict[str, float | bool]] = {}
    selected = None
    for name, errors in ape.items():
        gap = pd.Series(errors - ape[best]).groupby(groups.to_numpy()).mean()
        se = float(gap.std(ddof=1) / np.sqrt(len(gap))) if len(gap) > 1 else 0.0
        within = bool(gap.mean() <= se)
        details[name] = {"mape": float(errors.mean()), "gap_to_best": float(gap.mean()), "gap_se": se,
                         "within_one_se": within}
        if within and selected is None:
            selected = name
    return selected or best, {"best_by_mape": best, "candidates": details}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Price-error metrics. Non-positive predictions are counted, not hidden."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true
    log_true = np.log(y_true)
    log_pred = np.log(np.clip(y_pred, 1.0, None))  # a ≤ ₹0 prediction is scored as ₹1
    ss_res = float(np.sum((log_true - log_pred) ** 2))
    ss_tot = float(np.sum((log_true - log_true.mean()) ** 2))
    return {
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "within_20pct": float(np.mean(ape <= 0.20)),
        "mae_inr": float(np.mean(np.abs(y_pred - y_true))),
        "r2_log_price": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "nonpositive_predictions": int(np.sum(y_pred <= 0)),
    }


def summarise_repeats(y_true, oof: np.ndarray) -> dict[str, dict[str, float]]:
    """Metrics on each repeat's pooled out-of-fold predictions: mean and std."""
    per_repeat = [regression_metrics(y_true, row) for row in oof]
    keys = per_repeat[0].keys()
    return {
        "mean": {k: float(np.mean([m[k] for m in per_repeat])) for k in keys},
        "std": {k: float(np.std([m[k] for m in per_repeat], ddof=1)) if len(per_repeat) > 1 else 0.0
                for k in keys},
    }


def log_residual_interval(y_true, oof: np.ndarray, level: float = 0.8) -> dict[str, float]:
    """An empirical prediction interval from out-of-fold log residuals.

    ``price × exp(log_low)`` to ``price × exp(log_high)`` covered ``level`` of
    the held-out listings during cross-validation. Treat it as an estimate,
    because so few listings back it.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(oof, dtype=float)
    residuals = (np.log(y_true)[None, :] - np.log(np.clip(pred, 1.0, None))).ravel()
    residuals = residuals[np.isfinite(residuals)]
    alpha = (1.0 - level) / 2.0
    return {
        "level": level,
        "log_low": float(np.quantile(residuals, alpha)),
        "log_high": float(np.quantile(residuals, 1.0 - alpha)),
        "n_residuals": int(residuals.size),
    }


def interval_coverage(y_true, y_pred, interval: dict[str, float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    low = y_pred * np.exp(interval["log_low"])
    high = y_pred * np.exp(interval["log_high"])
    return float(np.mean((y_true >= low) & (y_true <= high)))
