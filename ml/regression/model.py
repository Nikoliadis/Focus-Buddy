"""Candidate models for "how many minutes will this session be focused?".

Baselines first: a model is only worth shipping if it beats the simplest rule a person
would write by hand.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_COLUMNS

# Raw `hour` and `weekday` are categories stored as integers. A linear model would read
# them as quantities (Sunday = 6 × Monday), so it gets the cyclic encoding instead.
# One-hot encoding comes with the full preprocessing pipeline (Phase 11).
LINEAR_FEATURES = [c for c in FEATURE_COLUMNS if c not in ("hour", "weekday")]


class PlannedMinutesBaseline(RegressorMixin, BaseEstimator):
    """Predict that the user focuses for exactly the planned duration."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame):
        return X["planned_minutes"].to_numpy(dtype=float)


class UserHistoryBaseline(RegressorMixin, BaseEstimator):
    """Predict the user's past share of planned time × this session's planned length.
    Falls back to the planned length when the user has no history."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame):
        planned = X["planned_minutes"]
        past_ratio = (X["prev_mean_minutes"] / planned).clip(upper=1)
        return (past_ratio * planned).fillna(planned).to_numpy(dtype=float)


def make_models(random_state: int = 0, n_jobs: int = 1) -> dict[str, BaseEstimator]:
    """``n_jobs`` defaults to 1: joblib's parallel workers deadlock under gevent
    monkey-patching (which the web app applies), so parallelism must be opted into
    from a plain process such as the training CLI."""
    linear = Pipeline([
        ("prep", ColumnTransformer(
            [("num", Pipeline([
                # Medians are learned from the training data only, inside fit().
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), LINEAR_FEATURES)],
            remainder="drop",
        )),
        ("model", LinearRegression()),
    ])
    forest = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,   # no leaf predicts from fewer than 5 sessions: limits overfitting
        max_features=0.5,
        n_jobs=n_jobs,
        random_state=random_state,
    )  # handles NaN natively (scikit-learn >= 1.4), so no imputer
    return {
        "baseline_global_mean": DummyRegressor(strategy="mean"),
        "baseline_planned": PlannedMinutesBaseline(),
        "baseline_user_history": UserHistoryBaseline(),
        "linear_regression": linear,
        "random_forest": forest,
    }


def is_baseline(name: str) -> bool:
    return name.startswith("baseline_")
