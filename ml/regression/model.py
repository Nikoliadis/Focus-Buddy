"""Candidate models for "how many minutes will this session be focused?".

Baselines first: a model is only worth shipping if it beats the simplest rule a person
would write by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from analytics.analysis import FULL_DURATION_THRESHOLD
from ml.pipelines import make_preprocessor


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


class HurdleRegressor(RegressorMixin, BaseEstimator):
    """Two-part model for a target that is mostly exactly the planned length.

        P(early)       logistic classifier: will the session end early?
        early_ratio    random forest on early sessions only: what share of the plan?

    point="mean":    ŷ = (1 - P) · planned + P · early_ratio · planned
                     the expected value, the right summary for squared-error metrics
    point="median":  ŷ = planned if P < 0.5 else early_ratio · planned
                     approximately the median of that mixture, the right summary for MAE
                     (the median lies in the early part only when P >= 0.5)
    """

    def __init__(self, point: str = "mean", random_state: int = 0, n_jobs: int = 1):
        self.point = point
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y):
        from ml.classification.model import make_logistic  # local: classification imports this module

        if self.point not in ("mean", "median"):
            raise ValueError("point must be 'mean' or 'median'")
        planned = X["planned_minutes"].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        early = y < FULL_DURATION_THRESHOLD * planned
        self.classifier_ = make_logistic().fit(X, early.astype(int))
        self.early_model_ = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, max_features=0.5,
            random_state=self.random_state, n_jobs=self.n_jobs,
        ).fit(X[early], y[early] / planned[early])
        return self

    def predict(self, X: pd.DataFrame):
        planned = X["planned_minutes"].to_numpy(dtype=float)
        p_early = self.classifier_.predict_proba(X)[:, 1]
        early_minutes = np.clip(self.early_model_.predict(X), 0, 1) * planned
        if self.point == "mean":
            return (1 - p_early) * planned + p_early * early_minutes
        return np.where(p_early < 0.5, planned, early_minutes)


def make_models(random_state: int = 0, n_jobs: int = 1) -> dict[str, BaseEstimator]:
    """``n_jobs`` defaults to 1: joblib's parallel workers deadlock under gevent
    monkey-patching (which the web app applies), so parallelism must be opted into
    from a plain process such as the training CLI."""
    # Linear models need explicit preprocessing (one-hot, log, imputation, scaling, shrinkage);
    # see ml.pipelines. It is fitted inside the pipeline, so only on training rows.
    linear = Pipeline([("prep", make_preprocessor()), ("model", LinearRegression())])
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
        "hurdle_mean": HurdleRegressor("mean", random_state, n_jobs),
        "hurdle_median": HurdleRegressor("median", random_state, n_jobs),
    }


def is_baseline(name: str) -> bool:
    return name.startswith("baseline_")
