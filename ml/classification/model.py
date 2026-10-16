"""Candidate classifiers for "will this session end early?" (positive class = ended early).

The positive class is the *minority* event we want to catch. Making it class 1 keeps
precision and recall meaningful: they describe how well we find early endings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.regression.model import LINEAR_FEATURES


def ended_early(table: pd.DataFrame) -> pd.Series:
    """Classification target: 1 if the session did not run its full planned length."""
    return (1 - table["target_full_duration"]).rename("ended_early").astype("int64")


class UserHistoryClassifier(ClassifierMixin, BaseEstimator):
    """P(ended early) = 1 - the user's past full-duration rate; the training prior for new users."""

    def fit(self, X: pd.DataFrame, y):
        self.classes_ = np.array([0, 1])
        self.prior_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = (1 - X["prev_full_duration_rate"]).fillna(self.prior_).clip(0, 1).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_logistic(class_weight: str | None = None, C: float = 1.0) -> Pipeline:
    """Scaled logistic regression on the linear-safe features (cyclic hour, no raw categories).

    Scaling matters twice here: the L2 penalty (strength 1/C) treats all coefficients alike,
    and standardised coefficients are comparable as "effect per 1 SD of the feature"."""
    return Pipeline([
        ("prep", ColumnTransformer(
            [("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), LINEAR_FEATURES)],
            remainder="drop",
            verbose_feature_names_out=False,
        )),
        ("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000)),
    ])


def make_models(random_state: int = 0, n_jobs: int = 1) -> dict[str, BaseEstimator]:
    """``n_jobs`` defaults to 1 (joblib deadlocks under gevent, see ml.regression.model)."""
    return {
        "baseline_majority": DummyClassifier(strategy="prior"),
        "baseline_user_history": UserHistoryClassifier(),
        "logistic_regression": make_logistic(),
        "logistic_regression_balanced": make_logistic(class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_features=0.5,
            n_jobs=n_jobs, random_state=random_state,
        ),
    }


def is_baseline(name: str) -> bool:
    return name.startswith("baseline_")


def logistic_coefficients(model: Pipeline) -> pd.Series:
    """Standardised coefficients (log-odds change per +1 SD), largest effect first."""
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(model.named_steps["model"].coef_[0], index=names)
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index)
