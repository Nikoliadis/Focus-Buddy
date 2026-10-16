"""Preprocessing pipelines for linear models.

Every transformation that *learns something from data* (a median, a mean and SD, the set
of categories, a prior) lives inside a scikit-learn ``Pipeline``. ``Pipeline.fit`` learns
those values from the rows it is given and nothing else, so within cross-validation each
fold's preprocessing sees only that fold's training data. Fitting a scaler or imputer on
the whole dataset before splitting leaks test-set statistics into training.

Tree models do not need this preprocessing: splits are invariant to monotone transforms
and scikit-learn forests handle missing values natively.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from ml.features import FEATURE_SPECS

CATEGORICAL = [s.name for s in FEATURE_SPECS if s.kind == "categorical"]
# `is_weekend` is left out: with weekday one-hot encoded it equals weekday_5 + weekday_6,
# an exact linear combination of other columns (perfect multicollinearity).
BINARY = [s.name for s in FEATURE_SPECS if s.kind == "binary" and s.name != "is_weekend"]
# Counts and durations with a long right tail: log1p makes their effect closer to linear
# (the 1st vs 2nd session matters more than the 100th vs 101st).
SKEWED = ["prev_sessions", "minutes_since_prev", "sessions_7d_before", "user_tenure_days", "room_age_days"]
RATE = "prev_full_duration_rate"
CYCLIC = ["hour_sin", "hour_cos"]
MIN_CATEGORY_COUNT = 20
NUMERIC = [s.name for s in FEATURE_SPECS
           if s.kind == "numeric" and s.name not in (*SKEWED, RATE, *CYCLIC)]


class ShrunkRate(TransformerMixin, BaseEstimator):
    """Shrink a per-user rate towards the population rate when the user has little history.

        shrunk = (rate · n + prior · strength) / (n + strength)

    After 1 session a user's rate is 0 or 1, which is mostly noise; after 50 it is reliable.
    ``strength`` is how many "virtual" sessions at the prior rate are mixed in. This is the
    posterior mean of a Beta-Binomial model. The ``prior`` is learned in ``fit``, so it
    comes from training data only.

    Input columns: [rate, n]. Output: one column.
    """

    def __init__(self, strength: float = 5.0):
        self.strength = strength

    def fit(self, X, y=None):
        rate, n = self._split(X)
        has_history = n > 0
        weights = n[has_history]
        self.prior_ = float(np.sum(rate[has_history] * weights) / np.sum(weights)) if weights.sum() else 0.5
        self.n_features_in_ = 2
        return self

    def transform(self, X):
        check_is_fitted(self, "prior_")
        rate, n = self._split(X)
        rate = np.where(np.isnan(rate), self.prior_, rate)
        n = np.where(np.isnan(n), 0, n)
        return ((rate * n + self.prior_ * self.strength) / (n + self.strength)).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{RATE}_shrunk"], dtype=object)

    @staticmethod
    def _split(X) -> tuple[np.ndarray, np.ndarray]:
        arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        return arr[:, 0], arr[:, 1]


def _impute_scale() -> Pipeline:
    # No missing-value indicators: every history feature is missing exactly when
    # `is_first_session` = 1, so each indicator would duplicate that column.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def make_preprocessor(hour_encoding: Literal["onehot", "cyclic"] = "onehot") -> Pipeline:
    """Column-wise preprocessing for linear models; returns a pandas DataFrame with named columns.

    hour_encoding="onehot": one weight per hour, can fit any daily shape (24 parameters)
    hour_encoding="cyclic": sin/cos, a single smooth daily wave (2 parameters)
    """
    categorical = CATEGORICAL if hour_encoding == "onehot" else [c for c in CATEGORICAL if c != "hour"]
    transformers = [
        # drop="first" avoids perfect collinearity with the intercept (the "dummy variable trap").
        # Categories seen fewer than MIN_CATEGORY_COUNT times in training share one "infrequent"
        # column, and a category never seen in training goes there too. With plain
        # handle_unknown="ignore" an unseen hour would become all zeros, which is exactly how
        # the reference hour is encoded, silently treating 03:00 as if it were 00:00.
        ("categorical", OneHotEncoder(drop="first", min_frequency=MIN_CATEGORY_COUNT,
                                      handle_unknown="infrequent_if_exist", sparse_output=False), categorical),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
        ("numeric", _impute_scale(), NUMERIC),
        ("skewed", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), SKEWED),
        ("rate", Pipeline([("shrink", ShrunkRate()), ("scale", StandardScaler())]), [RATE, "prev_sessions"]),
    ]
    if hour_encoding == "cyclic":
        transformers.append(("cyclic", "passthrough", CYCLIC))
    columns = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([
        ("columns", columns),
        # A column constant in the training data (e.g. no private rooms yet) is collinear with
        # the intercept and carries no information; it is dropped, decided on training data only.
        ("drop_constant", VarianceThreshold(threshold=0.0)),
    ]).set_output(transform="pandas")
