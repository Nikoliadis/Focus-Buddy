"""Preprocessing pipelines: correct transforms, no collinearity, and no preprocessing leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ml import synthetic
from ml.features import FEATURE_COLUMNS, build_feature_table
from ml.pipelines import SKEWED, ShrunkRate, make_preprocessor


@pytest.fixture(scope="module")
def X():
    logs, rooms = synthetic.generate(n_users=25, days=40, seed=3)
    return build_feature_table(logs, rooms)[list(FEATURE_COLUMNS)]


class TestPreprocessor:
    def test_output_is_named_complete_and_numeric(self, X):
        out = make_preprocessor().fit_transform(X)

        assert isinstance(out, pd.DataFrame)
        assert not out.isna().any().any()
        assert any(c.startswith("hour_") for c in out.columns)
        assert "prev_full_duration_rate_shrunk" in out.columns
        assert "is_weekend" not in out.columns  # = weekday_5 + weekday_6

    @pytest.mark.parametrize("encoding", ["onehot", "cyclic"])
    def test_design_matrix_has_full_rank(self, X, encoding):
        """No column is a linear combination of others (or of the intercept)."""
        Z = make_preprocessor(encoding).fit_transform(X).to_numpy()
        D = np.column_stack([np.ones(len(Z)), Z])
        assert np.linalg.matrix_rank(D) == D.shape[1]

    def test_cyclic_encoding_replaces_hour_dummies(self, X):
        cols = make_preprocessor("cyclic").fit_transform(X).columns
        assert {"hour_sin", "hour_cos"} <= set(cols)
        assert not any(c.startswith("hour_") and c not in ("hour_sin", "hour_cos") for c in cols)

    def test_statistics_come_from_training_rows_only(self, X):
        train, test = X.iloc[: len(X) // 2], X.iloc[len(X) // 2:]
        shifted_test = test.assign(planned_minutes=test["planned_minutes"] + 1000)

        prep = make_preprocessor().fit(train)
        scaler = prep.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
        idx = list(prep.named_steps["columns"].transformers_[2][2]).index("planned_minutes")

        assert scaler.mean_[idx] == pytest.approx(train["planned_minutes"].mean())
        prep.transform(shifted_test)  # transforming new data must not refit anything
        assert scaler.mean_[idx] == pytest.approx(train["planned_minutes"].mean())

    def test_unseen_category_is_not_the_reference_category(self, X):
        prep = make_preprocessor().fit(X)
        hours_seen = set(X["hour"])
        unseen = next(h for h in range(24) if h not in hours_seen)
        row = X.iloc[[0]].assign(hour=unseen)

        out = prep.transform(row)

        hour_cols = [c for c in out.columns if c.startswith("hour_")]
        assert out.loc[:, hour_cols].sum(axis=1).iloc[0] == 1  # not all zeros
        assert out["hour_infrequent_sklearn"].iloc[0] == 1

    def test_log_transform_on_skewed_features(self, X):
        out = make_preprocessor().fit_transform(X)
        # monotone transform: ordering of users by prev_sessions is preserved
        ranks_in = X["prev_sessions"].rank(method="dense")
        ranks_out = out["prev_sessions"].rank(method="dense")
        assert (ranks_in == ranks_out).all()
        assert set(SKEWED) <= set(out.columns)

    def test_constant_training_column_is_dropped(self, X):
        out = make_preprocessor().fit_transform(X.assign(room_is_private=0.0))
        assert "room_is_private" not in out.columns

    def test_is_cloneable(self):
        assert clone(make_preprocessor("cyclic")).get_params()["columns__cyclic"] == "passthrough"


class TestShrunkRate:
    def test_hand_checked(self):
        X = pd.DataFrame({"rate": [1.0, 0.0, 0.5, np.nan], "n": [1, 10, 4, 0]})
        t = ShrunkRate(strength=5).fit(X)

        # prior = (1·1 + 0·10 + 0.5·4) / 15 = 0.2
        assert t.prior_ == pytest.approx(0.2)
        out = t.transform(X).ravel()
        assert out[0] == pytest.approx((1 * 1 + 0.2 * 5) / 6)   # one lucky session pulled down
        assert out[1] == pytest.approx((0 * 10 + 0.2 * 5) / 15)
        assert out[3] == pytest.approx(0.2)                     # no history -> prior

    def test_more_history_means_less_shrinkage(self):
        X = pd.DataFrame({"rate": [1.0, 1.0], "n": [1, 100]})
        out = ShrunkRate(strength=5).fit(pd.DataFrame({"rate": [0.5], "n": [10]})).transform(X).ravel()
        assert out[0] < out[1] < 1

    def test_zero_strength_is_the_raw_rate(self):
        X = pd.DataFrame({"rate": [0.3, 0.9], "n": [3, 7]})
        np.testing.assert_allclose(ShrunkRate(strength=0).fit(X).transform(X).ravel(), [0.3, 0.9])

    def test_prior_is_learned_in_fit_only(self):
        t = ShrunkRate().fit(pd.DataFrame({"rate": [0.9], "n": [10]}))
        t.transform(pd.DataFrame({"rate": [0.1], "n": [1000]}))
        assert t.prior_ == pytest.approx(0.9)


def test_feature_selection_outside_the_pipeline_leaks():
    """Classic preprocessing leak. With pure-noise features and a random target, picking the
    'best' features on ALL rows before cross-validation looks predictive; doing the same
    selection inside the pipeline (so per fold, on training rows only) correctly shows chance."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2000))
    y = rng.integers(0, 2, 200)
    folds = StratifiedKFold(5, shuffle=True, random_state=0)

    X_selected = SelectKBest(f_classif, k=20).fit_transform(X, y)    # sees every row's label
    leaky = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y, cv=folds, scoring="roc_auc")

    safe_pipeline = Pipeline([("select", SelectKBest(f_classif, k=20)), ("model", LogisticRegression(max_iter=1000))])
    safe = cross_val_score(safe_pipeline, X, y, cv=folds, scoring="roc_auc")

    assert leaky.mean() > 0.75   # looks impressive, is noise
    assert safe.mean() < 0.62    # the honest answer: no signal
