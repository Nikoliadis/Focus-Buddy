"""Evaluation framework: CV schemes, fit diagnosis, bootstrap CIs, evaluation cards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from ml import synthetic
from ml.evaluation import cv
from ml.evaluation.card import evaluate
from ml.evaluation.diagnostics import bootstrap_ci, diagnose_fit, learning_curve, paired_bootstrap_difference
from ml.evaluation.tasks import get_task
from ml.features import FEATURE_COLUMNS, build_feature_table
from ml.regression.model import PlannedMinutesBaseline


@pytest.fixture(scope="module")
def table():
    logs, rooms = synthetic.generate(n_users=25, days=45, seed=11)
    return build_feature_table(logs, rooms)


class TestCrossValidationSchemes:
    def test_temporal_folds_never_train_on_the_future(self, table):
        folds = list(cv.temporal_folds(table, n_splits=4))

        assert len(folds) == 4
        for train, validation in folds:
            assert train["as_of"].max() < validation["as_of"].min()
            assert (train["label_available_at"] < validation["as_of"].min()).all()  # purged
        sizes = [len(train) for train, _ in folds]
        assert sizes == sorted(sizes)  # expanding window

    def test_user_folds_keep_each_user_on_one_side(self, table):
        seen = set()
        for train, validation in cv.user_folds(table, n_splits=5):
            assert not set(train["user_id"]) & set(validation["user_id"])
            seen |= set(validation["user_id"])
        assert seen == set(table["user_id"])  # every user validated exactly once

    def test_cross_validate_reports_train_and_validation(self, table):
        task = get_task("regression")
        folds = cv.cross_validate(PlannedMinutesBaseline(), table, task, "temporal", 3)

        assert len(folds) == 3
        assert {"train_mae", "val_mae", "n_train", "n_validation"} <= set(folds.columns)
        summary = cv.summarise(folds, "mae")
        assert summary["folds"] == 3
        assert summary["min"] <= summary["mean"] <= summary["max"]

    def test_classification_folds_skip_single_class(self):
        task = get_task("classification")
        one_class = pd.DataFrame({
            "as_of": pd.date_range("2026-01-01", periods=400, freq="h", tz="UTC"),
            "label_available_at": pd.date_range("2026-01-01 00:30", periods=400, freq="h", tz="UTC"),
            "user_id": np.arange(400) % 20, "target_full_duration": 1,
            **{c: 0.0 for c in FEATURE_COLUMNS},
        })
        assert cv.cross_validate(task.make_models()["baseline_majority"], one_class, task).empty


class TestFitDiagnosis:
    task = get_task("regression")

    def test_underfitting(self):
        d = diagnose_fit(self.task, train_score=6.0, val_score=6.1, baseline_val_score=6.0)
        assert d.verdict == "underfitting"

    def test_overfitting(self):
        d = diagnose_fit(self.task, train_score=1.0, val_score=5.0, baseline_val_score=6.0)
        assert d.verdict == "overfitting"
        assert d.relative_gap == pytest.approx(0.8)

    def test_reasonable(self):
        assert diagnose_fit(self.task, 4.5, 4.8, 6.0).verdict == "reasonable fit"

    def test_higher_is_better_metrics_are_flipped(self):
        clf = get_task("classification")
        assert diagnose_fit(clf, train_score=0.95, val_score=0.40, baseline_val_score=0.30).verdict == "overfitting"
        assert diagnose_fit(clf, train_score=0.31, val_score=0.29, baseline_val_score=0.30).verdict == "underfitting"

    def test_an_unpruned_tree_is_diagnosed_as_overfitting(self, table):
        """A depth-unlimited tree memorises the training sessions: tiny train error, worse validation."""
        task = get_task("regression")
        folds = cv.cross_validate(DecisionTreeRegressor(random_state=0), table, task, "temporal", 3)
        baseline = cv.cross_validate(PlannedMinutesBaseline(), table, task, "temporal", 3)

        d = diagnose_fit(task, folds["train_mae"].mean(), folds["val_mae"].mean(), baseline["val_mae"].mean())

        assert folds["train_mae"].mean() < 1
        assert d.verdict in ("overfitting", "underfitting")
        assert d.relative_gap > 0.5

    def test_learning_curve(self, table):
        task = get_task("regression")
        ordered = table.sort_values("as_of")
        fit, validation = ordered.iloc[:800], ordered.iloc[800:1000]

        curve = learning_curve(PlannedMinutesBaseline(), fit, validation, task, fractions=(0.25, 0.5, 1.0))

        assert curve["n_train"].tolist() == [200, 400, 800]
        assert curve["validation"].nunique() == 1  # a rule that ignores training data cannot improve


class TestBootstrap:
    def test_ci_contains_estimate_and_narrows_with_n(self):
        rng = np.random.default_rng(0)
        small_y, big_y = rng.normal(size=50), rng.normal(size=5000)
        small = bootstrap_ci(small_y, small_y + rng.normal(size=50), mean_absolute_error)
        big = bootstrap_ci(big_y, big_y + rng.normal(size=5000), mean_absolute_error)

        assert small[1] <= small[0] <= small[2]
        assert (big[2] - big[1]) < (small[2] - small[1]) / 5

    def test_paired_difference_detects_a_real_gap(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=2000)
        good, bad = y + rng.normal(0, 0.5, 2000), y + rng.normal(0, 1.0, 2000)

        diff, lo, hi = paired_bootstrap_difference(y, good, bad, mean_absolute_error)

        assert diff < 0 and hi < 0  # good has lower MAE, CI excludes 0

    def test_paired_difference_of_identical_models_is_zero(self):
        y = np.arange(100.0)
        assert paired_bootstrap_difference(y, y + 1, y + 1, mean_absolute_error) == (0, 0, 0)


class TestEvaluationCard:
    @pytest.fixture(scope="class")
    def card(self, table):
        return evaluate(table, "classification", synthetic.SYNTHETIC, n_splits=3)

    def test_contains_every_required_section(self, card):
        md = card.to_markdown()
        for section in ("## Data", "## Split (by time)", "## Test performance", "## Cross-validation",
                        "## Fit diagnosis", "## Limitations"):
            assert section in md
        assert "SYNTHETIC DATA" in md
        assert card.dataset["rows"] > 0 and len(card.features) == 17

    def test_model_and_baseline_chosen_on_validation(self, card):
        assert not card.model.startswith("baseline_")
        assert card.baseline.startswith("baseline_")

    def test_significance_follows_the_ci(self, card):
        _, lo, hi = card.difference_vs_baseline_ci
        assert card.significantly_better_than_baseline == (lo > 0)

    def test_limitations_mention_imbalance_and_source(self, card):
        text = " ".join(card.limitations)
        assert "synthetic" in text and "imbalance" in text.lower() and "not causes" in text

    def test_regression_card_and_save(self, table, tmp_path):
        card = evaluate(table, "regression", synthetic.SYNTHETIC, n_splits=3)
        path = card.save(tmp_path)

        assert path.suffix == ".md" and path.with_suffix(".json").exists()
        assert card.primary_metric == "mae" and not card.higher_is_better
        assert "| MAE |" in path.read_text(encoding="utf-8")

    def test_unknown_task(self, table):
        with pytest.raises(ValueError):
            evaluate(table, "ranking", "synthetic")
