"""Hyperparameter tuning: temporal splitter, paired 1-SE rule, searches, logging, no test peeking."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from ml import synthetic, tuning
from ml.classification.model import ended_early
from ml.data import model_splits
from ml.evaluation.cv import PurgedTemporalSplit
from ml.features import FEATURE_COLUMNS, build_feature_table
from ml.pipelines import make_preprocessor


@pytest.fixture(scope="module")
def table():
    logs, rooms = synthetic.generate(n_users=25, days=45, seed=21)
    return build_feature_table(logs, rooms)


@pytest.fixture(scope="module", autouse=True)
def small_forest_space():
    """Keep test searches tiny (module-wide, so class-scoped fixtures see it too)."""
    small = tuning.SearchSpace(
        grid={"min_samples_leaf": [5, 50], "max_depth": [None, 4], "max_features": [0.5]},
        complexity=tuning.SPACES[("classification", "random_forest")].complexity,
        rationale="test space", fixed={"n_estimators": 30},
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(tuning.SPACES, ("classification", "random_forest"), small)
        yield


class TestPurgedTemporalSplit:
    def test_positions_respect_time_and_purge(self, table):
        cv = PurgedTemporalSplit(table["as_of"], table["label_available_at"], n_splits=4)

        assert cv.get_n_splits() == 4
        for train_idx, val_idx in cv.split(table):
            start = table["as_of"].iloc[val_idx].min()
            assert (table["as_of"].iloc[train_idx] < start).all()
            assert (table["label_available_at"].iloc[train_idx] < start).all()

    def test_rejects_misaligned_X(self, table):
        cv = PurgedTemporalSplit(table["as_of"], table["label_available_at"])
        with pytest.raises(ValueError, match="same rows"):
            list(cv.split(table.head(10)))

    def test_works_inside_gridsearchcv(self, table):
        cv = PurgedTemporalSplit(table["as_of"], table["label_available_at"], n_splits=3)
        pipeline = Pipeline([("prep", make_preprocessor()), ("model", LogisticRegression(max_iter=500))])

        search = GridSearchCV(pipeline, {"model__C": [0.1, 1.0]}, cv=cv, scoring="average_precision")
        search.fit(table[list(FEATURE_COLUMNS)], ended_early(table))

        assert len(search.cv_results_["params"]) == 2
        assert "split2_test_score" in search.cv_results_


def frame(candidates: dict[str, list[float]], complexity: dict[str, int]) -> pd.DataFrame:
    rows = []
    for name, folds in candidates.items():
        folds = np.asarray(folds)
        rows.append({"params": {"name": name, "c": complexity[name]}, "mean": folds.mean(),
                     "std": folds.std(), "se": folds.std() / np.sqrt(folds.size), "fold_scores": folds})
    return pd.DataFrame(rows)


class TestOneStandardErrorRule:
    complexity = staticmethod(lambda p: p["c"])

    def test_picks_simplest_equivalent(self):
        results = frame(
            {"complex": [0.50, 0.40, 0.60], "simple": [0.51, 0.39, 0.60], "bad": [0.30, 0.20, 0.40]},
            {"complex": 2, "simple": 1, "bad": 0},
        )
        chosen = tuning.one_standard_error_choice(results, self.complexity)
        assert results.loc[chosen, "params"]["name"] == "simple"

    def test_paired_rule_rejects_a_consistently_worse_setting(self):
        """Folds differ hugely in difficulty, so each setting's own SE is large; but 'worse'
        loses on EVERY fold. Unpaired 1-SE would accept it, the paired rule does not."""
        best = [0.80, 0.20, 0.50, 0.65, 0.35]
        worse = [b - 0.03 for b in best]
        results = frame({"best": best, "worse": worse}, {"best": 1, "worse": 0})

        unpaired_threshold = results.loc[0, "mean"] - results.loc[0, "se"]
        assert results.loc[1, "mean"] >= unpaired_threshold  # unpaired would call them equivalent
        chosen = tuning.one_standard_error_choice(results, self.complexity)
        assert results.loc[chosen, "params"]["name"] == "best"

    def test_best_is_always_eligible(self):
        results = frame({"only": [0.4, 0.5]}, {"only": 3})
        assert tuning.one_standard_error_choice(results, self.complexity) == 0


class TestSearches:
    @pytest.fixture(scope="class")
    def result(self, table):
        return tuning.tune(table, "classification", "random_forest", "grid", n_splits=3)

    def test_result_contents(self, result):
        assert result.n_candidates == 4 and result.n_folds == 3
        assert 0 <= result.selected_index < len(result.results)
        assert len(result.default_cv["fold_scores"]) == 3
        assert {"selected", "default", "diff", "diff_lo", "diff_hi"} <= set(result.test)
        mean, se = result.paired_gain
        assert result.beyond_noise == (mean > 2 * se)

    def test_report_text(self, result):
        text = result.to_text()
        assert "SYNTHETIC DATA" in text
        assert "paired 1-SE rule" in text
        assert "Default configuration" in text and "Held-out test" in text

    def test_log_is_appended_as_jsonl(self, result, tmp_path):
        path = tmp_path / "tuning.jsonl"
        result.log(path)
        result.log(path)

        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(records) == 2
        assert {"git_commit", "space", "selected", "default", "test", "data_source"} <= set(records[0])

    def test_random_search(self, table):
        result = tuning.tune(table, "classification", "random_forest", "random", n_iter=3, n_splits=3,
                             evaluate_test=False)
        assert result.n_candidates == 3 and not result.test

    def test_optuna_never_refits_a_repeated_configuration(self, table):
        result = tuning.tune(table, "classification", "random_forest", "optuna", n_iter=12, n_splits=3,
                             evaluate_test=False)
        keys = result.results["params"].map(lambda p: tuple(sorted(p.items())))
        assert keys.is_unique
        assert result.n_candidates <= 4  # the test space only has 4 combinations

    def test_unknown_method_or_space(self, table):
        with pytest.raises(ValueError):
            tuning.tune(table, "classification", "random_forest", "bayes")
        with pytest.raises(ValueError, match="no search space"):
            tuning.tune(table, "regression", "logistic_regression")

    def test_test_labels_never_influence_the_choice(self, table):
        """Scrambling the held-out test labels must not change any search decision."""
        splits = model_splits(table)
        scrambled = table.copy()
        rows = scrambled["log_id"].isin(splits.test["log_id"])
        scrambled.loc[rows, "target_full_duration"] = np.random.default_rng(0).integers(0, 2, rows.sum())

        a = tuning.tune(table, "classification", "random_forest", "grid", n_splits=3, evaluate_test=False)
        b = tuning.tune(scrambled, "classification", "random_forest", "grid", n_splits=3, evaluate_test=False)

        assert a.selected["params"] == b.selected["params"]
        np.testing.assert_allclose(a.results["mean"], b.results["mean"])


def test_hour_encoding_helper():
    assert tuning.hour_encoding(make_preprocessor("cyclic")) == "cyclic"
    assert tuning.hour_encoding(make_preprocessor("onehot")) == "onehot"


def test_real_spaces_are_documented_and_sane():
    for (task, model), space in tuning.SPACES.items():
        assert space.rationale
        assert tuning.grid_size(task, model) >= 2
        if model == "random_forest":
            assert "n_estimators" not in space.grid  # more trees never overfit: not a tuning target
