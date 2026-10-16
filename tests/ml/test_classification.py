"""Classification: metrics, imbalance, threshold choice, protocol, persistence, hurdle model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import synthetic
from ml.classification import predict as cp
from ml.classification import train as ct
from ml.classification.model import UserHistoryClassifier, ended_early, make_logistic
from ml.data import InsufficientDataError, model_splits
from ml.evaluation.metrics import best_f1_threshold, classification_metrics
from ml.features import FEATURE_COLUMNS, build_feature_table
from ml.regression.model import HurdleRegressor


@pytest.fixture(scope="module")
def table():
    logs, rooms = synthetic.generate(n_users=30, days=50, seed=5)
    return build_feature_table(logs, rooms)


@pytest.fixture(scope="module")
def trained(table):
    return ct.train_and_evaluate(table, synthetic.SYNTHETIC)


class TestMetrics:
    def test_confusion_and_rates_hand_checked(self):
        y = [1, 1, 1, 0, 0, 0, 0, 0]
        p = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1]

        m = classification_metrics(y, p, threshold=0.5)

        assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["accuracy"] == pytest.approx(6 / 8)

    def test_accuracy_paradox(self):
        """Predicting 'never ends early' scores 72% accuracy and catches nothing."""
        y = np.array([1] * 28 + [0] * 72)
        m = classification_metrics(y, np.zeros(100))

        assert m["accuracy"] == pytest.approx(0.72)
        assert m["recall"] == 0 and m["f1"] == 0

    def test_chance_levels(self):
        y = np.array([1] * 30 + [0] * 70)
        m = classification_metrics(y, np.full(100, 0.3))
        assert m["roc_auc"] == pytest.approx(0.5)
        assert m["pr_auc"] == pytest.approx(0.3)  # = positive rate

    def test_single_class_has_undefined_auc(self):
        m = classification_metrics([0, 0, 0], [0.1, 0.2, 0.3])
        assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])

    def test_threshold_maximises_f1(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.25).astype(int)
        p = np.clip(0.25 + 0.3 * (y - 0.25) + rng.normal(0, 0.15, 500), 0, 1)

        t = best_f1_threshold(y, p)

        f1_best = classification_metrics(y, p, t)["f1"]
        for other in np.linspace(0.05, 0.95, 19):
            assert f1_best >= classification_metrics(y, p, other)["f1"] - 1e-12


class TestModels:
    def test_target_is_ended_early(self):
        t = pd.DataFrame({"target_full_duration": [1, 0, 1]})
        assert ended_early(t).tolist() == [0, 1, 0]

    def test_user_history_classifier(self):
        X = pd.DataFrame({"prev_full_duration_rate": [0.8, np.nan, 0.0]})
        clf = UserHistoryClassifier().fit(X, [0, 1, 1, 0])

        np.testing.assert_allclose(clf.predict_proba(X)[:, 1], [0.2, 0.5, 1.0])

    def test_class_weight_balanced_inflates_probabilities(self, table):
        """Reweighting helps recall at 0.5 but distorts probabilities: they stop meaning frequencies."""
        X, y = table[list(FEATURE_COLUMNS)], ended_early(table)
        plain = make_logistic().fit(X, y).predict_proba(X)[:, 1]
        balanced = make_logistic(class_weight="balanced").fit(X, y).predict_proba(X)[:, 1]

        assert plain.mean() == pytest.approx(y.mean(), abs=0.01)   # calibrated in the large
        assert balanced.mean() > y.mean() + 0.1


class TestProtocol:
    def test_selection_on_validation_pr_auc(self, trained):
        report, _ = trained
        ml_models = {n: s for n, s in report.validation.items() if not n.startswith("baseline_")}
        assert report.selected_model == max(ml_models, key=lambda n: ml_models[n]["pr_auc"])

    def test_test_labels_never_influence_decisions(self, table, trained):
        """Scramble the test labels: model choice and thresholds must not change."""
        report, _ = trained
        splits = model_splits(table)
        scrambled = table.copy()
        test_rows = scrambled["log_id"].isin(splits.test["log_id"])
        rng = np.random.default_rng(1)
        scrambled.loc[test_rows, "target_full_duration"] = rng.integers(0, 2, test_rows.sum())

        again, _ = ct.train_and_evaluate(scrambled, synthetic.SYNTHETIC)

        assert again.selected_model == report.selected_model
        assert again.thresholds == pytest.approx(report.thresholds)
        assert again.test != report.test  # only the reported test scores move

    def test_beats_baselines_and_majority_is_at_chance(self, trained):
        report, _ = trained
        assert report.beats_baselines
        assert report.test["baseline_majority"]["roc_auc"] == pytest.approx(0.5)
        assert report.test["baseline_majority"]["recall"] == 0

    def test_tuned_threshold_finds_more_early_endings(self, trained):
        report, _ = trained
        assert report.threshold < 0.5   # minority class: the F1-optimal cut is below 0.5
        assert report.test[report.selected_model]["recall"] > report.test_at_default_threshold["recall"]

    def test_logistic_recovers_ground_truth_directions(self, trained):
        """Generator: disciplined users, short sessions and weekdays finish more often."""
        c = trained[0].coefficients
        assert c["prev_full_duration_rate_shrunk"] < 0
        assert c["planned_minutes"] > 0
        # weekday is one-hot with Monday as reference: Saturday and Sunday raise the odds
        assert c["weekday_5"] > 0 and c["weekday_6"] > 0

    def test_single_class_refuses_to_train(self, table):
        one_class = table.assign(target_full_duration=1)
        with pytest.raises(InsufficientDataError, match="only one class"):
            ct.train_and_evaluate(one_class, "synthetic")

    def test_report_text(self, trained):
        text = trained[0].to_text()
        assert "SYNTHETIC DATA" in text
        assert "Confusion matrix" in text
        assert "chance level" in text


class TestPersistence:
    def test_round_trip_keeps_threshold(self, trained, table, tmp_path):
        report, model = trained
        path = ct.save(model, report, tmp_path)

        loaded, meta = cp.load(path)
        out = cp.predict(loaded, table.tail(50), meta["threshold"])

        assert meta["threshold"] == pytest.approx(report.threshold)
        assert set(out.columns) == {"p_ended_early", "predicted_ended_early"}
        assert ((out["p_ended_early"] >= meta["threshold"]) == out["predicted_ended_early"].astype(bool)).all()

    def test_missing_features(self, trained, table):
        with pytest.raises(ValueError, match="missing feature columns"):
            cp.predict(trained[1], table.drop(columns=["hour_sin"]), 0.5)


class TestHurdle:
    @pytest.fixture(scope="class")
    def data(self, table):
        X, y = table[list(FEATURE_COLUMNS)], table["target_focused_minutes"]
        return X, y

    def test_predictions_stay_within_the_plan(self, data):
        X, y = data
        for point in ("mean", "median"):
            pred = HurdleRegressor(point).fit(X, y).predict(X)
            assert (pred >= 0).all() and (pred <= X["planned_minutes"] + 1e-9).all()

    def test_median_predicts_the_plan_when_early_ending_is_unlikely(self, data):
        X, y = data
        model = HurdleRegressor("median").fit(X, y)
        p_early = model.classifier_.predict_proba(X)[:, 1]
        pred = model.predict(X)

        likely_full = p_early < 0.5
        np.testing.assert_allclose(pred[likely_full], X["planned_minutes"].to_numpy()[likely_full])

    def test_mean_is_between_early_estimate_and_plan(self, data):
        X, y = data
        mean_pred = HurdleRegressor("mean").fit(X, y).predict(X)
        assert (mean_pred <= X["planned_minutes"]).all()
        assert mean_pred.mean() == pytest.approx(y.mean(), rel=0.05)

    def test_invalid_point(self, data):
        X, y = data
        with pytest.raises(ValueError):
            HurdleRegressor("mode").fit(X, y)
