"""Regression: metrics, baselines, the train/validation/test protocol, persistence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import synthetic
from ml.evaluation.metrics import regression_metrics
from ml.features import FEATURE_COLUMNS, build_feature_table
from ml.regression import predict as pr
from ml.regression import train as tr
from ml.regression.model import PlannedMinutesBaseline, UserHistoryBaseline, make_models


@pytest.fixture(scope="module")
def table():
    logs, rooms = synthetic.generate(n_users=25, days=45, seed=1)
    return build_feature_table(logs, rooms)


@pytest.fixture(scope="module")
def trained(table):
    return tr.train_and_evaluate(table, synthetic.SYNTHETIC)


class TestMetrics:
    def test_hand_checked(self):
        m = regression_metrics([10, 20, 30], [12, 18, 33])
        # errors 2, -2, 3 -> MAE 7/3, MSE 17/3; mean 20, SST = 200, R² = 1 - 17/200
        assert m["mae"] == pytest.approx(7 / 3)
        assert m["mse"] == pytest.approx(17 / 3)
        assert m["rmse"] == pytest.approx(np.sqrt(17 / 3))
        assert m["r2"] == pytest.approx(1 - 17 / 200)

    def test_predicting_the_mean_gives_r2_zero(self):
        y = np.array([5, 10, 15, 30.0])
        assert regression_metrics(y, np.full(4, y.mean()))["r2"] == pytest.approx(0)

    def test_rmse_is_at_least_mae(self):
        m = regression_metrics([0, 0, 0, 0], [1, 1, 1, 9])
        assert m["rmse"] >= m["mae"]


class TestBaselines:
    X = pd.DataFrame({"planned_minutes": [25.0, 50.0, 25.0], "prev_mean_minutes": [20.0, np.nan, 40.0]})

    def test_planned(self):
        assert PlannedMinutesBaseline().fit(self.X).predict(self.X).tolist() == [25, 50, 25]

    def test_user_history_scales_past_ratio_and_falls_back(self):
        # 20/25 of 25 = 20; no history -> 50; ratio capped at 1 -> 25
        assert UserHistoryBaseline().fit(self.X).predict(self.X).tolist() == [20, 50, 25]

    def test_all_models_fit_and_predict_with_missing_values(self, table):
        X = table[list(FEATURE_COLUMNS)]
        assert X.isna().any().any()  # cold-start NaNs are present
        for name, model in make_models().items():
            preds = model.fit(X, table["target_focused_minutes"]).predict(X.head(20))
            assert np.isfinite(preds).all(), name


class TestProtocol:
    def test_report_shape(self, trained, table):
        report, _ = trained

        assert report.n_rows == len(table)
        assert report.n_fit + report.n_validation + report.purged <= report.n_train + report.n_test + report.purged
        assert set(report.test) == set(make_models())
        assert report.data_source == "synthetic"

    def test_model_is_selected_on_validation_not_test(self, trained):
        report, _ = trained
        ml_models = {n: s for n, s in report.validation.items() if not n.startswith("baseline_")}

        assert report.selected_model == min(ml_models, key=lambda n: ml_models[n]["mae"])
        assert not report.selected_model.startswith("baseline_")

    def test_beats_baseline_flag_is_honest(self, trained):
        report, _ = trained
        sel, base = report.test[report.selected_model]["mae"], report.test[report.best_baseline]["mae"]
        assert report.beats_best_baseline == (sel < base)

    def test_models_learn_something_beyond_the_mean(self, trained):
        report, _ = trained
        assert report.test["linear_regression"]["r2"] > report.test["baseline_global_mean"]["r2"] + 0.2

    def test_recovers_the_dominant_signal(self, trained):
        """Ground truth: focused time is mostly planned time. The model should find that."""
        report, _ = trained
        assert next(iter(report.importances)) == "planned_minutes"

    def test_synthetic_banner(self, trained):
        report, _ = trained
        assert "SYNTHETIC DATA" in report.to_text()
        report.data_source = "db"
        assert "SYNTHETIC" not in report.to_text()
        report.data_source = synthetic.SYNTHETIC

    def test_refuses_to_train_on_too_little_data(self, table):
        with pytest.raises(tr.InsufficientDataError):
            tr.train_and_evaluate(table.head(100), "synthetic")


class TestPersistence:
    def test_round_trip_gives_identical_predictions(self, trained, table, tmp_path):
        report, model = trained
        path = tr.save(model, report, tmp_path)

        loaded, meta = pr.load(path)
        sample = table.tail(30)

        pd.testing.assert_series_equal(pr.predict(model, sample), pr.predict(loaded, sample))
        assert meta["selected_model"] == report.selected_model
        assert meta["data_source"] == "synthetic"
        assert meta["features"] == list(FEATURE_COLUMNS)

    def test_rejects_model_with_other_feature_set(self, trained, tmp_path):
        import json

        report, model = trained
        path = tr.save(model, report, tmp_path)
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        meta["features"] = ["something_else"]
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="different feature set"):
            pr.load(path)

    def test_predict_requires_all_features(self, trained, table):
        _, model = trained
        with pytest.raises(ValueError, match="missing feature columns"):
            pr.predict(model, table.drop(columns=["planned_minutes"]))


class TestSyntheticGenerator:
    def test_deterministic(self):
        a, _ = synthetic.generate(n_users=5, days=10, seed=3)
        b, _ = synthetic.generate(n_users=5, days=10, seed=3)
        pd.testing.assert_frame_equal(a, b)

    def test_sessions_of_a_user_never_overlap(self):
        logs, _ = synthetic.generate(n_users=10, days=30, seed=2)
        for _, g in logs.sort_values("session_started_at").groupby("user_id"):
            assert (g["session_started_at"].iloc[1:].to_numpy() >= g["logged_at"].iloc[:-1].to_numpy()).all()

    def test_all_logs_pass_quality_rules(self):
        logs, _ = synthetic.generate(n_users=10, days=30, seed=2)
        assert (logs["quality"] == "ok").all()

    def test_ground_truth_effects_are_present(self):
        """The generator encodes 'late night and long sessions finish less often'."""
        logs, rooms = synthetic.generate(n_users=60, days=60, seed=4)
        t = build_feature_table(logs, rooms)
        late = t["hour"].isin([23, 0, 1, 2, 3, 4])

        assert t.loc[late, "target_full_duration"].mean() < t.loc[~late, "target_full_duration"].mean()
        by_planned = t.groupby("planned_minutes")["target_full_duration"].mean()
        assert by_planned[50] < by_planned[25]
