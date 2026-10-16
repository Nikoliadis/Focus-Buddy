"""Clustering: user features, recovering known structure, refusing to invent structure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from analytics import cleaning
from ml import synthetic
from ml.clustering import analyze
from ml.clustering import train as ctr
from ml.clustering.features import USER_FEATURES, build_user_features
from ml.data import InsufficientDataError

ROOMS = pd.DataFrame({"room_id": [1, 2], "is_private": [False, True]})


def make_logs(rows):
    """rows: (user, local Athens start, focused minutes, planned minutes, room)."""
    records = []
    for i, (user, start, minutes, planned, room) in enumerate(rows, start=1):
        t = pd.Timestamp(start, tz="Europe/Athens").tz_convert("UTC")
        records.append({
            "log_id": i, "user_id": user, "room_id": room, "session_id": i, "focused_seconds": minutes * 60,
            "logged_at": t + pd.Timedelta(minutes=minutes), "session_duration_seconds": planned * 60,
            "session_started_at": t, "session_ended_at": t, "session_paused_seconds": 0, "session_started_by": user,
        })
    logs, _ = cleaning.clean_focus_logs(pd.DataFrame(records), now=pd.Timestamp("2026-12-31", tz="UTC"))
    return logs


@pytest.fixture(scope="module")
def synthetic_users():
    logs, rooms, truth = synthetic.generate(n_users=60, days=60, seed=8, return_users=True)
    return build_user_features(logs, rooms), truth.set_index("user_id")


class TestUserFeatures:
    def test_hand_checked(self):
        rows = [
            (1, "2026-03-02 09:00", 25, 25, 1),   # Mon morning, full
            (1, "2026-03-03 19:00", 10, 25, 1),   # Tue evening, early
            (1, "2026-03-07 23:30", 50, 50, 2),   # Sat night, full, private, long
            (1, "2026-03-08 14:00", 25, 25, 1),   # Sun afternoon, full
            (1, "2026-03-16 09:00", 25, 25, 1),   # Mon morning, two weeks after the first
        ]
        u = build_user_features(make_logs(rows), ROOMS).loc[1]

        assert u["n_sessions"] == 5
        assert u["sessions_per_week"] == pytest.approx(5 / 2)
        assert u["mean_focus_minutes"] == pytest.approx(135 / 5)
        assert u["full_duration_rate"] == pytest.approx(4 / 5)
        assert u["long_session_share"] == pytest.approx(1 / 5)
        assert (u["morning_share"], u["evening_share"], u["night_share"]) == pytest.approx((0.4, 0.2, 0.2))
        assert u["weekend_share"] == pytest.approx(2 / 5)
        assert u["distinct_rooms"] == 2
        assert u["private_room_share"] == pytest.approx(1 / 5)

    def test_short_activity_span_counts_as_one_week(self):
        rows = [(1, f"2026-03-02 {h:02d}:00", 25, 25, 1) for h in range(8, 13)]  # 5 sessions in one day
        assert build_user_features(make_logs(rows), ROOMS).loc[1, "sessions_per_week"] == 5

    def test_users_with_few_sessions_are_excluded(self):
        rows = [(1, f"2026-03-0{d} 09:00", 25, 25, 1) for d in range(1, 7)] + [(2, "2026-03-02 09:00", 25, 25, 1)]
        users = build_user_features(make_logs(rows), ROOMS, min_sessions=5)
        assert users.index.tolist() == [1]

    def test_empty(self):
        users = build_user_features(make_logs([(1, "2026-03-02 09:00", 25, 25, 1)]).iloc[0:0], ROOMS)
        assert users.empty and set(USER_FEATURES) <= set(users.columns)


class TestRecoversKnownStructure:
    def test_finds_the_hidden_chronotypes(self, synthetic_users):
        users, truth = synthetic_users
        report, _, labels = ctr.fit_clusters(users, "synthetic")

        assert report.k == 2
        ari = adjusted_rand_score(truth.loc[labels.index, "chronotype"], labels)
        assert ari > 0.9
        assert report.stability_ari > 0.9
        assert not report.weak_structure

    def test_continuous_traits_do_not_become_clusters(self, synthetic_users):
        """Discipline is continuous in the generator: the clusters should not split on it."""
        users, truth = synthetic_users
        _, _, labels = ctr.fit_clusters(users, "synthetic")
        means = truth.loc[labels.index, "discipline"].groupby(labels).mean()
        assert means.max() - means.min() < 0.1

    def test_scaling_is_what_makes_it_work(self):
        """Structure lives in a 0-1 share; noise lives in a 0-20 count. Unscaled KMeans clusters the noise."""
        rng = np.random.default_rng(0)
        group = np.repeat([0, 1], 50)
        X = pd.DataFrame({
            "morning_share": np.where(group == 0, 0.9, 0.1) + rng.normal(0, 0.03, 100),
            "sessions_per_week": rng.uniform(0, 20, 100),
        })

        raw = KMeans(2, n_init=10, random_state=0).fit_predict(X)
        scaled = ctr.make_pipeline(2).fit(X).predict(X)

        assert adjusted_rand_score(group, raw) < 0.2
        assert adjusted_rand_score(group, scaled) > 0.9

    def test_scaling_also_amplifies_useless_features(self):
        """The flip side: after scaling, a feature of tiny pure noise weighs as much as the signal.
        Nine such features drown one informative feature, so feature choice matters."""
        rng = np.random.default_rng(0)
        group = np.repeat([0, 1], 50)
        X = pd.DataFrame({f"noise_{i}": rng.uniform(0, 0.05, 100) for i in range(9)})
        X["morning_share"] = np.where(group == 0, 0.9, 0.1) + rng.normal(0, 0.03, 100)

        scaled = ctr.make_pipeline(2).fit(X).predict(X)
        signal_only = ctr.make_pipeline(2).fit(X[["morning_share"]]).predict(X[["morning_share"]])

        assert adjusted_rand_score(group, scaled) < 0.5
        assert adjusted_rand_score(group, signal_only) > 0.9


class TestRefusesToInventStructure:
    def test_pure_noise_is_flagged_as_weak(self):
        rng = np.random.default_rng(1)
        noise = pd.DataFrame(rng.uniform(size=(120, len(USER_FEATURES))), columns=list(USER_FEATURES))

        report, _, _ = ctr.fit_clusters(noise, "synthetic")

        assert report.weak_structure
        assert "WARNING: weak or unstable structure" in analyze.report_text(
            report, noise, pd.Series(np.zeros(120, dtype=int), index=noise.index))

    def test_too_few_users(self, synthetic_users):
        users, _ = synthetic_users
        with pytest.raises(InsufficientDataError):
            ctr.fit_clusters(users.head(10), "synthetic")

    def test_missing_values_rejected(self, synthetic_users):
        users, _ = synthetic_users
        broken = users.copy()
        broken.iloc[0, 1] = np.nan
        with pytest.raises(ValueError, match="missing"):
            ctr.fit_clusters(broken, "synthetic")


class TestDescriptions:
    def test_measurable_language_only(self, synthetic_users):
        users, _ = synthetic_users
        _, _, labels = ctr.fit_clusters(users, "synthetic")

        texts = " ".join(analyze.describe(users, labels).values()).lower()

        assert "morning share" in texts and "sd" in texts and "overall" in texts
        for persona in ("owl", "lark", "lazy", "procrastinat", "disciplined", "motivated", "type"):
            assert persona not in texts

    def test_standardised_differences_hand_checked(self):
        users = pd.DataFrame({f: [0.0, 0.0, 1.0, 1.0] for f in USER_FEATURES})
        labels = pd.Series([0, 0, 1, 1])
        diffs = analyze.standardised_differences(users, labels)
        sd = np.std([0, 0, 1, 1], ddof=1)
        assert diffs.loc[1, "morning_share"] == pytest.approx(0.5 / sd)

    def test_profiles_include_overall_row_and_sizes(self, synthetic_users):
        users, _ = synthetic_users
        _, _, labels = ctr.fit_clusters(users, "synthetic")
        profiles = analyze.cluster_profiles(users, labels)

        assert profiles.loc["all users", "users"] == len(users)
        assert profiles.drop(index="all users")["users"].sum() == len(users)

    def test_heatmap_does_not_exaggerate_trivial_differences(self, synthetic_users):
        """Colour = SD across users; sessions/week barely differs between clusters, so it stays near 0."""
        users, _ = synthetic_users
        _, _, labels = ctr.fit_clusters(users, "synthetic")

        fig = analyze.profile_heatmap(users, labels)
        z = analyze.standardised_differences(users, labels)

        assert isinstance(fig, Figure)
        assert z["sessions_per_week"].abs().max() < 0.5
        assert z["morning_share"].abs().max() > 0.8
