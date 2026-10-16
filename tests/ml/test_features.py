"""Feature engineering: hand-checked values and, above all, no leakage from the future."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics import cleaning
from ml import features as f

NOW = pd.Timestamp("2026-06-01", tz="UTC")
ROOMS = pd.DataFrame({
    "room_id": [1, 2],
    "is_private": [False, True],
    "created_at": pd.to_datetime(["2026-02-01", "2026-03-01"], utc=True),
    "current_member_count": [5, 2],
})


def make_logs(rows: list[tuple]) -> pd.DataFrame:
    """rows: (user_id, UTC start, focused minutes[, planned minutes[, room_id]])."""
    records = []
    for i, row in enumerate(rows, start=1):
        user, start, focused = row[:3]
        planned = row[3] if len(row) > 3 else 25
        room = row[4] if len(row) > 4 else 1
        t = pd.Timestamp(start, tz="UTC")
        records.append({
            "log_id": i, "user_id": user, "room_id": room, "session_id": i,
            "focused_seconds": focused * 60, "logged_at": t + pd.Timedelta(minutes=focused),
            "session_duration_seconds": planned * 60, "session_started_at": t,
            "session_ended_at": t + pd.Timedelta(minutes=focused), "session_paused_seconds": 0,
            "session_started_by": user,
        })
    logs, _ = cleaning.clean_focus_logs(pd.DataFrame(records), now=NOW)
    return logs


def table_for(rows, rooms=ROOMS, tz="Europe/Athens"):
    return f.build_feature_table(make_logs(rows), rooms, tz=tz).set_index("log_id")


# One user, Athens = UTC+2 in early March.
USER_ROWS = [
    (1, "2026-03-02 07:00", 25),            # log 1: Mon 09:00 local, first ever
    (1, "2026-03-02 08:00", 10),            # log 2: same day, ended early
    (1, "2026-03-02 21:30", 25),            # log 3: Mon 23:30 local, still Monday
    (1, "2026-03-03 07:00", 50, 50),        # log 4: Tue, next local day
    (1, "2026-03-11 07:00", 25),            # log 5: 8 days later
]


class TestHandCheckedValues:
    def test_first_session_has_no_history(self):
        t = table_for(USER_ROWS)

        assert t.loc[1, "is_first_session"] == 1
        assert t.loc[1, "prev_sessions"] == 0
        assert np.isnan(t.loc[1, "prev_mean_minutes"])
        assert np.isnan(t.loc[1, "minutes_since_prev"])

    def test_history_aggregates(self):
        t = table_for(USER_ROWS)

        assert t.loc[4, "prev_sessions"] == 3
        assert t.loc[4, "prev_mean_minutes"] == pytest.approx((25 + 10 + 25) / 3)
        assert t.loc[4, "prev_full_duration_rate"] == pytest.approx(2 / 3)   # log 2 ended early
        assert t.loc[4, "prev_session_minutes"] == 25
        # log 3 ended 21:55 UTC, log 4 starts 07:00 UTC next day
        assert t.loc[4, "minutes_since_prev"] == pytest.approx(9 * 60 + 5)
        assert t.loc[2, "user_tenure_days"] == pytest.approx(35 / 60 / 24)  # first end 07:25, now 08:00

    def test_sessions_today_uses_local_day(self):
        t = table_for(USER_ROWS)

        assert t.loc[2, "sessions_today_before"] == 1
        assert t.loc[3, "sessions_today_before"] == 2   # 23:30 local is still Monday
        assert t.loc[4, "sessions_today_before"] == 0   # Tuesday

    def test_local_day_boundary_differs_from_utc(self):
        rows = [(1, "2026-03-02 20:00", 25), (1, "2026-03-02 22:30", 25)]  # 22:00 and 00:30 local
        athens = table_for(rows)
        utc = table_for(rows, tz="UTC")

        assert athens.loc[2, "sessions_today_before"] == 0  # a new day in Athens
        assert utc.loc[2, "sessions_today_before"] == 1     # same day in UTC

    def test_seven_day_window(self):
        t = table_for(USER_ROWS)
        assert t.loc[4, "sessions_7d_before"] == 3
        assert t.loc[5, "sessions_7d_before"] == 0  # 8 days after the last one

    def test_time_features(self):
        t = table_for(USER_ROWS)

        assert t.loc[1, "hour"] == 9
        assert t.loc[1, "weekday"] == 0
        assert t.loc[1, "is_weekend"] == 0
        assert t.loc[4, "planned_minutes"] == 50

    def test_cyclic_hour_encoding(self):
        t = table_for([(1, "2026-03-02 21:30", 25), (1, "2026-03-03 22:30", 25), (1, "2026-03-04 10:00", 25)])
        coords = t[["hour_sin", "hour_cos"]].to_numpy()

        late_vs_midnight = np.linalg.norm(coords[0] - coords[1])   # 23:00 vs 00:00 local
        late_vs_noon = np.linalg.norm(coords[0] - coords[2])       # 23:00 vs 12:00 local
        assert late_vs_midnight < late_vs_noon
        assert np.allclose(np.hypot(*coords.T), 1)

    def test_room_features(self):
        t = table_for([(1, "2026-03-02 07:00", 25, 25, 1), (1, "2026-03-05 07:00", 25, 25, 2),
                       (1, "2026-03-06 07:00", 25, 25, 99)])

        assert t.loc[1, "room_is_private"] == 0
        assert t.loc[2, "room_is_private"] == 1
        assert t.loc[1, "room_age_days"] == pytest.approx(29 + 7 / 24)
        assert np.isnan(t.loc[3, "room_is_private"])  # unknown room

    def test_session_before_room_creation_has_no_age(self):
        t = table_for([(1, "2026-01-15 07:00", 25)])  # room 1 was created on 2026-02-01
        assert np.isnan(t.loc[1, "room_age_days"])

    def test_targets(self):
        t = table_for(USER_ROWS)
        assert t.loc[2, "target_focused_minutes"] == 10
        assert t.loc[2, "target_full_duration"] == 0
        assert t.loc[1, "target_full_duration"] == 1

    def test_users_do_not_share_history(self):
        t = table_for([(1, "2026-03-02 07:00", 25), (2, "2026-03-02 09:00", 25)])
        assert t.loc[2, "prev_sessions"] == 0

    def test_invalid_logs_are_excluded(self):
        logs = make_logs([(1, "2026-03-02 07:00", 25), (1, "2026-03-02 09:00", 40)])  # 40 > planned 25
        assert logs["quality"].tolist() == [cleaning.OK, cleaning.INVALID_DURATION]
        assert len(f.build_feature_table(logs, ROOMS)) == 1


class TestNoLeakage:
    def test_overlapping_session_is_not_history(self):
        """B starts while A is still running: A has not ended, so it is not B's history.
        A row-order shift(1) would wrongly count it."""
        t = table_for([(1, "2026-03-02 07:00", 25), (1, "2026-03-02 07:10", 25)])
        assert t.loc[2, "prev_sessions"] == 0

    def test_session_ending_exactly_at_start_is_not_counted(self):
        t = table_for([(1, "2026-03-02 07:00", 25), (1, "2026-03-02 07:25", 25)])
        assert t.loc[2, "prev_sessions"] == 0  # strict "before": allow_exact_matches=False

    def test_features_ignore_the_future(self):
        """Property: each row's features are identical when everything that happened after
        its start is deleted. If any feature peeked ahead, this would fail."""
        rng = np.random.default_rng(0)
        rows = []
        for user in (1, 2, 3):
            t = pd.Timestamp("2026-03-01 06:00")
            for _ in range(25):
                t += pd.Timedelta(minutes=int(rng.integers(5, 2000)))
                planned = int(rng.choice([25, 50]))
                rows.append((user, str(t), int(rng.integers(5, planned + 1)), planned))
        logs = make_logs(rows)
        full = f.build_feature_table(logs, ROOMS).set_index("log_id")

        for log_id in rng.choice(full.index, size=25, replace=False):
            as_of = full.loc[log_id, "as_of"]
            past_only = logs[(logs["logged_at"] < as_of) | (logs["log_id"] == log_id)]
            truncated = f.build_feature_table(past_only, ROOMS).set_index("log_id")
            pd.testing.assert_series_equal(
                full.loc[log_id, list(f.FEATURE_COLUMNS)], truncated.loc[log_id, list(f.FEATURE_COLUMNS)],
                check_names=False,
            )

    def test_row_order_does_not_matter(self):
        logs = make_logs(USER_ROWS)
        a = f.build_feature_table(logs, ROOMS).set_index("log_id").sort_index()
        b = f.build_feature_table(logs.sample(frac=1, random_state=1), ROOMS).set_index("log_id").sort_index()
        pd.testing.assert_frame_equal(a, b)

    def test_forbidden_and_target_columns_are_not_features(self):
        assert not set(f.FEATURE_COLUMNS) & set(f.FORBIDDEN_FEATURES)
        assert not set(f.FEATURE_COLUMNS) & set(f.TARGETS)
        assert not set(f.FEATURE_COLUMNS) & set(f.KEY_COLUMNS)

    def test_features_and_target(self):
        X, y = f.features_and_target(table_for(USER_ROWS).reset_index(), "target_full_duration")

        assert list(X.columns) == list(f.FEATURE_COLUMNS)
        assert "current_member_count" not in X.columns
        assert y.tolist() == [1, 0, 1, 1, 1]
        with pytest.raises(ValueError):
            f.features_and_target(table_for(USER_ROWS), "focused_seconds")


class TestTemporalSplit:
    def test_train_strictly_before_test(self):
        rows = [(1, f"2026-03-{d:02d} 07:00", 25) for d in range(1, 21)]
        split = f.temporal_split(f.build_feature_table(make_logs(rows), ROOMS), test_fraction=0.25)

        assert len(split.test) == 5
        assert split.train["as_of"].max() < split.test["as_of"].min()
        assert not set(split.train["log_id"]) & set(split.test["log_id"])

    def test_purges_training_rows_whose_label_arrives_after_cutoff(self):
        # Log 3 starts before the cutoff (log 4's start) but ends after it.
        rows = [(1, "2026-03-01 07:00", 25), (1, "2026-03-02 07:00", 25),
                (2, "2026-03-03 06:50", 25), (1, "2026-03-03 07:00", 25)]
        table = f.build_feature_table(make_logs(rows), ROOMS)

        split = f.temporal_split(table, test_fraction=0.25)

        assert split.cutoff == pd.Timestamp("2026-03-03 07:00", tz="UTC")
        assert split.purged == 1
        assert 3 not in set(split.train["log_id"]) | set(split.test["log_id"])
        assert (split.train["label_available_at"] < split.cutoff).all()

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            f.temporal_split(table_for(USER_ROWS), test_fraction=1.0)


class TestEdgeCases:
    def test_empty_input(self):
        t = f.build_feature_table(make_logs(USER_ROWS).iloc[0:0], ROOMS)

        assert t.empty
        assert set(f.FEATURE_COLUMNS) <= set(t.columns)
        assert f.temporal_split(t).purged == 0

    def test_describe_features_reports_cold_start_missingness(self):
        desc = f.describe_features(table_for(USER_ROWS).reset_index()).set_index("feature")

        assert desc.loc["prev_mean_minutes", "missing_share"] == pytest.approx(1 / 5)
        assert desc.loc["hour", "missing_share"] == 0
        assert len(desc) == len(f.FEATURE_COLUMNS)
