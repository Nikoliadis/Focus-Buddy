"""EDA functions on small hand-built datasets with answers computed by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics import analysis as a
from analytics import cleaning

NOW = pd.Timestamp("2026-03-20 12:00", tz="UTC")


def make_logs(rows: list[dict]) -> pd.DataFrame:
    """rows: dicts with user_id, start (UTC string), focused (minutes); optional room_id, planned."""
    records = []
    for i, r in enumerate(rows, start=1):
        start = pd.Timestamp(r["start"], tz="UTC")
        focused = r.get("focused", 25) * 60
        records.append({
            "log_id": i,
            "user_id": r["user_id"],
            "room_id": r.get("room_id", 1),
            "session_id": r.get("session_id", i),
            "focused_seconds": focused,
            "logged_at": start + pd.Timedelta(seconds=focused),
            "session_duration_seconds": r.get("planned", 25) * 60,
            "session_started_at": start,
            "session_ended_at": start + pd.Timedelta(seconds=focused),
            "session_paused_seconds": 0,
            "session_started_by": r["user_id"],
        })
    logs, _ = cleaning.clean_focus_logs(pd.DataFrame(records), now=NOW)
    return logs


# 2026-03-02 is a Monday. In March before the 29th, Athens = UTC+2.
LOGS = make_logs([
    {"user_id": 1, "start": "2026-03-02 07:00", "focused": 25},             # Mon 09:00 local
    {"user_id": 1, "start": "2026-03-02 07:30", "focused": 50, "planned": 50},  # Mon 09:30 local
    {"user_id": 1, "start": "2026-03-04 18:00", "focused": 10},             # Wed 20:00 local, ended early
    {"user_id": 2, "start": "2026-03-07 08:00", "focused": 25},             # Sat 10:00 local
])


class TestOverallAndUsers:
    def test_overall_summary(self):
        s = a.overall_summary(LOGS)

        assert s["sessions"] == 4
        assert s["users"] == 2
        assert s["total_hours"] == pytest.approx(110 / 60)
        assert s["mean_minutes"] == pytest.approx(27.5)
        assert s["median_minutes"] == pytest.approx(25)

    def test_user_summary(self):
        users = a.user_focus_summary(LOGS).set_index("user_id")

        assert users.index.tolist() == [1, 2]  # sorted by total focus
        assert users.loc[1, "sessions"] == 3
        assert users.loc[1, "total_seconds"] == 85 * 60
        assert users.loc[1, "median_minutes"] == pytest.approx(25)
        assert users.loc[1, "mean_minutes"] == pytest.approx(85 / 3)
        assert users.loc[1, "active_days"] == 2

    def test_invalid_rows_are_ignored(self):
        bad = make_logs([{"user_id": 1, "start": "2026-03-02 07:00", "focused": 30, "planned": 25}])
        assert bad["quality"].iloc[0] == cleaning.INVALID_DURATION
        assert a.user_focus_summary(bad).empty


class TestTimeOfDay:
    def test_hours_use_local_time(self):
        hours = a.focus_by_hour(LOGS).set_index("hour")

        assert len(hours) == 24
        assert hours.loc[9, "sessions"] == 2      # 07:00 and 07:30 UTC -> 09:xx Athens
        assert hours.loc[7, "sessions"] == 0      # not the UTC hour
        assert hours.loc[9, "total_minutes"] == 75

    def test_empty_hours_have_zero_count_but_nan_mean(self):
        hours = a.focus_by_hour(LOGS).set_index("hour")

        assert hours.loc[3, "sessions"] == 0
        assert np.isnan(hours.loc[3, "mean_minutes"])

    def test_shares_sum_to_one(self):
        assert a.focus_by_hour(LOGS)["share_of_focus"].sum() == pytest.approx(1)

    def test_weekday_order_and_names(self):
        days = a.focus_by_weekday(LOGS)

        assert days["weekday_name"].tolist() == a.WEEKDAY_NAMES
        assert days.set_index("weekday_name").loc["Mon", "sessions"] == 2
        assert days.set_index("weekday_name").loc["Sat", "total_minutes"] == 25

    def test_weekday_hour_matrix(self):
        m = a.weekday_hour_matrix(LOGS)

        assert m.shape == (7, 24)
        assert m.loc["Mon", 9] == 75
        assert m.loc["Wed", 20] == 10
        assert m.to_numpy().sum() == 110

    def test_hour_respects_daylight_saving(self):
        summer = make_logs([{"user_id": 1, "start": "2025-07-01 07:00"}])
        assert a.focus_by_hour(summer).set_index("hour").loc[10, "sessions"] == 1  # UTC+3 in summer

    def test_other_timezone(self):
        hours = a.focus_by_hour(LOGS, tz="UTC").set_index("hour")
        assert hours.loc[7, "sessions"] == 2


class TestDailyFocus:
    def test_missing_days_are_zero_before_rolling(self):
        daily = a.daily_focus(LOGS, window=3)

        assert daily["date"].dt.strftime("%Y-%m-%d").tolist() == [
            "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06", "2026-03-07",
        ]
        assert daily["focus_minutes"].tolist() == [75, 0, 10, 0, 0, 25]
        # 3-day window over [75, 0, 10] -> 85/3; the gap days pull the average down.
        assert daily["rolling_3d_mean"].iloc[2] == pytest.approx(85 / 3)
        assert daily["rolling_3d_mean"].iloc[0] == 75  # min_periods=1

    def test_single_user(self):
        daily = a.daily_focus(LOGS, user_id=2)
        assert daily["focus_minutes"].tolist() == [25]

    def test_empty(self):
        daily = a.daily_focus(LOGS, user_id=999)
        assert daily.empty
        assert "rolling_7d_mean" in daily.columns


def make_sessions(rows: list[dict]) -> pd.DataFrame:
    base = {"room_id": 1, "started_by": 1, "duration_seconds": 1500, "paused_at": pd.NaT,
            "paused_seconds": 0, "room_exists": True}
    df = pd.DataFrame([{**base, **r} for r in rows])
    for col in ("started_at", "ended_at", "paused_at", "created_at"):
        df[col] = pd.to_datetime(df.get(col), utc=True)
    out, _ = cleaning.clean_focus_sessions(df, now=NOW)
    return out


class TestOutcomes:
    @pytest.fixture()
    def outcomes(self):
        sessions = make_sessions([
            {"session_id": 1, "status": "ended", "started_at": "2026-03-02 07:00", "ended_at": "2026-03-02 07:25"},
            {"session_id": 3, "status": "ended", "started_at": "2026-03-04 18:00", "ended_at": "2026-03-04 18:10"},
            {"session_id": 10, "status": "running", "started_at": "2026-03-19 07:00", "ended_at": None},
            {"session_id": 11, "status": "running", "started_at": "2026-03-20 11:50", "ended_at": None},
            {"session_id": 12, "status": "idle", "started_at": None, "ended_at": None},
            {"session_id": 13, "status": "ended", "started_at": "2026-03-05 07:00", "ended_at": "2026-03-05 07:25"},
        ])
        return a.session_outcomes(sessions, LOGS)

    def test_outcome_labels(self, outcomes):
        labels = outcomes.set_index("session_id")["outcome"].to_dict()

        assert labels == {
            1: a.FULL_DURATION,
            3: a.ENDED_EARLY,      # 10 of 25 minutes
            10: a.ABANDONED,
            11: a.IN_PROGRESS,
            12: a.NEVER_STARTED,
            13: a.INVALID,          # ended, but no valid log to tell how it went
        }

    def test_completion_rate_excludes_undecided_sessions(self, outcomes):
        rate = a.completion_rate(outcomes)

        assert rate["decided_sessions"].iloc[0] == 3   # full + early + abandoned
        assert rate["completion_rate"].iloc[0] == pytest.approx(1 / 3)

    def test_completion_rate_by_group(self, outcomes):
        rate = a.completion_rate(outcomes, by="started_by")
        assert rate.set_index("started_by").loc[1, "decided_sessions"] == 3

    def test_completion_rate_with_nothing_decided(self):
        sessions = make_sessions([{"session_id": 1, "status": "idle", "started_at": None, "ended_at": None}])
        rate = a.completion_rate(a.session_outcomes(sessions, LOGS))

        assert rate["decided_sessions"].iloc[0] == 0
        assert np.isnan(rate["completion_rate"].iloc[0])  # undefined, not 0%

    def test_threshold_tolerates_truncated_seconds(self):
        logs = LOGS.copy()
        logs.loc[logs["session_id"] == 1, "completion_ratio"] = 1499 / 1500
        sessions = make_sessions([
            {"session_id": 1, "status": "ended", "started_at": "2026-03-02 07:00", "ended_at": "2026-03-02 07:25"},
        ])
        assert a.session_outcomes(sessions, logs)["outcome"].iloc[0] == a.FULL_DURATION


class TestRoomsAndDurations:
    def test_room_activity_keeps_inactive_rooms(self):
        rooms = pd.DataFrame({
            "room_id": [1, 2],
            "is_private": [False, True],
            "current_member_count": [3, 1],
            "created_at": pd.to_datetime(["2026-03-01", "2026-03-01"], utc=True),
        })

        out = a.room_activity(LOGS, rooms).set_index("room_id")

        assert out.loc[1, "sessions"] == 4
        assert out.loc[1, "active_days"] == 3
        assert out.loc[2, "sessions"] == 0
        assert out.loc[2, "total_hours"] == 0

    def test_duration_profile(self):
        prof = a.session_duration_profile(LOGS).set_index("planned_minutes")

        assert prof.loc[25, "sessions"] == 3
        assert prof.loc[25, "median_focused_minutes"] == 25
        assert prof.loc[25, "mean_completion_ratio"] == pytest.approx((1 + 0.4 + 1) / 3)
        assert prof.loc[50, "share_of_sessions"] == pytest.approx(0.25)


def test_everything_handles_an_empty_dataset():
    empty = LOGS.iloc[0:0]

    assert a.overall_summary(empty)["sessions"] == 0
    assert a.user_focus_summary(empty).empty
    assert a.focus_by_hour(empty)["sessions"].sum() == 0
    assert a.weekday_hour_matrix(empty).to_numpy().sum() == 0
    assert a.session_duration_profile(empty).empty
