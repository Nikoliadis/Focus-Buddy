"""Pure-pandas tests for the data-quality rules. No database needed."""
from __future__ import annotations

import pandas as pd
import pytest

from analytics import cleaning as c

NOW = pd.Timestamp("2026-03-10 12:00", tz="UTC")


def ts(value: str | None) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC") if value else pd.NaT


def session(**overrides) -> dict:
    row = {
        "session_id": 1,
        "room_id": 1,
        "started_by": 1,
        "status": "ended",
        "duration_seconds": 1500,
        "started_at": ts("2026-03-10 09:00"),
        "ended_at": ts("2026-03-10 09:25"),
        "paused_at": pd.NaT,
        "paused_seconds": 0,
        "created_at": ts("2026-03-10 09:00"),
        "room_exists": True,
    }
    row.update(overrides)
    return row


def sessions_frame(*rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for col in ("started_at", "ended_at", "paused_at", "created_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def clean_one(**overrides) -> str:
    out, _ = c.clean_focus_sessions(sessions_frame(session(**overrides)), now=NOW)
    return out["quality"].iloc[0]


class TestCleanFocusSessions:
    def test_valid_ended_session(self):
        out, report = c.clean_focus_sessions(sessions_frame(session()), now=NOW)

        assert out["quality"].tolist() == [c.OK]
        assert out["wall_seconds"].iloc[0] == 1500
        assert report.usable_rows == 1

    @pytest.mark.parametrize("duration", [None, 0, -10, 30, 180 * 60 + 1])
    def test_invalid_duration(self, duration):
        assert clean_one(duration_seconds=duration) == c.INVALID_DURATION

    def test_negative_paused_seconds(self):
        assert clean_one(paused_seconds=-5) == c.INVALID_DURATION

    def test_null_paused_seconds_is_treated_as_zero(self):
        out, _ = c.clean_focus_sessions(sessions_frame(session(paused_seconds=None)), now=NOW)
        assert out["paused_seconds"].iloc[0] == 0
        assert out["quality"].iloc[0] == c.OK

    def test_idle_or_reset_session_never_started(self):
        assert clean_one(status="idle", started_at=None, ended_at=None) == c.NEVER_STARTED

    def test_unknown_status(self):
        assert clean_one(status="exploded") == c.INVALID_STATUS

    @pytest.mark.parametrize(
        "overrides",
        [
            {"ended_at": ts("2026-03-10 08:00")},                              # ends before it starts
            {"started_at": ts("2026-03-11 09:00"), "ended_at": None, "status": "running"},  # in the future
            {"ended_at": None},                                                # ended without ended_at
            {"status": "paused", "ended_at": None, "paused_at": None},         # paused without paused_at
        ],
        ids=["end-before-start", "future-start", "ended-no-end", "paused-no-pause-time"],
    )
    def test_invalid_timestamps(self, overrides):
        assert clean_one(**overrides) == c.INVALID_TIMESTAMP

    def test_running_session_still_within_its_time_is_in_progress(self):
        assert clean_one(status="running", started_at=ts("2026-03-10 11:50"), ended_at=None) == c.IN_PROGRESS

    def test_running_session_long_overdue_is_abandoned(self):
        # 25 min session started 3h ago and never ended: the owner closed the tab.
        assert clean_one(status="running", started_at=ts("2026-03-10 09:00"), ended_at=None) == c.ABANDONED

    def test_paused_time_extends_expected_end(self):
        # Started 1h50m ago, but paused for 1h40m in total: expected end is ~15 min ago, within grace.
        overrides = {
            "status": "running", "started_at": ts("2026-03-10 10:10"), "ended_at": None, "paused_seconds": 6000,
        }
        assert clean_one(**overrides) == c.IN_PROGRESS

    def test_long_paused_session_is_abandoned(self):
        overrides = {"status": "paused", "ended_at": None, "paused_at": ts("2026-03-10 09:05")}
        assert clean_one(**overrides) == c.ABANDONED

    def test_duplicates_are_dropped_and_counted(self):
        df = sessions_frame(session(), session(), session(session_id=2))

        out, report = c.clean_focus_sessions(df, now=NOW)

        assert len(out) == 2
        assert report.dropped["duplicate_session_id"] == 1

    def test_first_matching_rule_wins(self):
        # Both invalid duration and invalid timestamps: duration is reported.
        assert clean_one(duration_seconds=0, ended_at=ts("2026-03-10 08:00")) == c.INVALID_DURATION

    def test_empty_frame(self):
        out, report = c.clean_focus_sessions(sessions_frame(session()).iloc[0:0], now=NOW)
        assert out.empty
        assert report.rows_in == 0 and report.usable_rows == 0

    def test_input_is_not_mutated(self):
        df = sessions_frame(session(paused_seconds=None))
        c.clean_focus_sessions(df, now=NOW)
        assert "quality" not in df.columns
        assert pd.isna(df["paused_seconds"].iloc[0])


def logs_frame(*rows: dict) -> pd.DataFrame:
    base = {
        "log_id": 1, "user_id": 1, "room_id": 1, "session_id": 1, "focused_seconds": 1500,
        "logged_at": ts("2026-03-10 09:25"), "session_duration_seconds": 1500,
        "session_started_at": ts("2026-03-10 09:00"), "session_ended_at": ts("2026-03-10 09:25"),
        "session_paused_seconds": 0, "session_started_by": 1,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
    return df


class TestCleanFocusLogs:
    def test_valid_log_and_completion_ratio(self):
        out, _ = c.clean_focus_logs(logs_frame({}, {"log_id": 2, "session_id": 2, "focused_seconds": 750}), now=NOW)

        assert out["quality"].tolist() == [c.OK, c.OK]
        assert out["completion_ratio"].tolist() == [1.0, 0.5]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"focused_seconds": -1},
            {"focused_seconds": None},
            {"focused_seconds": 2000},
            {"session_duration_seconds": 0},
        ],
        ids=["negative", "null", "more-than-duration", "zero-duration"],
    )
    def test_invalid_durations(self, overrides):
        out, _ = c.clean_focus_logs(logs_frame(overrides), now=NOW)

        assert out["quality"].iloc[0] == c.INVALID_DURATION
        assert pd.isna(out["completion_ratio"].iloc[0])

    def test_rounding_tolerance(self):
        out, _ = c.clean_focus_logs(logs_frame({"focused_seconds": 1501}), now=NOW)
        assert out["quality"].iloc[0] == c.OK

    def test_future_timestamp(self):
        out, _ = c.clean_focus_logs(logs_frame({"logged_at": ts("2027-01-01")}), now=NOW)
        assert out["quality"].iloc[0] == c.INVALID_TIMESTAMP

    def test_same_user_credited_twice_for_one_session(self):
        df = logs_frame({}, {"log_id": 2, "logged_at": ts("2026-03-10 09:26")})

        out, report = c.clean_focus_logs(df, now=NOW)

        assert out["log_id"].tolist() == [1]  # earliest kept
        assert report.dropped == {"duplicate_log_id": 0, "duplicate_session_user": 1}

    def test_duplicate_log_ids(self):
        out, report = c.clean_focus_logs(logs_frame({}, {}), now=NOW)
        assert len(out) == 1
        assert report.dropped["duplicate_log_id"] == 1


def chat_frame(*rows: dict) -> pd.DataFrame:
    base = {"message_id": 1, "room_id": 1, "user_id": 1, "body": "hi", "created_at": ts("2026-03-10 09:00"),
            "room_is_private": False}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


class TestCleanChat:
    def test_whitespace_is_normalised(self):
        out, _ = c.clean_chat_messages(chat_frame({"body": "  hello \n  world  "}), now=NOW)

        assert out["body"].iloc[0] == "hello world"
        assert out["n_words"].iloc[0] == 2
        assert out["n_chars"].iloc[0] == 11

    def test_empty_and_null_bodies_are_dropped(self):
        df = chat_frame({"body": "   "}, {"message_id": 2, "body": None}, {"message_id": 3, "body": "ok"})

        out, report = c.clean_chat_messages(df, now=NOW)

        assert out["body"].tolist() == ["ok"]
        assert report.dropped["empty_body"] == 2

    def test_double_submit_is_deduplicated(self):
        df = chat_frame({}, {"message_id": 2})

        out, report = c.clean_chat_messages(df, now=NOW)

        assert len(out) == 1
        assert report.dropped["duplicate_message"] == 1

    def test_same_text_at_different_times_is_kept(self):
        out, _ = c.clean_chat_messages(chat_frame({}, {"message_id": 2, "created_at": ts("2026-03-10 09:05")}), now=NOW)
        assert len(out) == 2


def test_add_local_time_handles_daylight_saving():
    df = pd.DataFrame({"t": pd.to_datetime(["2026-01-15 19:00", "2026-07-15 19:00"], utc=True)})

    out = c.add_local_time(df, ["t"])

    # Athens is UTC+2 in winter and UTC+3 in summer.
    assert out["t_local"].dt.hour.tolist() == [21, 22]


def test_quality_report_str_lists_flags():
    report = c.QualityReport("x", rows_in=3, rows_out=2, dropped={"dup": 1}, flags={"ok": 1, "abandoned": 1})
    text = str(report)
    assert "x: 3 in -> 2 out (1 usable)" in text
    assert "dropped dup: 1" in text
    assert "flag abandoned: 1" in text
