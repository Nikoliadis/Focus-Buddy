"""Exploratory analysis of focus data with pandas.

Every function is pure: cleaned DataFrames in, a summary DataFrame out. Only rows with
``quality == "ok"`` are used, so the rules in ``analytics.cleaning`` decide what counts.

Time-of-day analyses use the *local* session start time (Europe/Athens by default),
because the database stores UTC.

Caveat that applies to every number here: with the current schema, focus logs exist only
for the user who ended a session (the room owner), and break sessions are logged as well.
"""
from __future__ import annotations

import pandas as pd

from analytics.cleaning import ABANDONED, DEFAULT_LOCAL_TZ, IN_PROGRESS, NEVER_STARTED, OK

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

FULL_DURATION = "full_duration"
ENDED_EARLY = "ended_early"
INVALID = "invalid"
# A session "ran its full duration" if at least this share of the planned time was focused.
# remaining_seconds() truncates, so a naturally finished 25-min session can log 1499s.
FULL_DURATION_THRESHOLD = 0.99


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["quality"] == OK]


def _with_local_start(logs: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Valid logs with a known start time, plus local hour / weekday / date columns."""
    ok = _ok(logs)
    ok = ok[ok["session_started_at"].notna()]
    local = ok["session_started_at"].dt.tz_convert(tz)
    return ok.assign(
        hour=local.dt.hour,
        weekday=local.dt.dayofweek,  # Monday = 0
        local_date=local.dt.date,
    )


def overall_summary(logs: pd.DataFrame) -> pd.Series:
    ok = _ok(logs)
    focused = ok["focused_seconds"]
    return pd.Series({
        "sessions": len(ok),
        "users": ok["user_id"].nunique(),
        "total_hours": focused.sum() / 3600,
        "mean_minutes": focused.mean() / 60,
        "median_minutes": focused.median() / 60,
        "std_minutes": focused.std() / 60,
    })


def user_focus_summary(logs: pd.DataFrame, tz: str = DEFAULT_LOCAL_TZ) -> pd.DataFrame:
    """Per user: total / mean / median focus, number of sessions and active days."""
    df = _with_local_start(logs, tz)
    summary = (
        df.groupby("user_id")
        .agg(
            sessions=("log_id", "count"),
            total_seconds=("focused_seconds", "sum"),
            mean_seconds=("focused_seconds", "mean"),
            median_seconds=("focused_seconds", "median"),
            active_days=("local_date", "nunique"),
            first_seen=("logged_at", "min"),
            last_seen=("logged_at", "max"),
        )
        .sort_values("total_seconds", ascending=False)
        .reset_index()
    )
    summary["total_hours"] = summary["total_seconds"] / 3600
    summary["mean_minutes"] = summary["mean_seconds"] / 60
    summary["median_minutes"] = summary["median_seconds"] / 60
    return summary


def _by_time_unit(logs: pd.DataFrame, unit: str, index: range, tz: str) -> pd.DataFrame:
    df = _with_local_start(logs, tz)
    grouped = df.groupby(unit).agg(
        sessions=("log_id", "count"),
        total_seconds=("focused_seconds", "sum"),
        mean_seconds=("focused_seconds", "mean"),
    )
    # Every hour/weekday appears, even with no data. Counts become 0; the mean stays
    # NaN because "no sessions" is not the same as "sessions of zero length".
    grouped = grouped.reindex(index)
    grouped[["sessions", "total_seconds"]] = grouped[["sessions", "total_seconds"]].fillna(0).astype("int64")
    grouped["total_minutes"] = grouped["total_seconds"] / 60
    grouped["mean_minutes"] = grouped["mean_seconds"] / 60
    grouped["share_of_focus"] = grouped["total_seconds"] / max(grouped["total_seconds"].sum(), 1)
    grouped.index.name = unit
    return grouped.reset_index()


def focus_by_hour(logs: pd.DataFrame, tz: str = DEFAULT_LOCAL_TZ) -> pd.DataFrame:
    """24 rows, one per local hour of session start."""
    return _by_time_unit(logs, "hour", range(24), tz)


def focus_by_weekday(logs: pd.DataFrame, tz: str = DEFAULT_LOCAL_TZ) -> pd.DataFrame:
    """7 rows, Monday first."""
    out = _by_time_unit(logs, "weekday", range(7), tz)
    out.insert(1, "weekday_name", [WEEKDAY_NAMES[d] for d in out["weekday"]])
    return out


def weekday_hour_matrix(logs: pd.DataFrame, tz: str = DEFAULT_LOCAL_TZ) -> pd.DataFrame:
    """7 × 24 pivot of focused minutes (rows: weekday, columns: hour). Input for a heatmap."""
    df = _with_local_start(logs, tz)
    matrix = pd.pivot_table(
        df, index="weekday", columns="hour", values="focused_seconds", aggfunc="sum", fill_value=0
    ) if not df.empty else pd.DataFrame()
    matrix = matrix.reindex(index=range(7), columns=range(24), fill_value=0) / 60
    matrix.index = pd.Index(WEEKDAY_NAMES, name="weekday")
    matrix.columns.name = "hour"
    return matrix


def daily_focus(
    logs: pd.DataFrame,
    window: int = 7,
    user_id: int | None = None,
    tz: str = DEFAULT_LOCAL_TZ,
) -> pd.DataFrame:
    """Focused minutes per local calendar day, with a rolling mean.

    Days without any session are filled with 0 *before* the rolling mean, otherwise a
    7-row window would silently span more than 7 calendar days.
    ``min_periods=1`` means the first ``window - 1`` days average over fewer days.
    """
    df = _with_local_start(logs, tz)
    if user_id is not None:
        df = df[df["user_id"] == user_id]
    if df.empty:
        return pd.DataFrame(columns=["date", "focus_minutes", f"rolling_{window}d_mean"])

    per_day = df.groupby(pd.to_datetime(df["local_date"]))["focused_seconds"].sum() / 60
    full_range = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D")
    per_day = per_day.reindex(full_range, fill_value=0)

    out = pd.DataFrame({"focus_minutes": per_day})
    out[f"rolling_{window}d_mean"] = per_day.rolling(window, min_periods=1).mean()
    out.index.name = "date"
    return out.reset_index()


def session_outcomes(
    sessions: pd.DataFrame,
    logs: pd.DataFrame,
    threshold: float = FULL_DURATION_THRESHOLD,
) -> pd.DataFrame:
    """One row per session with an ``outcome``.

    Outcomes are *derived*, because the app does not store why a session ended:
      full_duration  ended and >= threshold of the planned time was focused
      ended_early    ended before that (manual End, or the owner started a new one)
      abandoned      still running/paused long after it should have finished
      in_progress / never_started / invalid   excluded from completion rates
    """
    per_session = (
        _ok(logs).groupby("session_id", as_index=False)["completion_ratio"].max()
    )
    out = sessions.merge(per_session, on="session_id", how="left", validate="one_to_one")

    outcome = pd.Series(INVALID, index=out.index, dtype="object")
    outcome[out["quality"] == ABANDONED] = ABANDONED
    outcome[out["quality"] == IN_PROGRESS] = IN_PROGRESS
    outcome[out["quality"] == NEVER_STARTED] = NEVER_STARTED
    ended = out["quality"] == OK
    outcome[ended & (out["completion_ratio"] >= threshold)] = FULL_DURATION
    outcome[ended & (out["completion_ratio"] < threshold)] = ENDED_EARLY
    # An ended session without a valid log stays "invalid": we cannot tell how it went.
    out["outcome"] = outcome
    return out


def completion_rate(outcomes: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Share of *decided* sessions that ran their full duration.

    rate = full_duration / (full_duration + ended_early + abandoned)
    Sessions still in progress, never started, or invalid are not in the denominator.
    """
    decided = outcomes[outcomes["outcome"].isin([FULL_DURATION, ENDED_EARLY, ABANDONED])]
    if by:
        counts = decided.groupby(by)["outcome"].value_counts().unstack(fill_value=0)
    else:
        counts = decided["outcome"].value_counts().to_frame().T.reset_index(drop=True)
    counts = counts.reindex(columns=[FULL_DURATION, ENDED_EARLY, ABANDONED], fill_value=0)
    counts.columns.name = None
    counts["decided_sessions"] = counts.sum(axis=1)
    counts["completion_rate"] = counts[FULL_DURATION] / counts["decided_sessions"].where(counts["decided_sessions"] > 0)
    return counts.reset_index() if by else counts


def room_activity(logs: pd.DataFrame, rooms: pd.DataFrame, tz: str = DEFAULT_LOCAL_TZ) -> pd.DataFrame:
    """Per room: credited sessions, focus hours, active days. Rooms with no activity show 0.

    ``current_member_count`` is today's snapshot (see data_loader.load_rooms) and must not
    be read as the room size at the time of the sessions.
    """
    df = _with_local_start(logs, tz)
    activity = df.groupby("room_id").agg(
        sessions=("log_id", "count"),
        total_seconds=("focused_seconds", "sum"),
        active_days=("local_date", "nunique"),
        last_activity=("logged_at", "max"),
    )
    out = rooms[["room_id", "is_private", "current_member_count", "created_at"]].merge(
        activity, left_on="room_id", right_index=True, how="left", validate="one_to_one"
    )
    out[["sessions", "total_seconds", "active_days"]] = (
        out[["sessions", "total_seconds", "active_days"]].fillna(0).astype("int64")
    )
    out["total_hours"] = out["total_seconds"] / 3600
    return out.sort_values("total_seconds", ascending=False).reset_index(drop=True)


def session_duration_profile(logs: pd.DataFrame) -> pd.DataFrame:
    """Planned duration vs actual focused time, grouped by planned length in minutes."""
    ok = _ok(logs)
    out = (
        ok.assign(planned_minutes=ok["session_duration_seconds"] // 60, focused_minutes=ok["focused_seconds"] / 60)
        .groupby("planned_minutes")
        .agg(
            sessions=("log_id", "count"),
            mean_focused_minutes=("focused_minutes", "mean"),
            median_focused_minutes=("focused_minutes", "median"),
            mean_completion_ratio=("completion_ratio", "mean"),
        )
        .reset_index()
    )
    out["share_of_sessions"] = out["sessions"] / max(out["sessions"].sum(), 1)
    return out
