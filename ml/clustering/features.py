"""Per-user behaviour features for clustering. One row per user, all measurable.

Every feature is a rate or an average, so heavy and light users are comparable; the
volume itself is captured once, as ``sessions_per_week``. Users with fewer than
``min_sessions`` valid sessions are left out: their shares are too noisy (one evening
session makes a user "100% evening").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.analysis import FULL_DURATION_THRESHOLD
from analytics.cleaning import DEFAULT_LOCAL_TZ, OK

DEFAULT_MIN_SESSIONS = 5

# Local start-hour bands. Afternoon (12-18) is the remainder and is left out on purpose:
# the four shares sum to 1, so the fourth would add no information, only extra weight.
TIME_BANDS = {"morning": (5, 12), "evening": (18, 23)}

USER_FEATURES = {
    "sessions_per_week": "Valid sessions per week of activity span",
    "mean_focus_minutes": "Mean focused minutes per session",
    "full_duration_rate": "Share of sessions that ran their full planned length",
    "long_session_share": "Share of sessions planned at 50 minutes or more",
    "morning_share": "Share of sessions starting 05:00-11:59 local",
    "evening_share": "Share of sessions starting 18:00-22:59 local",
    "night_share": "Share of sessions starting 23:00-04:59 local",
    "weekend_share": "Share of sessions on Saturday or Sunday",
    "distinct_rooms": "Number of different rooms used",
    "private_room_share": "Share of sessions in password-protected rooms",
}


def build_user_features(
    logs: pd.DataFrame,
    rooms: pd.DataFrame,
    tz: str = DEFAULT_LOCAL_TZ,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
) -> pd.DataFrame:
    """Index: user_id. Columns: ``USER_FEATURES`` plus ``n_sessions``."""
    ok = logs[(logs["quality"] == OK) & logs["session_started_at"].notna()]
    if ok.empty:
        return pd.DataFrame(columns=["n_sessions", *USER_FEATURES]).rename_axis("user_id")

    hour = ok["session_started_at"].dt.tz_convert(tz).dt.hour
    weekday = ok["session_started_at"].dt.tz_convert(tz).dt.dayofweek
    private = ok["room_id"].map(rooms.drop_duplicates("room_id").set_index("room_id")["is_private"])

    df = pd.DataFrame({
        "user_id": ok["user_id"],
        "start": ok["session_started_at"],
        "minutes": ok["focused_seconds"] / 60,
        "full": ok["completion_ratio"] >= FULL_DURATION_THRESHOLD,
        "long": ok["session_duration_seconds"] >= 50 * 60,
        "morning": hour.between(*TIME_BANDS["morning"], inclusive="left"),
        "evening": hour.between(*TIME_BANDS["evening"], inclusive="left"),
        "night": (hour >= 23) | (hour < 5),
        "weekend": weekday >= 5,
        "room_id": ok["room_id"],
        "private": private.astype("float64"),
    })
    g = df.groupby("user_id")
    span_weeks = ((g["start"].max() - g["start"].min()).dt.total_seconds() / (7 * 86400)).clip(lower=1)

    out = pd.DataFrame({
        "n_sessions": g.size(),
        "sessions_per_week": g.size() / span_weeks,
        "mean_focus_minutes": g["minutes"].mean(),
        "full_duration_rate": g["full"].mean(),
        "long_session_share": g["long"].mean(),
        "morning_share": g["morning"].mean(),
        "evening_share": g["evening"].mean(),
        "night_share": g["night"].mean(),
        "weekend_share": g["weekend"].mean(),
        "distinct_rooms": g["room_id"].nunique(),
        "private_room_share": g["private"].mean(),
    })
    out = out[out["n_sessions"] >= min_sessions]
    # A room missing from `rooms` gives NaN privacy; treat unknown as not private.
    out["private_room_share"] = out["private_room_share"].fillna(0)
    return out.astype("float64").replace([np.inf, -np.inf], np.nan)
