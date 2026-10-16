"""Leakage-safe feature engineering.

Prediction unit: one credited focus session (a valid focus log).
Prediction time (``as_of``): the moment the session *starts*.

The rule every feature obeys: it may only use information that exists at ``as_of``.
- Known at start: the clock (hour, weekday), the planned duration the owner picked,
  static room properties, and the user's *finished* past sessions.
- Not known at start: how long this session will last, whether it will be paused,
  when it ends, and today's room membership (a present-day snapshot).

User-history features are computed with ``pd.merge_asof(..., allow_exact_matches=False)``
against each past session's *end* time: a past session counts only if it had already
ended strictly before this one started. Ordering rows and using ``shift(1)`` would be
subtly wrong when two sessions of one user overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from analytics.analysis import FULL_DURATION_THRESHOLD
from analytics.cleaning import DEFAULT_LOCAL_TZ, OK


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: Literal["numeric", "binary", "categorical"]
    description: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # Time context (known from the clock at session start)
    FeatureSpec("hour", "categorical", "Local hour of session start (0-23)"),
    FeatureSpec("hour_sin", "numeric", "sin(2*pi*hour/24): cyclic encoding, 23:00 is close to 00:00"),
    FeatureSpec("hour_cos", "numeric", "cos(2*pi*hour/24)"),
    FeatureSpec("weekday", "categorical", "Local weekday of session start, Monday = 0"),
    FeatureSpec("is_weekend", "binary", "Saturday or Sunday in local time"),
    # Session setup (chosen by the owner when starting)
    FeatureSpec("planned_minutes", "numeric", "Planned session length the owner selected"),
    # Room (static properties)
    FeatureSpec("room_is_private", "binary", "Room is password-protected"),
    FeatureSpec("room_age_days", "numeric", "Days between room creation and session start"),
    # User history: only sessions that ENDED before this one started
    FeatureSpec("is_first_session", "binary", "No finished session before this one"),
    FeatureSpec("prev_sessions", "numeric", "Number of finished sessions before this one"),
    FeatureSpec("prev_mean_minutes", "numeric", "Mean focused minutes of finished sessions"),
    FeatureSpec("prev_full_duration_rate", "numeric", "Share of finished sessions that ran full duration"),
    FeatureSpec("prev_session_minutes", "numeric", "Focused minutes of the most recent finished session"),
    FeatureSpec("minutes_since_prev", "numeric", "Minutes between the previous session's end and this start"),
    FeatureSpec("sessions_today_before", "numeric", "Finished sessions earlier the same local day"),
    FeatureSpec("sessions_7d_before", "numeric", "Finished sessions in the 7 days before this start"),
    FeatureSpec("user_tenure_days", "numeric", "Days since the user's first finished session"),
)
FEATURE_COLUMNS: tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS)

TARGETS = {
    "target_focused_minutes": "Regression target: minutes actually focused in this session",
    "target_full_duration": "Classification target: 1 if >= 99% of the planned time was focused",
}

# Columns that exist in the data but must never become features, and why.
FORBIDDEN_FEATURES = {
    "focused_seconds": "it is the regression target",
    "completion_ratio": "derived from the target",
    "logged_at": "the session's end time: only known after the session",
    "session_ended_at": "only known after the session",
    "session_paused_seconds": "pauses happen during the session",
    "current_member_count": "today's membership snapshot, not the size at session time",
    "label_available_at": "bookkeeping for the temporal split",
}

# Not features: identifiers and timestamps used for joining and splitting.
KEY_COLUMNS = ("log_id", "user_id", "room_id", "as_of", "label_available_at")


def _ok_events(logs: pd.DataFrame) -> pd.DataFrame:
    ok = logs[(logs["quality"] == OK) & logs["session_started_at"].notna() & logs["logged_at"].notna()]
    return ok.sort_values(["session_started_at", "log_id"]).reset_index(drop=True)


def _history_table(events: pd.DataFrame) -> pd.DataFrame:
    """Cumulative per-user stats, one row per finished session, keyed by its end time."""
    ends = events[["user_id", "logged_at", "focused_seconds", "completion_ratio"]].sort_values(
        ["logged_at", "user_id"], kind="stable"
    ).copy()
    by_user = ends.groupby("user_id", sort=False)
    ends["h_count"] = by_user.cumcount() + 1
    ends["h_minutes"] = by_user["focused_seconds"].cumsum() / 60
    ends["h_full"] = (ends["completion_ratio"] >= FULL_DURATION_THRESHOLD).astype("int64").groupby(
        ends["user_id"], sort=False).cumsum()
    ends["h_last_minutes"] = ends["focused_seconds"] / 60
    ends["h_last_end"] = ends["logged_at"]
    ends["h_first_end"] = by_user["logged_at"].transform("min")
    return ends.drop(columns=["focused_seconds", "completion_ratio"])


def _as_of_join(keys: pd.DataFrame, at: pd.Series, history: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """For each row, the history state from sessions that ended strictly before ``at``."""
    left = pd.DataFrame({"_row": keys.index, "user_id": keys["user_id"].to_numpy(), "_at": at.to_numpy()})
    left = left.sort_values("_at", kind="stable")
    joined = pd.merge_asof(
        left, history[["user_id", "logged_at", *columns]],
        left_on="_at", right_on="logged_at", by="user_id",
        direction="backward", allow_exact_matches=False,
    )
    return joined.set_index("_row").sort_index()[columns]


def build_feature_table(
    logs: pd.DataFrame,
    rooms: pd.DataFrame,
    tz: str = DEFAULT_LOCAL_TZ,
) -> pd.DataFrame:
    """One row per valid session: keys, ``FEATURE_COLUMNS`` and ``TARGETS``.

    ``logs`` and ``rooms`` are the cleaned frames from ``analytics.dataset.load_dataset``.
    History features are NaN for a user's first session (cold start); imputation is
    the model pipeline's job, so it can be fitted on training data only.
    """
    columns = [*KEY_COLUMNS, *FEATURE_COLUMNS, *TARGETS]
    events = _ok_events(logs)
    if events.empty:
        return pd.DataFrame(columns=columns)

    start = events["session_started_at"]
    local = start.dt.tz_convert(tz)
    hour = local.dt.hour

    table = pd.DataFrame({
        "log_id": events["log_id"],
        "user_id": events["user_id"],
        "room_id": events["room_id"],
        "as_of": start,
        "label_available_at": events["logged_at"],
        "hour": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday": local.dt.dayofweek,
        "is_weekend": (local.dt.dayofweek >= 5).astype("int64"),
        "planned_minutes": events["session_duration_seconds"] / 60,
    })

    room_info = rooms[["room_id", "is_private", "created_at"]].drop_duplicates("room_id")
    with_room = table[["room_id"]].merge(room_info, on="room_id", how="left", validate="many_to_one")
    # merge(how="left") keeps the left row order, and both frames have a 0..n-1 index.
    table["room_is_private"] = with_room["is_private"].astype("float64")
    age = (start - with_room["created_at"]).dt.total_seconds() / 86400
    table["room_age_days"] = age.where(age >= 0)  # a session before its room existed is impossible

    history = _history_table(events)
    cols = ["h_count", "h_minutes", "h_full", "h_last_minutes", "h_last_end", "h_first_end"]
    h = _as_of_join(table, start, history, cols)
    count = h["h_count"].fillna(0)
    table["is_first_session"] = (count == 0).astype("int64")
    table["prev_sessions"] = count
    table["prev_mean_minutes"] = h["h_minutes"] / h["h_count"]
    table["prev_full_duration_rate"] = h["h_full"] / h["h_count"]
    table["prev_session_minutes"] = h["h_last_minutes"]
    table["minutes_since_prev"] = (start - h["h_last_end"]).dt.total_seconds() / 60
    table["user_tenure_days"] = (start - h["h_first_end"]).dt.total_seconds() / 86400

    # Rolling-window counts = cumulative count now minus cumulative count at the window start.
    week_ago = _as_of_join(table, start - pd.Timedelta(days=7), history, ["h_count"])["h_count"].fillna(0)
    local_midnight = local.dt.normalize().dt.tz_convert("UTC")
    day_start = _as_of_join(table, local_midnight, history, ["h_count"])["h_count"].fillna(0)
    table["sessions_7d_before"] = count - week_ago
    table["sessions_today_before"] = count - day_start

    table["target_focused_minutes"] = events["focused_seconds"] / 60
    table["target_full_duration"] = (events["completion_ratio"] >= FULL_DURATION_THRESHOLD).astype("int64")
    return table[columns]


def features_and_target(table: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; choose one of {sorted(TARGETS)}")
    return table[list(FEATURE_COLUMNS)].copy(), table[target].copy()


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp
    purged: int  # training rows dropped because their label only became known after the cutoff


def temporal_split(table: pd.DataFrame, test_fraction: float = 0.2) -> TemporalSplit:
    """Train on the past, test on the future.

    Rows starting at or after the cutoff are the test set. A training row whose session
    ended *after* the cutoff is purged: at the cutoff its label did not exist yet.
    A random split would let the model train on sessions that happen after the ones it
    is tested on, which is not how it will be used.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if table.empty:
        return TemporalSplit(table, table, pd.NaT, 0)
    ordered = table.sort_values("as_of", kind="stable")
    cutoff = ordered["as_of"].iloc[int(len(ordered) * (1 - test_fraction))]
    before = ordered[ordered["as_of"] < cutoff]
    train = before[before["label_available_at"] < cutoff]
    test = ordered[ordered["as_of"] >= cutoff]
    return TemporalSplit(train, test, cutoff, len(before) - len(train))


def describe_features(table: pd.DataFrame) -> pd.DataFrame:
    """Per feature: kind, missing share and range. A pre-modelling sanity check."""
    rows = []
    for spec in FEATURE_SPECS:
        col = pd.to_numeric(table[spec.name], errors="coerce")
        rows.append({
            "feature": spec.name, "kind": spec.kind,
            "missing_share": float(col.isna().mean()) if len(col) else float("nan"),
            "min": col.min(), "median": col.median(), "max": col.max(),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """``python -m ml.features``: feature summary for the configured DB (read-only, no PII)."""
    import sys

    from analytics.dataset import load_dataset
    from analytics.quality import engine_from_env

    engine = engine_from_env()
    if engine is None:
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 1
    with engine.connect() as conn:
        ds = load_dataset(conn)
    table = build_feature_table(ds.logs, ds.rooms)
    split = temporal_split(table)
    print(f"rows: {len(table)}  users: {table['user_id'].nunique()}")
    print(f"temporal split: train {len(split.train)}, test {len(split.test)}, purged {split.purged}, "
          f"cutoff {split.cutoff}")
    if len(table):
        print(f"full-duration rate (target balance): {table['target_full_duration'].mean():.1%}")
    print(describe_features(table).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
