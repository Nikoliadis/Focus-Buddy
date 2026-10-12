"""Data-quality rules for the raw frames returned by ``analytics.data_loader``.

Philosophy: *flag, don't silently drop*. Every row gets a ``quality`` label so analysts
can see exactly what was excluded and why. Only true duplicates and empty chat messages
are removed, and those removals are counted in the returned ``QualityReport``.

Each cleaner is a pure function (DataFrame in, DataFrame out), which makes the rules
easy to unit-test without a database.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OK = "ok"
INVALID_DURATION = "invalid_duration"
INVALID_TIMESTAMP = "invalid_timestamp"
INVALID_STATUS = "invalid_status"
NEVER_STARTED = "never_started"   # idle row: created, or reset back to idle
IN_PROGRESS = "in_progress"       # running/paused and could still legitimately be active
ABANDONED = "abandoned"           # running/paused long after it should have finished

SESSION_STATUSES = {"idle", "running", "paused", "ended"}

# Mirrors the limits enforced by the app (rooms/routes.py, rooms/sockets.py): 1..180 minutes.
MIN_DURATION_SECONDS = 60
MAX_DURATION_SECONDS = 180 * 60
# How long after its expected end a running/paused session still counts as "in progress".
DEFAULT_ABANDON_GRACE = pd.Timedelta(hours=1)
# remaining_seconds() truncates to int, so allow a second of rounding.
_ROUNDING_TOLERANCE_SECONDS = 1

DEFAULT_LOCAL_TZ = "Europe/Athens"


@dataclass
class QualityReport:
    """What a cleaner did to a dataset. Counts only; never row contents."""

    dataset: str
    rows_in: int
    rows_out: int
    dropped: dict[str, int] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)

    @property
    def usable_rows(self) -> int:
        return self.flags.get(OK, 0)

    def __str__(self) -> str:
        lines = [f"{self.dataset}: {self.rows_in} in -> {self.rows_out} out ({self.usable_rows} usable)"]
        lines += [f"  dropped {name}: {n}" for name, n in self.dropped.items() if n]
        lines += [f"  flag {name}: {n}" for name, n in sorted(self.flags.items())]
        return "\n".join(lines)


def _now_utc(now: pd.Timestamp | None) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp.now(tz="UTC")
    return now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")


def _apply_flags(df: pd.DataFrame, rules: list[tuple[str, pd.Series]]) -> pd.Series:
    """First matching rule wins, so each row carries exactly one, most severe, flag."""
    conditions = [mask.fillna(False).to_numpy(dtype=bool) for _, mask in rules]
    labels = [label for label, _ in rules]
    return pd.Series(np.select(conditions, labels, default=OK), index=df.index, dtype="object")


def _flag_counts(quality: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in quality.value_counts().items()}


def clean_focus_sessions(
    df: pd.DataFrame,
    now: pd.Timestamp | None = None,
    abandon_grace: pd.Timedelta = DEFAULT_ABANDON_GRACE,
) -> tuple[pd.DataFrame, QualityReport]:
    """Validate room-level sessions and classify unfinished ones.

    Adds:
      quality         one of the flag constants above
      paused_seconds  NULL -> 0 (the column is NOT NULL in the model; this guards old rows)
      wall_seconds    ended_at - started_at, for ended sessions
    """
    now = _now_utc(now)
    rows_in = len(df)
    out = df.drop_duplicates(subset="session_id", keep="first").copy()
    dropped = {"duplicate_session_id": rows_in - len(out)}

    out["paused_seconds"] = pd.to_numeric(out["paused_seconds"], errors="coerce").fillna(0)
    duration = pd.to_numeric(out["duration_seconds"], errors="coerce")
    started, ended, paused_at = out["started_at"], out["ended_at"], out["paused_at"]
    status = out["status"]
    active = status.isin(["running", "paused"])

    # When should this session have finished at the latest?
    # paused: the clock stopped at paused_at; running: start + duration + time spent paused.
    expected_end = started + pd.to_timedelta(duration, unit="s") + pd.to_timedelta(out["paused_seconds"], unit="s")
    last_activity = paused_at.where(status == "paused", expected_end)

    rules = [
        (INVALID_STATUS, ~status.isin(SESSION_STATUSES)),
        (INVALID_DURATION, duration.isna() | (duration < MIN_DURATION_SECONDS) | (duration > MAX_DURATION_SECONDS)),
        (INVALID_DURATION, out["paused_seconds"] < 0),
        (NEVER_STARTED, (status == "idle") | started.isna()),
        (INVALID_TIMESTAMP, started > now),
        (INVALID_TIMESTAMP, (status == "ended") & ended.isna()),
        (INVALID_TIMESTAMP, ended.notna() & ((ended < started) | (ended > now))),
        (INVALID_TIMESTAMP, (status == "paused") & paused_at.isna()),
        (ABANDONED, active & (last_activity + abandon_grace < now)),
        (IN_PROGRESS, active),
    ]
    out["quality"] = _apply_flags(out, rules)
    out["wall_seconds"] = (ended - started).dt.total_seconds().where(status == "ended")

    return out, QualityReport("focus_sessions", rows_in, len(out), dropped, _flag_counts(out["quality"]))


def clean_focus_logs(
    df: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    """Validate credited focus logs.

    Adds:
      quality           flag
      completion_ratio  focused_seconds / session_duration_seconds, clipped to [0, 1]

    ``completion_ratio`` is descriptive only. It is *not* a completion label: the app does
    not record why a session ended (finished, ended early, or a new one started), and
    break sessions are logged too.
    """
    now = _now_utc(now)
    rows_in = len(df)
    out = df.drop_duplicates(subset="log_id", keep="first")
    dup_ids = rows_in - len(out)
    # end_session() is idempotent, but guard against the same user being credited twice.
    out = (
        out.sort_values(["logged_at", "log_id"])
        .drop_duplicates(subset=["session_id", "user_id"], keep="first")
        .copy()
    )
    dropped = {"duplicate_log_id": dup_ids, "duplicate_session_user": rows_in - dup_ids - len(out)}

    focused = pd.to_numeric(out["focused_seconds"], errors="coerce")
    duration = pd.to_numeric(out["session_duration_seconds"], errors="coerce")

    rules = [
        (INVALID_DURATION, focused.isna() | (focused < 0)),
        (INVALID_DURATION, duration.isna() | (duration < MIN_DURATION_SECONDS) | (duration > MAX_DURATION_SECONDS)),
        (INVALID_DURATION, focused > duration + _ROUNDING_TOLERANCE_SECONDS),
        (INVALID_TIMESTAMP, out["logged_at"].isna() | (out["logged_at"] > now)),
    ]
    out["quality"] = _apply_flags(out, rules)
    out["completion_ratio"] = (focused / duration).clip(0, 1).where(out["quality"] == OK)

    return out, QualityReport("focus_logs", rows_in, len(out), dropped, _flag_counts(out["quality"]))


def clean_chat_messages(df: pd.DataFrame, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, QualityReport]:
    """Normalise whitespace, drop empty and exact-duplicate messages.

    Adds ``n_chars`` and ``n_words`` (whitespace tokens; real tokenization comes in the NLP phase).
    """
    now = _now_utc(now)
    rows_in = len(df)
    out = df.copy()
    out["body"] = out["body"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()

    empty = out["body"].isna() | (out["body"] == "")
    out = out[~empty]
    after_empty = len(out)
    # Same person, same room, same text, same instant: a double submit, not two messages.
    out = out.drop_duplicates(subset=["user_id", "room_id", "body", "created_at"], keep="first").copy()
    dropped = {"empty_body": rows_in - after_empty, "duplicate_message": after_empty - len(out)}

    rules = [(INVALID_TIMESTAMP, out["created_at"].isna() | (out["created_at"] > now))]
    out["quality"] = _apply_flags(out, rules)
    out["n_chars"] = out["body"].str.len().astype("Int64")
    out["n_words"] = out["body"].str.split().str.len().astype("Int64")

    return out, QualityReport("chat_messages", rows_in, len(out), dropped, _flag_counts(out["quality"]))


def add_local_time(df: pd.DataFrame, columns: Iterable[str], tz: str = DEFAULT_LOCAL_TZ) -> pd.DataFrame:
    """Add ``<col>_local`` columns. Hour-of-day features must use local time:
    a 21:00 Athens study session is stored as 18:00 or 19:00 UTC depending on DST."""
    out = df.copy()
    for col in columns:
        out[f"{col}_local"] = out[col].dt.tz_convert(tz)
    return out
