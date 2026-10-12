"""Print an exploratory analysis of focus data. Aggregates only; users appear as ids.

Usage:
    python -m analytics.eda
    python -m analytics.eda --tz UTC
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from analytics import analysis, statistics
from analytics.cleaning import DEFAULT_LOCAL_TZ
from analytics.dataset import AnalyticsDataset, load_dataset
from analytics.quality import engine_from_env

MIN_SESSIONS_FOR_PATTERNS = 30


def _section(title: str, body: pd.DataFrame | pd.Series | str) -> str:
    text = body if isinstance(body, str) else body.to_string()
    return f"\n== {title} ==\n{text}"


def build_eda(ds: AnalyticsDataset, tz: str = DEFAULT_LOCAL_TZ) -> str:
    summary = analysis.overall_summary(ds.logs)
    parts = [_section("Overall", summary.round(2))]

    if summary["sessions"] < MIN_SESSIONS_FOR_PATTERNS:
        parts.append(
            f"\nOnly {int(summary['sessions'])} valid sessions: too few to read patterns into the tables below."
        )

    distribution = pd.Series(statistics.focus_distribution(ds.logs))
    parts.append(_section("Focused minutes per session (distribution)", distribution.round(2)))

    users = analysis.user_focus_summary(ds.logs, tz)
    parts.append(_section(
        "Top users by total focus",
        users[["user_id", "sessions", "active_days", "total_hours", "mean_minutes", "median_minutes"]]
        .head(10).round(1).to_string(index=False),
    ))

    hours = analysis.focus_by_hour(ds.logs, tz)
    parts.append(_section(
        f"Focus by hour of day ({tz})",
        hours[hours["sessions"] > 0][["hour", "sessions", "total_minutes", "mean_minutes", "share_of_focus"]]
        .round(2).to_string(index=False),
    ))

    weekdays = analysis.focus_by_weekday(ds.logs, tz)
    parts.append(_section(
        "Focus by weekday",
        weekdays[["weekday_name", "sessions", "total_minutes", "mean_minutes", "share_of_focus"]]
        .round(2).to_string(index=False),
    ))

    outcomes = analysis.session_outcomes(ds.sessions, ds.logs)
    parts.append(_section("Session outcomes (derived)", outcomes["outcome"].value_counts()))
    parts.append(_section("Completion rate", analysis.completion_rate(outcomes).round(3).to_string(index=False)))

    parts.append(_section(
        "Planned vs focused duration",
        analysis.session_duration_profile(ds.logs).round(2).to_string(index=False),
    ))

    rooms = analysis.room_activity(ds.logs, ds.rooms, tz)
    parts.append(_section(
        "Most active rooms",
        rooms[["room_id", "is_private", "current_member_count", "sessions", "active_days", "total_hours"]]
        .head(10).round(1).to_string(index=False),
    ))

    daily = analysis.daily_focus(ds.logs, tz=tz)
    last_days = daily.tail(14).assign(date=lambda d: pd.to_datetime(d["date"]).dt.date)
    parts.append(_section("Last 14 days (minutes)", last_days.round(1).to_string(index=False)))
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tz", default=DEFAULT_LOCAL_TZ, help="IANA timezone for hour/weekday analysis")
    args = parser.parse_args(argv)

    engine = engine_from_env()
    if engine is None:
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 1
    with engine.connect() as conn:
        ds = load_dataset(conn)
    print(build_eda(ds, args.tz))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
