from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import func

from main.db import db
from models.focus import FocusLog
from rooms.models import Room


def fmt_duration(seconds: int) -> str:
    """Convert seconds to human-readable string: 1h 25m, 45m, etc."""
    if seconds <= 0:
        return "0m"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m" if m else f"{h}h"


def get_user_stats(user_id: int) -> dict:
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=6)

    # Total focused seconds (all time)
    total_seconds = (
        db.session.query(func.sum(FocusLog.focused_seconds))
        .filter(FocusLog.user_id == user_id)
        .scalar() or 0
    )

    # This week (last 7 days)
    week_seconds = (
        db.session.query(func.sum(FocusLog.focused_seconds))
        .filter(FocusLog.user_id == user_id, FocusLog.created_at >= week_start)
        .scalar() or 0
    )

    # Today
    today_seconds = (
        db.session.query(func.sum(FocusLog.focused_seconds))
        .filter(FocusLog.user_id == user_id, FocusLog.created_at >= today_start)
        .scalar() or 0
    )

    # Number of completed sessions
    session_count = (
        db.session.query(func.count(FocusLog.id))
        .filter(FocusLog.user_id == user_id)
        .scalar() or 0
    )

    # Per-room breakdown
    per_room_rows = (
        db.session.query(Room.name, func.sum(FocusLog.focused_seconds))
        .join(Room, Room.id == FocusLog.room_id)
        .filter(FocusLog.user_id == user_id)
        .group_by(Room.id, Room.name)
        .order_by(func.sum(FocusLog.focused_seconds).desc())
        .limit(10)
        .all()
    )
    per_room = [(name, int(secs), fmt_duration(int(secs))) for name, secs in per_room_rows]

    # Last 7 days — daily focused minutes (for chart)
    recent_logs = (
        db.session.query(FocusLog.created_at, FocusLog.focused_seconds)
        .filter(FocusLog.user_id == user_id, FocusLog.created_at >= week_start)
        .all()
    )

    daily: dict[str, int] = defaultdict(int)
    for log_dt, secs in recent_logs:
        day_key = log_dt.strftime("%Y-%m-%d")
        daily[day_key] += secs

    chart_labels = []
    chart_values = []
    for i in range(6, -1, -1):
        d = now - timedelta(days=i)
        chart_labels.append(d.strftime("%a"))
        chart_values.append(round(daily.get(d.strftime("%Y-%m-%d"), 0) / 60, 1))

    return {
        "total_seconds": total_seconds,
        "total_fmt": fmt_duration(total_seconds),
        "week_seconds": week_seconds,
        "week_fmt": fmt_duration(week_seconds),
        "today_seconds": today_seconds,
        "today_fmt": fmt_duration(today_seconds),
        "session_count": session_count,
        "per_room": per_room,
        "chart_labels": chart_labels,
        "chart_values": chart_values,
    }
