from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import func

from main.db import db
from models.focus import FocusLog
from models.user import User
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

    # Best single day (all time)
    best_day_row = (
        db.session.query(
            func.date(FocusLog.created_at).label("day"),
            func.sum(FocusLog.focused_seconds).label("total")
        )
        .filter(FocusLog.user_id == user_id)
        .group_by(func.date(FocusLog.created_at))
        .order_by(func.sum(FocusLog.focused_seconds).desc())
        .first()
    )
    best_day_fmt = fmt_duration(int(best_day_row.total)) if best_day_row else "—"

    # Recent sessions (last 10)
    recent = (
        db.session.query(FocusLog)
        .filter(FocusLog.user_id == user_id)
        .order_by(FocusLog.created_at.desc())
        .limit(10)
        .all()
    )

    return {
        "total_seconds": total_seconds,
        "best_day_fmt": best_day_fmt,
        "recent_sessions": [(r.focused_seconds, fmt_duration(r.focused_seconds), r.created_at) for r in recent],
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


def get_leaderboard(limit: int = 10) -> list:
    """Global top users by all-time focused seconds."""
    now = datetime.utcnow()
    week_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=6)

    rows = (
        db.session.query(User.username, func.sum(FocusLog.focused_seconds).label("total"))
        .join(FocusLog, FocusLog.user_id == User.id)
        .group_by(User.id, User.username)
        .order_by(func.sum(FocusLog.focused_seconds).desc())
        .limit(limit)
        .all()
    )

    week_rows = (
        db.session.query(User.id, func.sum(FocusLog.focused_seconds).label("week"))
        .join(FocusLog, FocusLog.user_id == User.id)
        .filter(FocusLog.created_at >= week_start)
        .group_by(User.id)
        .all()
    )
    week_map = {uid: secs for uid, secs in week_rows}

    result = []
    for i, (username, total) in enumerate(rows):
        uid_row = db.session.query(User.id).filter(User.username == username).scalar()
        result.append({
            "rank": i + 1,
            "username": username,
            "total_fmt": fmt_duration(int(total)),
            "week_fmt": fmt_duration(int(week_map.get(uid_row, 0))),
        })
    return result
