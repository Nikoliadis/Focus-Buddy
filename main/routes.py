import csv
import io

from flask import render_template, session, Response

from main.auth_utils import login_required
from main.db import db
from main.stats_service import get_user_stats, get_leaderboard
from models.focus import FocusLog
from rooms.models import Room
from . import main_bp


@main_bp.get("/")
def home():
    return render_template("home.html")


@main_bp.get("/stats")
@login_required
def stats():
    data = get_user_stats(session["user_id"])
    return render_template("stats.html", **data)


@main_bp.get("/stats/export")
@login_required
def stats_export():
    rows = (
        db.session.query(FocusLog, Room.name)
        .join(Room, Room.id == FocusLog.room_id)
        .filter(FocusLog.user_id == session["user_id"])
        .order_by(FocusLog.created_at.desc())
        .all()
    )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "room", "duration_seconds", "duration_minutes"])
    for log, room_name in rows:
        writer.writerow([
            log.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            room_name,
            log.focused_seconds,
            round(log.focused_seconds / 60, 1),
        ])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=focus_stats.csv"},
    )


@main_bp.get("/leaderboard")
@login_required
def leaderboard():
    entries = get_leaderboard()
    return render_template("leaderboard.html", entries=entries)
