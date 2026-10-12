from __future__ import annotations

from datetime import datetime, timedelta

from analytics import eda
from analytics.dataset import load_dataset
from main.db import db
from models.focus import FocusLog, FocusSession
from rooms.service import create_room


def test_eda_end_to_end_from_database(app, make_user):
    user = make_user("alice")
    room = create_room(user.id, "Room")
    start = datetime(2026, 3, 2, 7, 0)
    s = FocusSession(
        room_id=room.id, started_by=user.id, status="ended", duration_seconds=1500,
        started_at=start, ended_at=start + timedelta(minutes=25), paused_seconds=0,
    )
    db.session.add(s)
    db.session.flush()
    db.session.add(FocusLog(user_id=user.id, room_id=room.id, session_id=s.id, focused_seconds=1500,
                            created_at=start + timedelta(minutes=25)))
    db.session.commit()

    ds = load_dataset()
    report = eda.build_eda(ds)

    assert len(ds.reports) == 2
    assert "== Focus by hour of day (Europe/Athens) ==" in report
    assert "too few to read patterns" in report      # 1 session is not a pattern
    assert "alice" not in report                     # ids only, no usernames


def test_eda_on_empty_database(app):
    report = eda.build_eda(load_dataset())
    assert "Only 0 valid sessions" in report
