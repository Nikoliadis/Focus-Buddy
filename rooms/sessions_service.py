from __future__ import annotations

from datetime import datetime

from main.db import db
from models.focus import FocusSession, FocusLog


def get_active_session(room_id: int) -> FocusSession | None:
    return (
        FocusSession.query
        .filter(
            FocusSession.room_id == room_id,
            FocusSession.status.in_(["running", "paused"])  # ΟΧΙ ended
        )
        .order_by(FocusSession.created_at.desc())
        .first()
    )


def get_latest_session(room_id: int) -> FocusSession | None:
    return (
        FocusSession.query
        .filter(FocusSession.room_id == room_id)
        .order_by(FocusSession.created_at.desc())
        .first()
    )


def start_session(room_id: int, user_id: int, duration_seconds: int) -> FocusSession:
    active = get_active_session(room_id)
    if active:
        end_session(active, user_id)

    s = FocusSession(
        room_id=room_id,
        started_by=user_id,
        status="running",
        duration_seconds=duration_seconds,
        started_at=datetime.utcnow(),
        paused_seconds=0,
        paused_at=None,
        ended_at=None,
    )
    db.session.add(s)
    db.session.commit()
    return s


def pause_session(s: FocusSession) -> FocusSession:
    if s.status != "running":
        return s

    s.status = "paused"
    s.paused_at = datetime.utcnow()
    db.session.commit()
    return s


def resume_session(s: FocusSession) -> FocusSession:
    if s.status != "paused":
        return s

    if s.paused_at:
        s.paused_seconds += int((datetime.utcnow() - s.paused_at).total_seconds())

    s.paused_at = None
    s.status = "running"
    db.session.commit()
    return s


def reset_session(s: FocusSession, user_id: int) -> FocusSession:
    s.status = "idle"
    s.started_by = user_id
    s.started_at = None
    s.ended_at = None
    s.paused_at = None
    s.paused_seconds = 0
    db.session.commit()
    return s


def end_session(s: FocusSession, user_id: int) -> FocusSession:
    if s.status == "ended":
        return s

    remaining = s.remaining_seconds()
    focused = max(0, s.duration_seconds - remaining)

    s.status = "ended"
    s.ended_at = datetime.utcnow()

    # 🔴 ΚΡΙΣΙΜΟ FIX:
    # Αν ήταν paused, αφαιρούμε paused time
    if s.paused_at:
        s.paused_seconds += int((datetime.utcnow() - s.paused_at).total_seconds())
        s.paused_at = None

    db.session.commit()

    log = FocusLog(
        user_id=user_id,
        room_id=s.room_id,
        session_id=s.id,
        focused_seconds=focused,
    )
    db.session.add(log)
    db.session.commit()

    return s