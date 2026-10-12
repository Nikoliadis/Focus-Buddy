"""Extract FocusBuddy data from the database into pandas DataFrames.

Design rules:
- Reuse the existing SQLAlchemy models; no second ORM layer.
- One SQL query per loader. Joins and aggregates run inside the database, never
  row-by-row in Python, so there is no N+1 problem.
- Loaders return typed but *uncleaned* data. Validation lives in ``analytics.cleaning``
  so that data-quality decisions are explicit and testable.
- All timestamps come back as timezone-aware UTC (the app stores naive UTC).
- No PII: emails and password hashes are never selected.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import Select, func, select
from sqlalchemy.engine import Connection, Engine

from main.db import db
from models.chat import ChatMessage
from models.focus import FocusLog, FocusSession
from models.user import User
from rooms.models import Room, RoomMember

Bind = Engine | Connection


@contextmanager
def _connection(bind: Bind | None) -> Iterator[Connection]:
    if isinstance(bind, Connection):
        yield bind
        return
    engine = bind if bind is not None else db.engine
    with engine.connect() as conn:
        yield conn


def _read(stmt: Select, bind: Bind | None, datetime_cols: Iterable[str] = ()) -> pd.DataFrame:
    # Execute through SQLAlchemy rather than pd.read_sql: read_sql issues an extra
    # table-lookup query on PostgreSQL and couples pandas to the SQLAlchemy version.
    with _connection(bind) as conn:
        result = conn.execute(stmt)
        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    for col in datetime_cols:
        # errors="coerce": unparseable timestamps become NaT and are flagged later by cleaning.
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def _naive_utc(value: datetime | None) -> datetime | None:
    """The DB stores naive UTC, so compare against naive UTC."""
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _window(stmt: Select, column, start: datetime | None, end: datetime | None) -> Select:
    """Half-open time window [start, end)."""
    start, end = _naive_utc(start), _naive_utc(end)
    if start is not None:
        stmt = stmt.where(column >= start)
    if end is not None:
        stmt = stmt.where(column < end)
    return stmt


def load_users(bind: Bind | None = None) -> pd.DataFrame:
    stmt = select(
        User.id.label("user_id"),
        User.username,
        User.email_verified,
        User.daily_goal_minutes,
    ).order_by(User.id)
    return _read(stmt, bind)


def load_rooms(bind: Bind | None = None) -> pd.DataFrame:
    """Rooms with their *current* member count.

    ``current_member_count`` is a snapshot of today, not of when past sessions happened
    (members are hard-deleted on leave/kick). Using it as a feature for historical
    sessions would leak present-day information into the past.
    """
    member_counts = (
        select(RoomMember.room_id, func.count(RoomMember.id).label("current_member_count"))
        .group_by(RoomMember.room_id)
        .subquery()
    )
    stmt = (
        select(
            Room.id.label("room_id"),
            Room.owner_id,
            Room.password_hash.is_not(None).label("is_private"),
            Room.created_at,
            func.coalesce(member_counts.c.current_member_count, 0).label("current_member_count"),
        )
        .outerjoin(member_counts, member_counts.c.room_id == Room.id)
        .order_by(Room.id)
    )
    return _read(stmt, bind, ["created_at"])


def load_focus_sessions(
    bind: Bind | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    room_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """One row per room-level Pomodoro session (focus *or* break; phase is not stored).

    ``room_exists`` is False when the room was deleted: ``focus_sessions.room_id`` has
    no foreign key, so these rows survive their room.
    """
    stmt = (
        select(
            FocusSession.id.label("session_id"),
            FocusSession.room_id,
            FocusSession.started_by,
            FocusSession.status,
            FocusSession.duration_seconds,
            FocusSession.started_at,
            FocusSession.ended_at,
            FocusSession.paused_at,
            FocusSession.paused_seconds,
            FocusSession.created_at,
            Room.id.is_not(None).label("room_exists"),
        )
        .outerjoin(Room, Room.id == FocusSession.room_id)
        .order_by(FocusSession.created_at, FocusSession.id)
    )
    stmt = _window(stmt, FocusSession.created_at, start, end)
    if room_ids is not None:
        stmt = stmt.where(FocusSession.room_id.in_(list(room_ids)))
    return _read(stmt, bind, ["started_at", "ended_at", "paused_at", "created_at"])


def load_focus_logs(
    bind: Bind | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    user_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """One row per credited focus log, joined with the session it came from.

    Known limitation of the current schema: logs are only written for the user who ended
    the session (the room owner), and break sessions are logged too.
    """
    stmt = (
        select(
            FocusLog.id.label("log_id"),
            FocusLog.user_id,
            FocusLog.room_id,
            FocusLog.session_id,
            FocusLog.focused_seconds,
            FocusLog.created_at.label("logged_at"),
            FocusSession.duration_seconds.label("session_duration_seconds"),
            FocusSession.started_at.label("session_started_at"),
            FocusSession.ended_at.label("session_ended_at"),
            FocusSession.paused_seconds.label("session_paused_seconds"),
            FocusSession.started_by.label("session_started_by"),
        )
        .join(FocusSession, FocusSession.id == FocusLog.session_id)
        .order_by(FocusLog.created_at, FocusLog.id)
    )
    stmt = _window(stmt, FocusLog.created_at, start, end)
    if user_ids is not None:
        stmt = stmt.where(FocusLog.user_id.in_(list(user_ids)))
    return _read(stmt, bind, ["logged_at", "session_started_at", "session_ended_at"])


def load_chat_messages(
    bind: Bind | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    room_ids: Iterable[int] | None = None,
    include_private: bool = False,
) -> pd.DataFrame:
    """Chat messages for NLP.

    Private (password-protected) rooms are excluded unless ``include_private=True``:
    their content must not leak into corpus-wide analyses. Usernames are not selected;
    ``user_id`` is enough for analysis.
    """
    stmt = (
        select(
            ChatMessage.id.label("message_id"),
            ChatMessage.room_id,
            ChatMessage.user_id,
            ChatMessage.body,
            ChatMessage.created_at,
            Room.password_hash.is_not(None).label("room_is_private"),
        )
        .join(Room, Room.id == ChatMessage.room_id)
        .order_by(ChatMessage.created_at, ChatMessage.id)
    )
    stmt = _window(stmt, ChatMessage.created_at, start, end)
    if room_ids is not None:
        stmt = stmt.where(ChatMessage.room_id.in_(list(room_ids)))
    if not include_private:
        stmt = stmt.where(Room.password_hash.is_(None))
    return _read(stmt, bind, ["created_at"])
