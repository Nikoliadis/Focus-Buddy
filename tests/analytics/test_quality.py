from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from analytics import quality
from main.db import db


def test_report_on_empty_database(app):
    report = quality.build_report(db.engine)

    assert "users: 0" in report
    assert "focus_sessions: 0 in -> 0 out (0 usable)" in report


def test_report_contains_counts_but_no_message_text(app, make_user):
    from models.chat import ChatMessage
    from rooms.service import create_room

    user = make_user("alice")
    room = create_room(user.id, "Room")
    db.session.add(ChatMessage(room_id=room.id, user_id=user.id, username="alice", body="top secret text"))
    db.session.commit()

    report = quality.build_report(db.engine)

    assert "users: 1" in report
    assert "chat_messages: 1 in -> 1 out (1 usable)" in report
    assert "top secret text" not in report
    assert "alice@example.com" not in report


@pytest.mark.skipif(
    not os.environ["DATABASE_URL"].startswith("postgresql"), reason="read-only mode is a PostgreSQL feature"
)
def test_readonly_engine_rejects_writes(app):
    engine = quality.make_readonly_engine(os.environ["DATABASE_URL"])

    with engine.connect() as conn, pytest.raises(DBAPIError, match="read-only"):
        conn.execute(text("CREATE TABLE should_not_exist (id int)"))
    engine.dispose()
