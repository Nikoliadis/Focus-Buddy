from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import event

from analytics import data_loader
from main.db import db
from models.chat import ChatMessage
from models.focus import FocusLog, FocusSession
from rooms.models import RoomMember
from rooms.service import create_room


@pytest.fixture()
def count_queries(app):
    """Count SQL statements sent to the database inside the `with` block."""
    class Counter:
        n = 0

    def _before(*_args, **_kwargs):
        Counter.n += 1

    event.listen(db.engine, "before_cursor_execute", _before)
    yield Counter
    event.remove(db.engine, "before_cursor_execute", _before)


@pytest.fixture()
def dataset(make_user):
    """Two users, a public and a private room, sessions, logs and chat."""
    alice, bob = make_user("alice"), make_user("bob")
    public = create_room(alice.id, "Public")
    private = create_room(bob.id, "Private", password="secret")
    db.session.add(RoomMember(room_id=public.id, user_id=bob.id))

    t0 = datetime(2026, 3, 2, 18, 0)  # naive UTC, as the app stores it
    sessions = []
    for i, room in enumerate([public, public, private]):
        s = FocusSession(
            room_id=room.id,
            started_by=room.owner_id,
            status="ended",
            duration_seconds=1500,
            started_at=t0 + timedelta(days=i),
            ended_at=t0 + timedelta(days=i, minutes=25),
            paused_seconds=0,
            created_at=t0 + timedelta(days=i),
        )
        db.session.add(s)
        sessions.append(s)
    db.session.flush()
    for s in sessions:
        db.session.add(FocusLog(
            user_id=s.started_by, room_id=s.room_id, session_id=s.id,
            focused_seconds=1500, created_at=s.ended_at,
        ))

    db.session.add_all([
        ChatMessage(room_id=public.id, user_id=alice.id, username="alice", body="hello", created_at=t0),
        ChatMessage(room_id=private.id, user_id=bob.id, username="bob", body="private note", created_at=t0),
    ])
    db.session.commit()
    return {"alice": alice, "bob": bob, "public": public, "private": private, "t0": t0}


class TestLoadUsers:
    def test_no_pii_columns(self, dataset):
        df = data_loader.load_users()

        assert list(df.columns) == ["user_id", "username", "email_verified", "daily_goal_minutes"]
        assert "email" not in df.columns and "password_hash" not in df.columns
        assert len(df) == 2


class TestLoadRooms:
    def test_member_count_and_privacy(self, dataset):
        df = data_loader.load_rooms().set_index("room_id")

        assert df.loc[dataset["public"].id, "current_member_count"] == 2
        assert df.loc[dataset["private"].id, "current_member_count"] == 1
        assert bool(df.loc[dataset["private"].id, "is_private"]) is True
        assert bool(df.loc[dataset["public"].id, "is_private"]) is False

    def test_room_without_members_has_zero(self, dataset):
        RoomMember.query.filter_by(room_id=dataset["private"].id).delete()
        db.session.commit()

        df = data_loader.load_rooms().set_index("room_id")

        assert df.loc[dataset["private"].id, "current_member_count"] == 0


class TestLoadFocusSessions:
    def test_returns_all_sessions_with_utc_timestamps(self, dataset):
        df = data_loader.load_focus_sessions()

        assert len(df) == 3
        assert str(df["started_at"].dt.tz) == "UTC"
        assert df["started_at"].iloc[0] == pd.Timestamp("2026-03-02 18:00", tz="UTC")

    def test_time_window_is_half_open(self, dataset):
        t0 = dataset["t0"]
        df = data_loader.load_focus_sessions(start=t0, end=t0 + timedelta(days=1))
        assert len(df) == 1

    def test_accepts_timezone_aware_bounds(self, dataset):
        start = datetime(2026, 3, 2, 20, 0, tzinfo=UTC)  # after the first session
        df = data_loader.load_focus_sessions(start=start)
        assert len(df) == 2

    def test_filter_by_room(self, dataset):
        df = data_loader.load_focus_sessions(room_ids=[dataset["private"].id])
        assert set(df["room_id"]) == {dataset["private"].id}

    def test_session_of_deleted_room_is_kept_and_marked(self, dataset):
        orphan_session = FocusSession(
            room_id=99999, started_by=dataset["alice"].id, status="idle", duration_seconds=1500
        )
        db.session.add(orphan_session)
        db.session.commit()

        df = data_loader.load_focus_sessions()

        orphan = df[df["room_id"] == 99999]
        assert len(orphan) == 1
        assert bool(orphan["room_exists"].iloc[0]) is False


class TestLoadFocusLogs:
    def test_joined_with_session(self, dataset):
        df = data_loader.load_focus_logs()

        assert len(df) == 3
        assert (df["session_duration_seconds"] == 1500).all()
        assert df["logged_at"].notna().all()

    def test_filter_by_user(self, dataset):
        df = data_loader.load_focus_logs(user_ids=[dataset["bob"].id])
        assert set(df["user_id"]) == {dataset["bob"].id}


class TestLoadChat:
    def test_private_rooms_excluded_by_default(self, dataset):
        df = data_loader.load_chat_messages()

        assert df["body"].tolist() == ["hello"]
        assert "username" not in df.columns

    def test_private_rooms_included_on_request_and_marked(self, dataset):
        df = data_loader.load_chat_messages(include_private=True).set_index("body")

        assert bool(df.loc["private note", "room_is_private"]) is True


@pytest.mark.parametrize(
    "loader",
    [
        data_loader.load_users,
        data_loader.load_rooms,
        data_loader.load_focus_sessions,
        data_loader.load_focus_logs,
        data_loader.load_chat_messages,
    ],
)
def test_each_loader_runs_exactly_one_query(dataset, count_queries, loader):
    """No N+1: the row count must not change how many queries run."""
    count_queries.n = 0
    loader()
    assert count_queries.n == 1


@pytest.mark.parametrize(
    "loader",
    [
        data_loader.load_rooms,
        data_loader.load_focus_sessions,
        data_loader.load_focus_logs,
        data_loader.load_chat_messages,
    ],
)
def test_empty_database_gives_empty_frame_with_columns(app, loader):
    df = loader()
    assert df.empty
    assert len(df.columns) > 0


def test_accepts_explicit_connection(dataset):
    with db.engine.connect() as conn:
        df = data_loader.load_users(conn)
    assert len(df) == 2
