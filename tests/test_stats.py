from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from main.db import db
from main.stats_service import fmt_duration, get_leaderboard, get_user_stats
from models.focus import FocusLog, FocusSession
from rooms.service import create_room


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0m"), (-5, "0m"), (45, "45s"), (60, "1m"), (1500, "25m"), (3600, "1h"), (5100, "1h 25m")],
)
def test_fmt_duration(seconds, expected):
    assert fmt_duration(seconds) == expected


@pytest.fixture()
def log_focus(make_user):
    """Create a FocusLog `days_ago` days back for the given user."""
    room_cache = {}

    def _log_focus(user, seconds: int, days_ago: int = 0):
        if user.id not in room_cache:
            room_cache[user.id] = create_room(user.id, f"{user.username}'s room")
        room = room_cache[user.id]
        s = FocusSession(room_id=room.id, started_by=user.id, status="ended", duration_seconds=seconds)
        db.session.add(s)
        db.session.flush()
        db.session.add(FocusLog(
            user_id=user.id,
            room_id=room.id,
            session_id=s.id,
            focused_seconds=seconds,
            created_at=datetime.utcnow() - timedelta(days=days_ago),
        ))
        db.session.commit()

    return _log_focus


class TestUserStats:
    def test_empty_user(self, make_user):
        stats = get_user_stats(make_user().id)

        assert stats["total_seconds"] == 0
        assert stats["session_count"] == 0
        assert stats["current_streak"] == 0
        assert stats["chart_values"] == [0] * 7

    def test_totals_and_windows(self, make_user, log_focus):
        user = make_user()
        log_focus(user, 1500, days_ago=0)
        log_focus(user, 1200, days_ago=3)
        log_focus(user, 3000, days_ago=30)

        stats = get_user_stats(user.id)

        assert stats["total_seconds"] == 5700
        assert stats["week_seconds"] == 2700
        assert stats["today_seconds"] == 1500
        assert stats["session_count"] == 3
        assert stats["chart_values"][-1] == 25.0

    def test_streaks(self, make_user, log_focus):
        user = make_user()
        for days_ago in (0, 1, 2):        # current streak: 3
            log_focus(user, 600, days_ago=days_ago)
        for days_ago in (10, 11, 12, 13):  # older, longer streak: 4
            log_focus(user, 600, days_ago=days_ago)

        stats = get_user_stats(user.id)

        assert stats["current_streak"] == 3
        assert stats["longest_streak"] == 4

    def test_streak_broken_if_no_focus_today_or_yesterday(self, make_user, log_focus):
        user = make_user()
        log_focus(user, 600, days_ago=3)

        assert get_user_stats(user.id)["current_streak"] == 0

    def test_daily_goal_progress(self, make_user, log_focus):
        user = make_user()
        user.daily_goal_minutes = 50
        db.session.commit()
        log_focus(user, 1500)

        assert get_user_stats(user.id)["daily_goal_pct"] == 50

    def test_stats_are_per_user(self, make_user, log_focus):
        alice, bob = make_user("alice"), make_user("bob")
        log_focus(alice, 1500)

        assert get_user_stats(bob.id)["total_seconds"] == 0


def test_leaderboard_orders_by_total_focus(make_user, log_focus):
    alice, bob = make_user("alice"), make_user("bob")
    log_focus(alice, 600)
    log_focus(bob, 3600)

    board = get_leaderboard()

    assert [e["username"] for e in board] == ["bob", "alice"]
    assert board[0]["rank"] == 1
    assert board[0]["total_fmt"] == "1h"


def test_csv_export(client, login, make_user, log_focus):
    user = make_user()
    log_focus(user, 1500)
    login(user)

    res = client.get("/stats/export")

    assert res.mimetype == "text/csv"
    lines = res.get_data(as_text=True).strip().splitlines()
    assert lines[0] == "date,room,duration_seconds,duration_minutes"
    assert lines[1].endswith(",1500,25.0")
