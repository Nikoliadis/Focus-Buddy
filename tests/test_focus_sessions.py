from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from models.focus import FocusLog, FocusSession
from rooms.service import create_room
from rooms.sessions_service import (
    end_session,
    get_active_session,
    pause_session,
    reset_session,
    resume_session,
    start_session,
)


def _ago(**kwargs) -> datetime:
    return datetime.utcnow() - timedelta(**kwargs)


class TestRemainingSeconds:
    """Pure timer maths on the model — no DB needed."""

    def test_idle_returns_full_duration(self):
        s = FocusSession(status="idle", duration_seconds=1500, paused_seconds=0)
        assert s.remaining_seconds() == 1500

    def test_running_counts_down(self):
        s = FocusSession(status="running", duration_seconds=1500, started_at=_ago(seconds=600), paused_seconds=0)
        assert s.remaining_seconds() == pytest.approx(900, abs=1)

    def test_paused_time_is_not_counted(self):
        s = FocusSession(status="running", duration_seconds=1500, started_at=_ago(seconds=600), paused_seconds=300)
        assert s.remaining_seconds() == pytest.approx(1200, abs=1)

    def test_paused_session_is_frozen(self):
        s = FocusSession(
            status="paused",
            duration_seconds=1500,
            started_at=_ago(seconds=600),
            paused_at=_ago(seconds=500),
            paused_seconds=0,
        )
        assert s.remaining_seconds() == pytest.approx(1400, abs=1)

    def test_never_negative(self):
        s = FocusSession(status="running", duration_seconds=60, started_at=_ago(hours=2), paused_seconds=0)
        assert s.remaining_seconds() == 0

    def test_ended_is_zero(self):
        s = FocusSession(status="ended", duration_seconds=1500, paused_seconds=0)
        assert s.remaining_seconds() == 0


@pytest.fixture()
def room(make_user):
    return create_room(make_user("alice").id, "Study")


class TestSessionLifecycle:
    def test_start_creates_running_session(self, room):
        s = start_session(room.id, room.owner_id, 1500)

        assert s.status == "running"
        assert get_active_session(room.id).id == s.id

    def test_starting_again_ends_previous_session(self, room):
        first = start_session(room.id, room.owner_id, 1500)
        second = start_session(room.id, room.owner_id, 1500)

        assert first.status == "ended"
        assert get_active_session(room.id).id == second.id

    def test_pause_then_resume_accumulates_paused_seconds(self, room):
        s = start_session(room.id, room.owner_id, 1500)
        pause_session(s)
        assert s.status == "paused"

        s.paused_at = _ago(seconds=120)
        resume_session(s)

        assert s.status == "running"
        assert s.paused_at is None
        assert s.paused_seconds == pytest.approx(120, abs=1)

    def test_pause_is_noop_when_not_running(self, room):
        s = start_session(room.id, room.owner_id, 1500)
        pause_session(s)
        paused_at = s.paused_at

        pause_session(s)

        assert s.paused_at == paused_at

    def test_end_logs_focused_time(self, room):
        s = start_session(room.id, room.owner_id, 1500)
        s.started_at = _ago(seconds=600)

        end_session(s, room.owner_id)

        log = FocusLog.query.one()
        assert s.status == "ended"
        assert log.session_id == s.id
        assert log.focused_seconds == pytest.approx(600, abs=1)
        assert get_active_session(room.id) is None

    def test_end_is_idempotent(self, room):
        s = start_session(room.id, room.owner_id, 1500)
        end_session(s, room.owner_id)
        end_session(s, room.owner_id)

        assert FocusLog.query.count() == 1

    def test_reset_returns_to_idle(self, room):
        s = start_session(room.id, room.owner_id, 1500)
        s.started_at = _ago(seconds=600)

        reset_session(s, room.owner_id)

        assert s.status == "idle"
        assert s.remaining_seconds() == 1500


class TestSessionRoutes:
    def test_owner_can_start_session(self, client, login, post_form, room, make_user):
        from models.user import User
        login(User.query.get(room.owner_id))

        post_form(f"/rooms/{room.id}/session/start", {"minutes": "50", "break_minutes": "10"})

        s = get_active_session(room.id)
        assert s.duration_seconds == 50 * 60

        data = client.get(f"/rooms/{room.id}/session").get_json()
        assert data["status"] == "running"

    def test_duration_is_clamped(self, login, post_form, room):
        from models.user import User
        login(User.query.get(room.owner_id))

        post_form(f"/rooms/{room.id}/session/start", {"minutes": "9999"})

        assert get_active_session(room.id).duration_seconds == 180 * 60

    def test_owner_controls_full_lifecycle(self, client, login, post_form, room):
        from models.user import User
        login(User.query.get(room.owner_id))
        post_form(f"/rooms/{room.id}/session/start", {"minutes": "25"})

        post_form(f"/rooms/{room.id}/session/pause")
        assert client.get(f"/rooms/{room.id}/session").get_json()["status"] == "paused"

        post_form(f"/rooms/{room.id}/session/resume")
        assert client.get(f"/rooms/{room.id}/session").get_json()["status"] == "running"

        post_form(f"/rooms/{room.id}/session/end")
        assert client.get(f"/rooms/{room.id}/session").get_json()["status"] == "ended"
        assert FocusLog.query.count() == 1

        post_form(f"/rooms/{room.id}/session/reset")
        assert client.get(f"/rooms/{room.id}/session").get_json()["status"] == "idle"

    def test_idle_room_reports_default_timer(self, client, login, room):
        from models.user import User
        login(User.query.get(room.owner_id))

        data = client.get(f"/rooms/{room.id}/session").get_json()

        assert data["status"] == "idle"
        assert data["remaining_seconds"] == 25 * 60

    @pytest.mark.parametrize("action", ["pause", "resume", "reset", "end"])
    def test_non_owner_cannot_control_session(self, login, post_form, room, make_user, action):
        start_session(room.id, room.owner_id, 1500)
        login(make_user("bob"))

        post_form(f"/rooms/{room.id}/session/{action}")

        assert get_active_session(room.id).status == "running"

    def test_non_owner_cannot_start_session(self, login, post_form, room, make_user):
        login(make_user("bob"))

        post_form(f"/rooms/{room.id}/session/start", {"minutes": "25"})

        assert get_active_session(room.id) is None
