from __future__ import annotations

import pytest

from main.db import db
from models.chat import ChatMessage
from rooms.models import RoomMember
from rooms.presence_store import PRESENCE
from rooms.service import create_room


def _events(sc, name: str) -> list[dict]:
    return [e["args"][0] for e in sc.get_received() if e["name"] == name]


@pytest.fixture()
def room_setup(make_user, login, client, second_client, socket_client):
    """Alice owns a room, Bob is a member; both are connected over Socket.IO."""
    alice, bob = make_user("alice"), make_user("bob")
    room = create_room(alice.id, "Study")
    db.session.add(RoomMember(room_id=room.id, user_id=bob.id))
    db.session.commit()

    login(alice, client)
    login(bob, second_client)
    alice_sc = socket_client(client)
    bob_sc = socket_client(second_client)
    return room, alice, bob, alice_sc, bob_sc


def _join_both(room, alice_sc, bob_sc):
    alice_sc.emit("room:join", {"room_id": room.id})
    bob_sc.emit("room:join", {"room_id": room.id})
    alice_sc.get_received()
    bob_sc.get_received()


class TestPresence:
    def test_join_updates_presence_and_sends_timer(self, room_setup):
        room, alice, _bob, alice_sc, _bob_sc = room_setup

        alice_sc.emit("room:join", {"room_id": room.id})

        received = alice_sc.get_received()
        names = [e["name"] for e in received]
        assert "presence:update" in names
        assert "timer:update" in names
        assert alice.id in PRESENCE[room.id]

    def test_non_member_cannot_join(self, room_setup, make_user, app, socket_client):
        room = room_setup[0]
        mallory_client = app.test_client()
        with mallory_client.session_transaction() as sess:
            sess["user_id"] = make_user("mallory").id
        mallory_sc = socket_client(mallory_client)

        mallory_sc.emit("room:join", {"room_id": room.id})

        assert _events(mallory_sc, "error") == [{"message": "Not a room member"}]
        assert room.id not in PRESENCE

    def test_leave_removes_presence(self, room_setup):
        room, alice, bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)

        bob_sc.emit("room:leave", {"room_id": room.id})

        assert PRESENCE[room.id] == {alice.id}
        update = _events(alice_sc, "presence:update")[-1]
        assert update["online_ids"] == [alice.id]
        assert str(bob.id) in update["last_seen"]


class TestTimer:
    def test_owner_start_is_broadcast_to_room(self, room_setup):
        room, _alice, _bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)

        alice_sc.emit("timer:start", {"room_id": room.id, "minutes": 30})

        update = _events(bob_sc, "timer:update")[-1]
        assert update["status"] == "running"
        assert update["duration_seconds"] == 30 * 60

    def test_member_cannot_control_timer(self, room_setup):
        room, _alice, _bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)

        bob_sc.emit("timer:start", {"room_id": room.id, "minutes": 30})

        assert _events(bob_sc, "error") == [{"message": "Only the room owner can control sessions."}]
        assert _events(alice_sc, "timer:update") == []

    def test_pause_resume_end(self, room_setup):
        room, _alice, _bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)
        alice_sc.emit("timer:start", {"room_id": room.id, "minutes": 25})

        for event, expected in [("timer:pause", "paused"), ("timer:resume", "running"), ("timer:end", "idle")]:
            alice_sc.emit(event, {"room_id": room.id})
            assert _events(bob_sc, "timer:update")[-1]["status"] == expected

    def test_reset_returns_timer_to_idle(self, room_setup):
        room, _alice, _bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)
        alice_sc.emit("timer:start", {"room_id": room.id, "minutes": 25})

        alice_sc.emit("timer:reset", {"room_id": room.id})

        assert _events(bob_sc, "timer:update")[-1]["status"] == "idle"

    def test_cycle_next_switches_to_break(self, room_setup):
        room, _alice, _bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)
        alice_sc.emit("timer:start", {"room_id": room.id, "minutes": 25, "break_minutes": 5})

        alice_sc.emit("timer:cycle_next", {"room_id": room.id})

        update = _events(bob_sc, "timer:update")[-1]
        assert update["phase"] == "break"
        assert update["duration_seconds"] == 5 * 60


    def test_disconnect_marks_user_offline(self, room_setup):
        room, alice, bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)

        bob_sc.disconnect()

        assert PRESENCE[room.id] == {alice.id}

    def test_presence_endpoint_lists_online_members(self, room_setup, client):
        room, alice, bob, alice_sc, _bob_sc = room_setup
        alice_sc.emit("room:join", {"room_id": room.id})

        data = client.get(f"/rooms/{room.id}/presence").get_json()

        assert data["online_ids"] == [alice.id]
        assert {m["username"] for m in data["members"]} == {"alice", "bob"}


class TestChat:
    def test_message_is_saved_and_broadcast(self, room_setup):
        room, _alice, _bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)

        bob_sc.emit("chat:message", {"room_id": room.id, "body": "  hello!  "})

        msg = _events(alice_sc, "chat:message")[-1]
        assert msg["username"] == "bob"
        assert msg["body"] == "hello!"
        assert ChatMessage.query.one().body == "hello!"

    def test_long_message_is_truncated(self, room_setup):
        room, _alice, _bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)

        bob_sc.emit("chat:message", {"room_id": room.id, "body": "x" * 600})

        assert len(ChatMessage.query.one().body) == 500

    def test_muted_user_cannot_send(self, room_setup):
        room, _alice, bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)
        alice_sc.emit("chat:mute", {"room_id": room.id, "user_id": bob.id})
        bob_sc.get_received()

        bob_sc.emit("chat:message", {"room_id": room.id, "body": "can you hear me?"})

        assert _events(bob_sc, "chat:muted_self")
        assert ChatMessage.query.count() == 0

    def test_unmute_restores_chat(self, room_setup):
        room, _alice, bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)
        alice_sc.emit("chat:mute", {"room_id": room.id, "user_id": bob.id})
        alice_sc.emit("chat:unmute", {"room_id": room.id, "user_id": bob.id})

        bob_sc.emit("chat:message", {"room_id": room.id, "body": "back!"})

        assert ChatMessage.query.one().body == "back!"

    def test_non_member_messages_are_ignored(self, room_setup, make_user, app, socket_client):
        room = room_setup[0]
        mallory_client = app.test_client()
        with mallory_client.session_transaction() as sess:
            sess["user_id"] = make_user("mallory").id
        mallory_sc = socket_client(mallory_client)

        mallory_sc.emit("chat:message", {"room_id": room.id, "body": "spam"})

        assert ChatMessage.query.count() == 0

    def test_only_owner_can_mute(self, room_setup):
        room, alice, _bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)

        bob_sc.emit("chat:mute", {"room_id": room.id, "user_id": alice.id})

        assert _events(bob_sc, "error") == [{"message": "Only the owner can mute users."}]

    def test_history_returns_previous_messages(self, room_setup):
        room, _alice, _bob, alice_sc, bob_sc = room_setup
        _join_both(room, alice_sc, bob_sc)
        alice_sc.emit("chat:message", {"room_id": room.id, "body": "first"})
        alice_sc.emit("chat:message", {"room_id": room.id, "body": "second"})
        bob_sc.get_received()

        bob_sc.emit("chat:history", {"room_id": room.id})

        history = _events(bob_sc, "chat:history")[-1]
        assert [m["body"] for m in history["messages"]] == ["first", "second"]


def test_webrtc_signal_is_tagged_with_sender(room_setup):
    room, _alice, bob, alice_sc, bob_sc = room_setup
    _join_both(room, alice_sc, bob_sc)

    bob_sc.emit("webrtc:signal", {"room_id": room.id, "to_user_id": 1, "sdp": "offer", "from_user_id": 999})

    signal = _events(alice_sc, "webrtc:signal")[-1]
    assert signal["from_user_id"] == bob.id  # client-supplied sender is overwritten
