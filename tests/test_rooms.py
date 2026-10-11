from __future__ import annotations

from rooms.models import Room, RoomMember
from rooms.service import create_room


def _is_member(room_id: int, user_id: int) -> bool:
    return RoomMember.query.filter_by(room_id=room_id, user_id=user_id).first() is not None


class TestCreateRoom:
    def test_owner_becomes_member(self, login, post_form, make_user):
        owner = make_user("alice")
        login(owner)

        res = post_form("/rooms/create", {"name": "Exam prep", "with_code": "on"})

        room = Room.query.one()
        assert res.status_code == 302
        assert room.owner_id == owner.id
        assert room.join_code and len(room.join_code) == 8
        assert _is_member(room.id, owner.id)

    def test_name_length_is_validated(self, login, post_form, make_user):
        login(make_user("alice"))

        post_form("/rooms/create", {"name": "ab"})
        post_form("/rooms/create", {"name": "x" * 51})

        assert Room.query.count() == 0

    def test_password_makes_room_private(self, make_user):
        room = create_room(make_user("alice").id, "Secret", password="hunter22")
        assert room.is_private is True


class TestJoinRoom:
    def test_join_by_code(self, login, post_form, make_user):
        room = create_room(make_user("alice").id, "Study")
        bob = make_user("bob")
        login(bob)

        post_form("/rooms/join", {"code": room.join_code.lower()})

        assert _is_member(room.id, bob.id)

    def test_invalid_code(self, login, post_form, make_user):
        login(make_user("bob"))
        res = post_form("/rooms/join", {"code": "NOPE1234"})

        assert res.status_code == 302
        assert RoomMember.query.count() == 0

    def test_private_room_requires_correct_password(self, login, post_form, make_user):
        room = create_room(make_user("alice").id, "Secret", password="hunter22")
        bob = make_user("bob")
        login(bob)

        post_form("/rooms/join", {"code": room.join_code, "password": "wrong"})
        assert not _is_member(room.id, bob.id)

        post_form("/rooms/join", {"code": room.join_code, "password": "hunter22"})
        assert _is_member(room.id, bob.id)

    def test_join_public_room(self, login, post_form, make_user):
        room = create_room(make_user("alice").id, "Open room")
        bob = make_user("bob")
        login(bob)

        post_form(f"/rooms/{room.id}/join-public")

        assert _is_member(room.id, bob.id)

    def test_cannot_join_private_room_via_public_endpoint(self, login, post_form, make_user):
        room = create_room(make_user("alice").id, "Secret", password="hunter22")
        bob = make_user("bob")
        login(bob)

        post_form(f"/rooms/{room.id}/join-public")

        assert not _is_member(room.id, bob.id)

    def test_joining_twice_is_idempotent(self, login, post_form, make_user):
        room = create_room(make_user("alice").id, "Study")
        bob = make_user("bob")
        login(bob)

        post_form("/rooms/join", {"code": room.join_code})
        post_form("/rooms/join", {"code": room.join_code})

        assert RoomMember.query.filter_by(room_id=room.id, user_id=bob.id).count() == 1


class TestRoomAccess:
    def test_member_can_view_room(self, client, login, make_user):
        alice = make_user("alice")
        room = create_room(alice.id, "Study")
        login(alice)

        res = client.get(f"/rooms/{room.id}")

        assert res.status_code == 200
        assert b"Study" in res.data

    def test_non_member_is_redirected(self, client, login, make_user):
        room = create_room(make_user("alice").id, "Study")
        login(make_user("mallory"))

        res = client.get(f"/rooms/{room.id}")

        assert res.status_code == 302

    def test_non_member_cannot_read_session_or_presence(self, client, login, make_user):
        room = create_room(make_user("alice").id, "Study")
        login(make_user("mallory"))

        assert client.get(f"/rooms/{room.id}/session").status_code == 403
        assert client.get(f"/rooms/{room.id}/presence").status_code == 403

    def test_rooms_index_lists_public_rooms(self, client, login, make_user):
        create_room(make_user("alice").id, "Public room")
        create_room(make_user("carol").id, "Hidden room", password="pw")
        login(make_user("bob"))

        res = client.get("/rooms")

        assert b"Public room" in res.data
        assert b"Hidden room" not in res.data


class TestMembership:
    def test_member_can_leave(self, login, post_form, make_user):
        room = create_room(make_user("alice").id, "Study")
        bob = make_user("bob")
        login(bob)
        post_form("/rooms/join", {"code": room.join_code})

        post_form(f"/rooms/{room.id}/leave")

        assert not _is_member(room.id, bob.id)

    def test_owner_cannot_leave(self, login, post_form, make_user):
        alice = make_user("alice")
        room = create_room(alice.id, "Study")
        login(alice)

        post_form(f"/rooms/{room.id}/leave")

        assert _is_member(room.id, alice.id)

    def test_only_owner_can_delete(self, login, post_form, make_user, second_client):
        alice = make_user("alice")
        room = create_room(alice.id, "Study")

        login(make_user("bob"), second_client)
        post_form(f"/rooms/{room.id}/delete", c=second_client)
        assert Room.query.count() == 1

        login(alice)
        post_form(f"/rooms/{room.id}/delete")
        assert Room.query.count() == 0

    def test_owner_can_kick_member(self, client, login, make_user):
        alice = make_user("alice")
        bob = make_user("bob")
        room = create_room(alice.id, "Study")
        RoomMember.query.session.add(RoomMember(room_id=room.id, user_id=bob.id))
        RoomMember.query.session.commit()
        login(alice)

        res = client.post(f"/rooms/{room.id}/kick/{bob.id}", headers={"X-CSRF-Token": "test-csrf-token"})

        assert res.status_code == 200
        assert not _is_member(room.id, bob.id)

    def test_non_owner_cannot_kick(self, client, login, make_user):
        alice = make_user("alice")
        bob = make_user("bob")
        room = create_room(alice.id, "Study")
        RoomMember.query.session.add(RoomMember(room_id=room.id, user_id=bob.id))
        RoomMember.query.session.commit()
        login(bob)

        res = client.post(f"/rooms/{room.id}/kick/{alice.id}", headers={"X-CSRF-Token": "test-csrf-token"})

        assert res.status_code == 403
        assert _is_member(room.id, alice.id)


def test_user_rooms_report_total_member_count(make_user):
    from rooms.service import add_member, get_user_rooms

    alice = make_user("alice")
    room = create_room(alice.id, "Study")
    for name in ("bob", "carol"):
        add_member(room, make_user(name).id)

    [entry] = get_user_rooms(alice.id)

    assert entry["member_count"] == 3
