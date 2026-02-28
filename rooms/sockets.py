from __future__ import annotations

from datetime import datetime
from typing import Dict, Set

from flask import session
from flask_socketio import join_room, leave_room, emit

from main.socketio_ext import socketio
from .models import RoomMember
from .service import get_room
from .sessions_service import (
    get_active_session,
    start_session,
    pause_session,
    resume_session,
    reset_session,
    end_session,
)

PRESENCE: Dict[int, Set[int]] = {}


def _room_key(room_id: int) -> str:
    return f"room:{room_id}"


def _broadcast_presence(room_id: int) -> None:
    users = sorted(list(PRESENCE.get(room_id, set())))
    emit(
        "presence:update",
        {"room_id": room_id, "count": len(users), "users": users},
        room=_room_key(room_id),
    )


def _is_member(room_id: int, user_id: int) -> bool:
    return RoomMember.query.filter_by(room_id=room_id, user_id=user_id).first() is not None


def _is_owner(room_id: int, user_id: int) -> bool:
    room = get_room(room_id)
    return bool(room and room.owner_id == user_id)


def _session_payload(room_id: int):
    s = get_active_session(room_id)
    if not s:
        return {"status": "idle", "remaining_seconds": 25 * 60}
    return {
        "status": s.status,
        "remaining_seconds": s.remaining_seconds(),
        "duration_seconds": s.duration_seconds,
        "started_by": s.started_by,
    }


@socketio.on("room:join")
def on_room_join(data):
    room_id = int(data.get("room_id") or 0)
    user_id = session.get("user_id")

    if not user_id or not room_id:
        emit("error", {"message": "Unauthorized"})
        return

    if not _is_member(room_id, user_id):
        emit("error", {"message": "Not a room member"})
        return

    join_room(_room_key(room_id))

    PRESENCE.setdefault(room_id, set()).add(int(user_id))
    _broadcast_presence(room_id)

    # Send current session state to the joining user
    emit("timer:update", {"room_id": room_id, **_session_payload(room_id)})


@socketio.on("room:leave")
def on_room_leave(data):
    room_id = int(data.get("room_id") or 0)
    user_id = session.get("user_id")
    if not user_id or not room_id:
        return

    leave_room(_room_key(room_id))

    if room_id in PRESENCE and int(user_id) in PRESENCE[room_id]:
        PRESENCE[room_id].remove(int(user_id))
        if not PRESENCE[room_id]:
            PRESENCE.pop(room_id, None)
    _broadcast_presence(room_id)


@socketio.on("disconnect")
def on_disconnect():
    user_id = session.get("user_id")
    if not user_id:
        return

    uid = int(user_id)
    empty_rooms = []
    for rid, users in PRESENCE.items():
        if uid in users:
            users.remove(uid)
            _broadcast_presence(rid)
        if not users:
            empty_rooms.append(rid)
    for rid in empty_rooms:
        PRESENCE.pop(rid, None)


def _require_owner(room_id: int, user_id: int) -> bool:
    if not _is_owner(room_id, user_id):
        emit("error", {"message": "Only the room owner can control sessions."})
        return False
    return True


@socketio.on("timer:start")
def on_timer_start(data):
    room_id = int(data.get("room_id") or 0)
    minutes = int(data.get("minutes") or 25)
    user_id = session.get("user_id")

    if not user_id or not room_id:
        emit("error", {"message": "Unauthorized"})
        return

    if not _is_member(room_id, int(user_id)):
        emit("error", {"message": "Not a room member"})
        return

    if not _require_owner(room_id, int(user_id)):
        return

    minutes = max(1, min(minutes, 180))
    start_session(room_id, int(user_id), minutes * 60)

    emit("timer:update", {"room_id": room_id, **_session_payload(room_id)}, room=_room_key(room_id))


@socketio.on("timer:pause")
def on_timer_pause(data):
    room_id = int(data.get("room_id") or 0)
    user_id = session.get("user_id")
    if not user_id or not room_id:
        return
    if not _require_owner(room_id, int(user_id)):
        return

    s = get_active_session(room_id)
    if s:
        pause_session(s)

    emit("timer:update", {"room_id": room_id, **_session_payload(room_id)}, room=_room_key(room_id))


@socketio.on("timer:resume")
def on_timer_resume(data):
    room_id = int(data.get("room_id") or 0)
    user_id = session.get("user_id")
    if not user_id or not room_id:
        return
    if not _require_owner(room_id, int(user_id)):
        return

    s = get_active_session(room_id)
    if s:
        resume_session(s)

    emit("timer:update", {"room_id": room_id, **_session_payload(room_id)}, room=_room_key(room_id))


@socketio.on("timer:reset")
def on_timer_reset(data):
    room_id = int(data.get("room_id") or 0)
    user_id = session.get("user_id")
    if not user_id or not room_id:
        return
    if not _require_owner(room_id, int(user_id)):
        return

    s = get_active_session(room_id) or None
    if s:
        reset_session(s, int(user_id))

    emit("timer:update", {"room_id": room_id, **_session_payload(room_id)}, room=_room_key(room_id))


@socketio.on("timer:end")
def on_timer_end(data):
    room_id = int(data.get("room_id") or 0)
    user_id = session.get("user_id")
    if not user_id or not room_id:
        return
    if not _require_owner(room_id, int(user_id)):
        return

    s = get_active_session(room_id)
    if s:
        end_session(s, int(user_id))

    emit("timer:update", {"room_id": room_id, **_session_payload(room_id)}, room=_room_key(room_id))