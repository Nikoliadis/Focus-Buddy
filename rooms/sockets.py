from __future__ import annotations

from datetime import datetime

from flask import session
from flask_socketio import join_room, leave_room, emit

from main.db import db
from main.socketio_ext import socketio
from models.chat import ChatMessage
from .models import RoomMember
from .service import get_room
from .presence_store import PRESENCE, LAST_SEEN, ROOM_CYCLES
from .sessions_service import (
    get_active_session,
    start_session,
    pause_session,
    resume_session,
    reset_session,
    end_session,
)


def _room_key(room_id: int) -> str:
    return f"room:{room_id}"


def _broadcast_presence(room_id: int) -> None:
    online_ids = sorted(list(PRESENCE.get(room_id, set())))
    last_seen = {
        str(uid): ts.isoformat()
        for uid, ts in LAST_SEEN.items()
        if uid not in PRESENCE.get(room_id, set())
    }
    emit(
        "presence:update",
        {"room_id": room_id, "count": len(online_ids), "online_ids": online_ids, "last_seen": last_seen},
        room=_room_key(room_id),
    )


def _is_member(room_id: int, user_id: int) -> bool:
    return RoomMember.query.filter_by(room_id=room_id, user_id=user_id).first() is not None


def _is_owner(room_id: int, user_id: int) -> bool:
    room = get_room(room_id)
    return bool(room and room.owner_id == user_id)


def _session_payload(room_id: int):
    s = get_active_session(room_id)
    cycle = ROOM_CYCLES.get(room_id, {"phase": "focus", "count": 1})
    if not s:
        return {"status": "idle", "remaining_seconds": 25 * 60, "phase": cycle["phase"], "cycle_count": cycle["count"]}
    return {
        "status": s.status,
        "remaining_seconds": s.remaining_seconds(),
        "duration_seconds": s.duration_seconds,
        "started_by": s.started_by,
        "phase": cycle["phase"],
        "cycle_count": cycle["count"],
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

    uid = int(user_id)
    if room_id in PRESENCE and uid in PRESENCE[room_id]:
        PRESENCE[room_id].remove(uid)
        if not PRESENCE[room_id]:
            PRESENCE.pop(room_id, None)
        LAST_SEEN[uid] = datetime.utcnow()
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

    LAST_SEEN[uid] = datetime.utcnow()


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
    break_minutes = max(1, min(int(data.get("break_minutes") or 5), 60))
    ROOM_CYCLES[room_id] = {"phase": "focus", "count": 1, "break_minutes": break_minutes}
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


@socketio.on("timer:cycle_next")
def on_timer_cycle_next(data):
    room_id = int(data.get("room_id") or 0)
    user_id = session.get("user_id")
    if not user_id or not room_id:
        return
    if not _require_owner(room_id, int(user_id)):
        return

    # End current session
    s = get_active_session(room_id)
    if s:
        end_session(s, int(user_id))

    # Advance cycle
    current = ROOM_CYCLES.get(room_id, {"phase": "focus", "count": 1, "break_minutes": 5})
    break_minutes = current.get("break_minutes", 5)
    if current["phase"] == "focus":
        next_phase = "break"
        next_duration = break_minutes * 60
        next_count = current["count"]
    else:
        next_phase = "focus"
        next_duration = 25 * 60
        next_count = current["count"] + 1

    ROOM_CYCLES[room_id] = {"phase": next_phase, "count": next_count, "break_minutes": break_minutes}
    start_session(room_id, int(user_id), next_duration)

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


# ── Chat ─────────────────────────────────────────────────────────────────────

@socketio.on("chat:message")
def on_chat_message(data):
    room_id = int(data.get("room_id") or 0)
    user_id = session.get("user_id")
    body = (data.get("body") or "").strip()

    if not user_id or not room_id or not body:
        return
    if len(body) > 500:
        body = body[:500]
    if not _is_member(room_id, int(user_id)):
        return

    username = session.get("username", "?")
    msg = ChatMessage(
        room_id=room_id,
        user_id=int(user_id),
        username=username,
        body=body,
    )
    db.session.add(msg)
    db.session.commit()

    emit(
        "chat:message",
        {
            "id": msg.id,
            "username": username,
            "user_id": int(user_id),
            "body": body,
            "ts": msg.created_at.strftime("%H:%M"),
        },
        room=_room_key(room_id),
    )


@socketio.on("chat:history")
def on_chat_history(data):
    room_id = int(data.get("room_id") or 0)
    user_id = session.get("user_id")
    if not user_id or not room_id:
        return
    if not _is_member(room_id, int(user_id)):
        return

    msgs = (
        ChatMessage.query
        .filter_by(room_id=room_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(50)
        .all()
    )
    emit("chat:history", {
        "messages": [
            {
                "id": m.id,
                "username": m.username,
                "user_id": m.user_id,
                "body": m.body,
                "ts": m.created_at.strftime("%H:%M"),
            }
            for m in msgs
        ]
    })