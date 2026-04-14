from __future__ import annotations

from flask import render_template, request, redirect, url_for, flash, session, jsonify

from main.auth_utils import login_required
from main.db import db
from models.user import User

from . import rooms_bp
from .models import RoomMember
from .presence_store import PRESENCE, LAST_SEEN, ROOM_CYCLES
from .service import (
    create_room,
    get_user_rooms,
    get_public_rooms,
    get_room,
    add_member,
    find_room_by_code,
    get_room_members,
    remove_member,
    delete_room,
    verify_room_password,
)
from .sessions_service import (
    get_active_session,
    get_latest_session,
    start_session,
    pause_session,
    resume_session,
    reset_session,
    end_session,
)


@rooms_bp.get("/rooms")
@login_required
def rooms_index():
    user_id = session["user_id"]
    my_rooms = get_user_rooms(user_id)
    my_room_ids = {r["room"].id for r in my_rooms}
    public_rooms = [r for r in get_public_rooms() if r["room"].id not in my_room_ids]
    return render_template("rooms/index.html", my_rooms=my_rooms, public_rooms=public_rooms)


@rooms_bp.post("/rooms/<int:room_id>/join-public")
@login_required
def room_join_public(room_id: int):
    room = get_room(room_id)
    if not room:
        flash("Room not found.", "error")
        return redirect(url_for("rooms.rooms_index"))

    if room.is_private:
        flash("This room is private. Use the join code and password.", "error")
        return redirect(url_for("rooms.rooms_join_page", code=room.join_code or ""))

    add_member(room, session["user_id"])
    flash("Joined room ✅", "success")
    return redirect(url_for("rooms.room_detail", room_id=room.id))


@rooms_bp.get("/rooms/create")
@login_required
def rooms_create_page():
    return render_template("rooms/create.html")


@rooms_bp.post("/rooms/create")
@login_required
def rooms_create_post():
    name = (request.form.get("name") or "").strip()
    with_code = (request.form.get("with_code") == "on")
    password = (request.form.get("password") or "").strip()

    if len(name) < 3 or len(name) > 50:
        flash("Room name must be between 3 and 50 characters.", "error")
        return redirect(url_for("rooms.rooms_create_page"))

    if password and len(password) > 128:
        flash("Password must be 128 characters or fewer.", "error")
        return redirect(url_for("rooms.rooms_create_page"))

    room = create_room(
        owner_id=session["user_id"],
        name=name,
        with_code=with_code,
        password=password or None,
    )

    flash("Room created ✅", "success")
    return redirect(url_for("rooms.room_detail", room_id=room.id))


@rooms_bp.get("/rooms/join")
@login_required
def rooms_join_page():
    prefill = (request.args.get("code") or "").strip().upper()
    return render_template("rooms/join.html", prefill_code=prefill)


@rooms_bp.post("/rooms/join")
@login_required
def rooms_join_post():
    code = (request.form.get("code") or "").strip().upper()
    password = (request.form.get("password") or "").strip()

    if not code:
        flash("Please enter a room code.", "error")
        return redirect(url_for("rooms.rooms_join_page"))

    room = find_room_by_code(code)
    if not room:
        flash("Invalid room code.", "error")
        return redirect(url_for("rooms.rooms_join_page"))

    if not verify_room_password(room, password):
        flash("Incorrect room password.", "error")
        return redirect(url_for("rooms.rooms_join_page", code=code))

    add_member(room, session["user_id"])

    is_member = RoomMember.query.filter_by(room_id=room.id, user_id=session["user_id"]).first()
    if not is_member:
        flash("Could not join the room. Please try again.", "error")
        return redirect(url_for("rooms.rooms_join_page"))

    flash("Joined room ✅", "success")
    return redirect(url_for("rooms.room_detail", room_id=room.id))


@rooms_bp.get("/rooms/<int:room_id>")
@login_required
def room_detail(room_id: int):
    room = get_room(room_id)
    if not room:
        flash("Room not found.", "error")
        return redirect(url_for("rooms.rooms_index"))

    user_id = session["user_id"]

    is_member = RoomMember.query.filter_by(room_id=room.id, user_id=user_id).first()
    if not is_member:
        flash("You are not a member of this room.", "error")
        return redirect(url_for("rooms.rooms_index"))

    members = get_room_members(room)
    is_owner = (user_id == room.owner_id)

    invite_link = url_for(
        "rooms.rooms_join_page",
        code=room.join_code,
        _external=True
    ) if room.join_code else None

    return render_template(
        "rooms/room.html",
        room=room,
        members=members,
        member_count=len(members),
        is_owner=is_owner,
        invite_link=invite_link,
    )


@rooms_bp.post("/rooms/<int:room_id>/leave")
@login_required
def room_leave(room_id: int):
    room = get_room(room_id)
    if not room:
        flash("Room not found.", "error")
        return redirect(url_for("rooms.rooms_index"))

    user_id = session["user_id"]

    if user_id == room.owner_id:
        flash("Owner cannot leave the room. You can delete the room instead.", "error")
        return redirect(url_for("rooms.room_detail", room_id=room.id))

    is_member = RoomMember.query.filter_by(room_id=room.id, user_id=user_id).first()
    if not is_member:
        flash("You are not a member of this room.", "error")
        return redirect(url_for("rooms.rooms_index"))

    remove_member(room.id, user_id)
    flash("You left the room.", "info")
    return redirect(url_for("rooms.rooms_index"))


@rooms_bp.post("/rooms/<int:room_id>/delete")
@login_required
def room_delete(room_id: int):
    room = get_room(room_id)
    if not room:
        flash("Room not found.", "error")
        return redirect(url_for("rooms.rooms_index"))

    user_id = session["user_id"]
    if user_id != room.owner_id:
        flash("Only the owner can delete this room.", "error")
        return redirect(url_for("rooms.room_detail", room_id=room.id))

    delete_room(room.id)
    flash("Room deleted ✅", "success")
    return redirect(url_for("rooms.rooms_index"))


@rooms_bp.get("/rooms/<int:room_id>/session")
@login_required
def room_session_status(room_id: int):
    room = get_room(room_id)
    if not room:
        return jsonify({"error": "Room not found"}), 404

    is_member = RoomMember.query.filter_by(room_id=room_id, user_id=session["user_id"]).first()
    if not is_member:
        return jsonify({"error": "Forbidden"}), 403

    s = get_active_session(room_id) or get_latest_session(room_id)
    cycle = ROOM_CYCLES.get(room_id, {"phase": "focus", "count": 1})

    if not s:
        return jsonify({
            "status": "idle",
            "duration_seconds": 25 * 60,
            "remaining_seconds": 25 * 60,
            "phase": cycle["phase"],
            "cycle_count": cycle["count"],
        })

    return jsonify({
        "status": s.status,
        "duration_seconds": s.duration_seconds,
        "remaining_seconds": s.remaining_seconds(),
        "started_by": s.started_by,
        "phase": cycle["phase"],
        "cycle_count": cycle["count"],
    })


@rooms_bp.post("/rooms/<int:room_id>/session/start")
@login_required
def room_session_start(room_id: int):
    room = get_room(room_id)
    if not room:
        flash("Room not found.", "error")
        return redirect(url_for("rooms.rooms_index"))

    if session["user_id"] != room.owner_id:
        flash("Only the room owner can start a session.", "error")
        return redirect(url_for("rooms.room_detail", room_id=room_id))

    try:
        minutes = int(request.form.get("minutes") or "25")
    except ValueError:
        minutes = 25

    try:
        break_minutes = int(request.form.get("break_minutes") or "5")
    except ValueError:
        break_minutes = 5

    minutes = max(1, min(minutes, 180))
    break_minutes = max(1, min(break_minutes, 60))
    ROOM_CYCLES[room_id] = {"phase": "focus", "count": 1, "break_minutes": break_minutes}
    start_session(room_id=room_id, user_id=session["user_id"], duration_seconds=minutes * 60)

    flash("Focus session started ✅", "success")
    return redirect(url_for("rooms.room_detail", room_id=room_id))


@rooms_bp.post("/rooms/<int:room_id>/session/pause")
@login_required
def room_session_pause(room_id: int):
    room = get_room(room_id)
    if not room or session["user_id"] != room.owner_id:
        flash("Only the room owner can control the session.", "error")
        return redirect(url_for("rooms.room_detail", room_id=room_id))
    s = get_active_session(room_id)
    if s:
        pause_session(s)
        flash("Paused ⏸️", "info")
    return redirect(url_for("rooms.room_detail", room_id=room_id))


@rooms_bp.post("/rooms/<int:room_id>/session/resume")
@login_required
def room_session_resume(room_id: int):
    room = get_room(room_id)
    if not room or session["user_id"] != room.owner_id:
        flash("Only the room owner can control the session.", "error")
        return redirect(url_for("rooms.room_detail", room_id=room_id))
    s = get_active_session(room_id)
    if s:
        resume_session(s)
        flash("Resumed ▶️", "success")
    return redirect(url_for("rooms.room_detail", room_id=room_id))


@rooms_bp.post("/rooms/<int:room_id>/session/reset")
@login_required
def room_session_reset(room_id: int):
    room = get_room(room_id)
    if not room or session["user_id"] != room.owner_id:
        flash("Only the room owner can control the session.", "error")
        return redirect(url_for("rooms.room_detail", room_id=room_id))
    s = get_active_session(room_id) or get_latest_session(room_id)
    if s:
        reset_session(s, session["user_id"])
        flash("Reset 🔄", "info")
    return redirect(url_for("rooms.room_detail", room_id=room_id))


@rooms_bp.post("/rooms/<int:room_id>/session/end")
@login_required
def room_session_end(room_id: int):
    room = get_room(room_id)
    if not room or session["user_id"] != room.owner_id:
        flash("Only the room owner can control the session.", "error")
        return redirect(url_for("rooms.room_detail", room_id=room_id))
    s = get_active_session(room_id) or get_latest_session(room_id)
    if s:
        end_session(s, session["user_id"])
        flash("Session ended ✅", "success")
    return redirect(url_for("rooms.room_detail", room_id=room_id))


@rooms_bp.post("/rooms/<int:room_id>/kick/<int:target_id>")
@login_required
def room_kick(room_id: int, target_id: int):
    room = get_room(room_id)
    if not room:
        return jsonify({"error": "Room not found"}), 404

    if session["user_id"] != room.owner_id:
        return jsonify({"error": "Only the owner can kick members"}), 403

    if target_id == room.owner_id:
        return jsonify({"error": "Cannot kick the owner"}), 400

    is_member = RoomMember.query.filter_by(room_id=room_id, user_id=target_id).first()
    if not is_member:
        return jsonify({"error": "User is not a member"}), 404

    remove_member(room_id, target_id)

    from main.socketio_ext import socketio
    socketio.emit("user:kicked", {"room_id": room_id, "user_id": target_id}, room=f"room:{room_id}")

    return jsonify({"ok": True})


@rooms_bp.get("/rooms/<int:room_id>/presence")
@login_required
def room_presence(room_id: int):
    room = get_room(room_id)
    if not room:
        return jsonify({"error": "Room not found"}), 404

    is_member = RoomMember.query.filter_by(room_id=room_id, user_id=session["user_id"]).first()
    if not is_member:
        return jsonify({"error": "Forbidden"}), 403

    rows = (
        db.session.query(User.username, RoomMember.user_id)
        .join(RoomMember, RoomMember.user_id == User.id)
        .filter(RoomMember.room_id == room_id)
        .order_by(RoomMember.joined_at.asc())
        .all()
    )

    online_ids = list(PRESENCE.get(room_id, set()))
    last_seen = {
        str(uid): ts.isoformat()
        for uid, ts in LAST_SEEN.items()
        if uid not in online_ids
    }
    members = [{"user_id": uid, "username": uname} for (uname, uid) in rows]
    return jsonify({"count": len(online_ids), "members": members, "online_ids": online_ids, "last_seen": last_seen})