from __future__ import annotations

from flask import render_template, request, redirect, url_for, flash, session, jsonify

from main.auth_utils import login_required
from main.db import db
from models.user import User

from . import rooms_bp
from .models import RoomMember
from .service import (
    create_room,
    get_user_rooms,
    get_room,
    add_member,
    find_room_by_code,
    get_room_members,
    remove_member,
    delete_room,
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
    rooms = get_user_rooms(user_id)
    return render_template("rooms/index.html", rooms=rooms)


@rooms_bp.get("/rooms/create")
@login_required
def rooms_create_page():
    return render_template("rooms/create.html")


@rooms_bp.post("/rooms/create")
@login_required
def rooms_create_post():
    name = (request.form.get("name") or "").strip()
    with_code = (request.form.get("with_code") == "on")

    if len(name) < 3:
        flash("Room name must be at least 3 characters.", "error")
        return redirect(url_for("rooms.rooms_create_page"))

    room = create_room(owner_id=session["user_id"], name=name, with_code=with_code)

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

    if not code:
        flash("Please enter a room code.", "error")
        return redirect(url_for("rooms.rooms_join_page"))

    room = find_room_by_code(code)
    if not room:
        flash("Invalid room code.", "error")
        return redirect(url_for("rooms.rooms_join_page"))

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

    if not s:
        return jsonify({
            "status": "idle",
            "duration_seconds": 25 * 60,
            "remaining_seconds": 25 * 60,
        })

    return jsonify({
        "status": s.status,
        "duration_seconds": s.duration_seconds,
        "remaining_seconds": s.remaining_seconds(),
        "started_by": s.started_by,
    })


@rooms_bp.post("/rooms/<int:room_id>/session/start")
@login_required
def room_session_start(room_id: int):
    room = get_room(room_id)
    if not room:
        flash("Room not found.", "error")
        return redirect(url_for("rooms.rooms_index"))

    is_member = RoomMember.query.filter_by(room_id=room_id, user_id=session["user_id"]).first()
    if not is_member:
        flash("You are not a member of this room.", "error")
        return redirect(url_for("rooms.rooms_index"))

    try:
        minutes = int(request.form.get("minutes") or "25")
    except ValueError:
        minutes = 25

    minutes = max(1, min(minutes, 180))
    start_session(room_id=room_id, user_id=session["user_id"], duration_seconds=minutes * 60)

    flash("Focus session started ✅", "success")
    return redirect(url_for("rooms.room_detail", room_id=room_id))


@rooms_bp.post("/rooms/<int:room_id>/session/pause")
@login_required
def room_session_pause(room_id: int):
    s = get_active_session(room_id)
    if s:
        pause_session(s)
        flash("Paused ⏸️", "info")
    return redirect(url_for("rooms.room_detail", room_id=room_id))


@rooms_bp.post("/rooms/<int:room_id>/session/resume")
@login_required
def room_session_resume(room_id: int):
    s = get_active_session(room_id)
    if s:
        resume_session(s)
        flash("Resumed ▶️", "success")
    return redirect(url_for("rooms.room_detail", room_id=room_id))


@rooms_bp.post("/rooms/<int:room_id>/session/reset")
@login_required
def room_session_reset(room_id: int):
    s = get_active_session(room_id) or get_latest_session(room_id)
    if s:
        reset_session(s, session["user_id"])
        flash("Reset 🔄", "info")
    return redirect(url_for("rooms.room_detail", room_id=room_id))


@rooms_bp.post("/rooms/<int:room_id>/session/end")
@login_required
def room_session_end(room_id: int):
    s = get_active_session(room_id) or get_latest_session(room_id)
    if s:
        end_session(s, session["user_id"])
        flash("Session ended ✅", "success")
    return redirect(url_for("rooms.room_detail", room_id=room_id))


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

    members = [{"user_id": uid, "username": uname} for (uname, uid) in rows]
    return jsonify({"count": len(members), "members": members})