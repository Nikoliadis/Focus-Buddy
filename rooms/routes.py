from __future__ import annotations

from flask import render_template, request, redirect, url_for, flash, session

from main.auth_utils import login_required
from main.db import db

from . import rooms_bp
from .models import RoomMember
from .service import create_room, get_user_rooms, get_room, add_member, find_room_by_code


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
    return render_template("rooms/join.html")


@rooms_bp.post("/rooms/join")
@login_required
def rooms_join_post():
    code = (request.form.get("code") or "").strip().upper()
    room = find_room_by_code(code)
    if not room:
        flash("Invalid room code.", "error")
        return redirect(url_for("rooms.rooms_join_page"))

    add_member(room, session["user_id"])
    flash("Joined room ✅", "success")
    return redirect(url_for("rooms.room_detail", room_id=room.id))


@rooms_bp.get("/rooms/<int:room_id>")
@login_required
def room_detail(room_id: int):
    room = get_room(room_id)
    if not room:
        flash("Room not found.", "error")
        return redirect(url_for("rooms.rooms_index"))

    # authorization: must be a member
    is_member = RoomMember.query.filter_by(room_id=room.id, user_id=session["user_id"]).first()
    if not is_member:
        flash("You are not a member of this room.", "error")
        return redirect(url_for("rooms.rooms_index"))

    members = (
        db.session.query(RoomMember.user_id)
        .filter(RoomMember.room_id == room.id)
        .order_by(RoomMember.joined_at.asc())
        .all()
    )
    member_count = len(members)

    return render_template("rooms/room.html", room=room, member_count=member_count)