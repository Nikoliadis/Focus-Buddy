from __future__ import annotations

from flask import render_template, request, redirect, url_for, flash, session

from main.auth_utils import login_required
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

    # verify membership
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

    # must be a member
    is_member = RoomMember.query.filter_by(room_id=room.id, user_id=user_id).first()
    if not is_member:
        flash("You are not a member of this room.", "error")
        return redirect(url_for("rooms.rooms_index"))

    members = get_room_members(room)
    is_owner = (user_id == room.owner_id)

    # invite link (prefill join page)
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

    # owner cannot leave
    if user_id == room.owner_id:
        flash("Owner cannot leave the room. You can delete the room instead.", "error")
        return redirect(url_for("rooms.room_detail", room_id=room.id))

    # must be member
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