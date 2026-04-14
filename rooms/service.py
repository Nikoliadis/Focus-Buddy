from __future__ import annotations

import secrets
from typing import Optional, List, Tuple

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash

from main.db import db
from models.user import User
from .models import Room, RoomMember


def _make_code(length: int = 8) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def create_room(owner_id: int, name: str, with_code: bool = True, password: str | None = None) -> Room:
    room = Room(name=name.strip(), owner_id=owner_id)
    if password:
        room.room_password = password
        room.password_hash = generate_password_hash(password)

    if with_code:
        for _ in range(5):
            code = _make_code()
            if not Room.query.filter_by(join_code=code).first():
                room.join_code = code
                break

    db.session.add(room)
    db.session.commit()

    # owner auto member
    db.session.add(RoomMember(room_id=room.id, user_id=owner_id))
    db.session.commit()

    return room


def add_member(room: Room, user_id: int) -> None:
    try:
        db.session.add(RoomMember(room_id=room.id, user_id=user_id))
        db.session.commit()
    except IntegrityError:
        db.session.rollback()  # already member


def remove_member(room_id: int, user_id: int) -> None:
    RoomMember.query.filter_by(room_id=room_id, user_id=user_id).delete()
    db.session.commit()


def delete_room(room_id: int) -> None:
    # Cascade deletes members (FK ondelete="CASCADE") if configured.
    Room.query.filter_by(id=room_id).delete()
    db.session.commit()


def get_room(room_id: int) -> Optional[Room]:
    return Room.query.get(room_id)


def get_user_rooms(user_id: int) -> list[dict]:
    rows = (
        db.session.query(Room, User.username, func.count(RoomMember.id))
        .join(User, User.id == Room.owner_id)
        .join(RoomMember, RoomMember.room_id == Room.id)
        .filter(RoomMember.user_id == user_id)
        .group_by(Room.id, User.username)
        .order_by(Room.created_at.desc())
        .all()
    )
    return [{"room": r, "owner": uname, "member_count": cnt} for r, uname, cnt in rows]


def get_public_rooms() -> list[dict]:
    """All public (no password) rooms with owner name and member count."""
    rows = (
        db.session.query(Room, User.username, func.count(RoomMember.id))
        .join(User, User.id == Room.owner_id)
        .outerjoin(RoomMember, RoomMember.room_id == Room.id)
        .filter(Room.password_hash.is_(None))
        .group_by(Room.id, User.username)
        .order_by(Room.created_at.desc())
        .all()
    )
    return [{"room": r, "owner": uname, "member_count": cnt} for r, uname, cnt in rows]


def find_room_by_code(code: str) -> Optional[Room]:
    c = (code or "").strip().upper()
    if not c:
        return None
    return Room.query.filter_by(join_code=c).first()


def verify_room_password(room: Room, password: str) -> bool:
    """Return True if room is public OR the given password matches."""
    if not room.is_private:
        return True
    return room.room_password == (password or "")


def get_room_members(room: Room) -> List[Tuple[User, bool]]:
    """
    Returns list of (User, is_owner) for a room.
    """
    rows = (
        db.session.query(User, RoomMember)
        .join(RoomMember, RoomMember.user_id == User.id)
        .filter(RoomMember.room_id == room.id)
        .order_by(RoomMember.joined_at.asc())
        .all()
    )

    members: List[Tuple[User, bool]] = []
    for user, _membership in rows:
        members.append((user, user.id == room.owner_id))
    return members