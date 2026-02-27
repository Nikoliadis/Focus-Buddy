from __future__ import annotations

import secrets
from typing import Optional

from sqlalchemy.exc import IntegrityError

from main.db import db
from .models import Room, RoomMember


def _make_code(length: int = 8) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def create_room(owner_id: int, name: str, with_code: bool = True) -> Room:
    room = Room(name=name.strip(), owner_id=owner_id)

    if with_code:
        # ensure unique join_code
        for _ in range(5):
            code = _make_code()
            if not Room.query.filter_by(join_code=code).first():
                room.join_code = code
                break

    db.session.add(room)
    db.session.commit()

    # owner becomes member
    db.session.add(RoomMember(room_id=room.id, user_id=owner_id))
    db.session.commit()
    return room


def add_member(room: Room, user_id: int) -> None:
    try:
        db.session.add(RoomMember(room_id=room.id, user_id=user_id))
        db.session.commit()
    except IntegrityError:
        db.session.rollback()  # already a member


def get_room(room_id: int) -> Optional[Room]:
    return Room.query.get(room_id)


def get_user_rooms(user_id: int):
    return (
        db.session.query(Room)
        .join(RoomMember, RoomMember.room_id == Room.id)
        .filter(RoomMember.user_id == user_id)
        .order_by(Room.created_at.desc())
        .all()
    )


def find_room_by_code(code: str) -> Optional[Room]:
    c = (code or "").strip().upper()
    if not c:
        return None
    return Room.query.filter_by(join_code=c).first()