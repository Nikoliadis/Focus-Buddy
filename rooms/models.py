from datetime import datetime
from main.db import db

class Room(db.Model):
    __tablename__ = "rooms"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    owner_id = db.Column(db.Integer, nullable=False, index=True)

    join_code = db.Column(db.String(12), nullable=True, index=True, unique=True)
    password_hash = db.Column(db.String(256), nullable=True)
    room_password = db.Column(db.String(256), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    @property
    def is_private(self) -> bool:
        return self.room_password is not None

class RoomMember(db.Model):
    __tablename__ = "room_members"

    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.Integer, db.ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    joined_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("room_id", "user_id", name="uq_room_member"),
    )


class RoomMute(db.Model):
    __tablename__ = "room_mutes"

    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.Integer, db.ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = db.Column(db.Integer, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("room_id", "user_id", name="uq_room_mute"),
    )