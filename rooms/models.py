from datetime import datetime
from main.db import db

class Room(db.Model):
    __tablename__ = "rooms"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    owner_id = db.Column(db.Integer, nullable=False, index=True)

    join_code = db.Column(db.String(12), nullable=True, index=True, unique=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class RoomMember(db.Model):
    __tablename__ = "room_members"

    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.Integer, db.ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    joined_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("room_id", "user_id", name="uq_room_member"),
    )