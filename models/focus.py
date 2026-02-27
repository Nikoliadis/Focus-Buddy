from __future__ import annotations

from datetime import datetime
from main.db import db


class FocusSession(db.Model):
    __tablename__ = "focus_sessions"

    id = db.Column(db.Integer, primary_key=True)

    room_id = db.Column(db.Integer, db.ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    started_by = db.Column(db.Integer, nullable=False, index=True)

    # idle | running | paused | ended
    status = db.Column(db.String(16), nullable=False, default="idle", index=True)

    duration_seconds = db.Column(db.Integer, nullable=False, default=25 * 60)

    started_at = db.Column(db.DateTime, nullable=True)
    ended_at = db.Column(db.DateTime, nullable=True)

    # pause tracking
    paused_at = db.Column(db.DateTime, nullable=True)
    paused_seconds = db.Column(db.Integer, nullable=False, default=0)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        db.Index("ix_focus_sessions_room_status", "room_id", "status"),
    )

    def remaining_seconds(self) -> int:
        """Server-side remaining calc for polling UI (no websockets yet)."""
        if self.status == "idle":
            return self.duration_seconds

        if not self.started_at:
            return self.duration_seconds

        now = datetime.utcnow()

        effective_now = now
        extra_pause = 0

        if self.status == "paused" and self.paused_at:
            # time is "frozen" while paused
            effective_now = self.paused_at

        elapsed = int((effective_now - self.started_at).total_seconds())
        elapsed -= int(self.paused_seconds)

        remaining = self.duration_seconds - max(0, elapsed)
        return max(0, remaining)


class FocusLog(db.Model):
    __tablename__ = "focus_logs"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, nullable=False, index=True)
    room_id = db.Column(db.Integer, db.ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = db.Column(db.Integer, db.ForeignKey("focus_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    focused_seconds = db.Column(db.Integer, nullable=False, default=0)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)