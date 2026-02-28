from __future__ import annotations

from datetime import datetime
from main.db import db


class FocusSession(db.Model):
    __tablename__ = "focus_sessions"

    id = db.Column(db.Integer, primary_key=True)
    room_id = db.Column(db.Integer, nullable=False, index=True)
    started_by = db.Column(db.Integer, nullable=False)

    # idle | running | paused | ended
    status = db.Column(db.String(20), nullable=False, default="idle")

    duration_seconds = db.Column(db.Integer, nullable=False)

    started_at = db.Column(db.DateTime, nullable=True)
    ended_at = db.Column(db.DateTime, nullable=True)
    
    # pause tracking
    paused_at = db.Column(db.DateTime, nullable=True)
    paused_seconds = db.Column(db.Integer, nullable=False, default=0)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def remaining_seconds(self) -> int:
        if self.status == "ended":
            return 0

        if self.status == "idle" or not self.started_at:
            return self.duration_seconds

        now = datetime.utcnow()

        # paused
        if self.status == "paused" and self.paused_at:
            elapsed = (self.paused_at - self.started_at).total_seconds()
        else:
            elapsed = (now - self.started_at).total_seconds()

        elapsed -= self.paused_seconds

        remaining = int(self.duration_seconds - elapsed)

        return max(0, remaining)

class FocusLog(db.Model):
    __tablename__ = "focus_logs"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, nullable=False, index=True)
    room_id = db.Column(db.Integer, db.ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = db.Column(db.Integer, db.ForeignKey("focus_sessions.id", ondelete="CASCADE"), nullable=False, index=True)

    focused_seconds = db.Column(db.Integer, nullable=False, default=0)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)