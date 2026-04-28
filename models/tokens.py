from __future__ import annotations

import secrets
from datetime import datetime

from main.db import db


class EmailVerificationToken(db.Model):
    __tablename__ = "email_verification_tokens"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token      = db.Column(db.String(86), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    used       = db.Column(db.Boolean, nullable=False, default=False)

    @staticmethod
    def generate(user_id: int) -> "EmailVerificationToken":
        tok = EmailVerificationToken(user_id=user_id, token=secrets.token_urlsafe(64))
        db.session.add(tok)
        return tok

    def is_valid(self) -> bool:
        if self.used:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() < 86400  # 24h


class PasswordResetToken(db.Model):
    __tablename__ = "password_reset_tokens"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token      = db.Column(db.String(86), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    used       = db.Column(db.Boolean, nullable=False, default=False)

    @staticmethod
    def generate(user_id: int) -> "PasswordResetToken":
        tok = PasswordResetToken(user_id=user_id, token=secrets.token_urlsafe(64))
        db.session.add(tok)
        return tok

    def is_valid(self) -> bool:
        if self.used:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() < 3600  # 1h
