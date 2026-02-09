from __future__ import annotations

from typing import Optional
from werkzeug.security import generate_password_hash, check_password_hash

from main.db import db
from models.user import User


def find_user_by_username(username: str) -> Optional[User]:
    u = (username or "").strip().lower()
    if not u:
        return None
    return User.query.filter_by(username=u).first()


def create_user(username: str, password: str) -> User:
    u = username.strip().lower()
    user = User(username=u, password_hash=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()
    return user


def verify_password(user: User, password: str) -> bool:
    return check_password_hash(user.password_hash, password)