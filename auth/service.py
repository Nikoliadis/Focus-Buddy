from __future__ import annotations

from typing import Optional
from werkzeug.security import generate_password_hash, check_password_hash

from sqlalchemy import or_

from main.db import db
from models.user import User


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def find_user_by_login(login: str) -> Optional[User]:
    """Login can be username OR email."""
    value = _norm(login)
    if not value:
        return None

    return User.query.filter(
        or_(User.username == value, User.email == value)
    ).first()


def user_exists(username: str, email: str) -> bool:
    u = _norm(username)
    e = _norm(email)
    return User.query.filter(or_(User.username == u, User.email == e)).first() is not None


def create_user(username: str, email: str, password: str) -> User:
    u = _norm(username)
    e = _norm(email)

    user = User(
        username=u,
        email=e,
        password_hash=generate_password_hash(password),
    )
    db.session.add(user)
    db.session.commit()
    return user


def verify_password(user: User, password: str) -> bool:
    return check_password_hash(user.password_hash, password)
