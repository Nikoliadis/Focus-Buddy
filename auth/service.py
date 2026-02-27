from __future__ import annotations

from werkzeug.security import generate_password_hash, check_password_hash

from main.db import db
from models.user import User


def find_user_by_username(username: str):
    return User.query.filter_by(username=username).first()


def find_user_by_email(email: str):
    return User.query.filter_by(email=email).first()


def find_user_by_login_identifier(identifier: str):
    """Identifier can be username OR email."""
    ident = (identifier or "").strip()
    if not ident:
        return None

    # if looks like email, try email first
    if "@" in ident:
        user = User.query.filter(db.func.lower(User.email) == ident.lower()).first()
        if user:
            return user

    # fallback username
    return User.query.filter(db.func.lower(User.username) == ident.lower()).first()


def create_user(username: str, email: str, password: str) -> User:
    user = User(
        username=username.strip(),
        email=email.strip().lower(),
        password_hash=generate_password_hash(password),
    )
    db.session.add(user)
    db.session.commit()
    return user


def verify_password(user: User, password: str) -> bool:
    return check_password_hash(user.password_hash, password)