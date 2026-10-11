from __future__ import annotations

import os

# Must be set before the app module is imported: Config reads env at import time.
# CI points TEST_DATABASE_URL at a real PostgreSQL service; locally we fall back to SQLite.
os.environ["DATABASE_URL"] = os.getenv("TEST_DATABASE_URL", "sqlite://")
os.environ["SECRET_KEY"] = "test-secret"
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("RESEND_API_KEY", None)

import pytest
from werkzeug.security import generate_password_hash

from app import create_app
from main.db import db
from main.socketio_ext import socketio
from models.user import User
from rooms.presence_store import PRESENCE, LAST_SEEN, ROOM_CYCLES

CSRF_TOKEN = "test-csrf-token"


@pytest.fixture(scope="session")
def _app():
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": os.environ["DATABASE_URL"],
        "RATELIMIT_ENABLED": False,
    })
    with app.app_context():
        db.drop_all()
        db.create_all()
    yield app
    with app.app_context():
        db.drop_all()


@pytest.fixture()
def app(_app):
    """Schema is created once per run; each test starts from empty tables (much faster on PostgreSQL)."""
    with _app.app_context():
        yield _app
        db.session.remove()
        for table in reversed(db.metadata.sorted_tables):
            db.session.execute(table.delete())
        db.session.commit()

    PRESENCE.clear()
    LAST_SEEN.clear()
    ROOM_CYCLES.clear()


@pytest.fixture()
def client(app):
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["csrf_token"] = CSRF_TOKEN
    return client


@pytest.fixture()
def make_user(app):
    def _make_user(username: str = "alice", password: str = "password123", verified: bool = True) -> User:
        user = User(
            username=username,
            email=f"{username}@example.com",
            # Cheap hash for speed; the app itself uses Werkzeug's default (scrypt).
            password_hash=generate_password_hash(password, method="pbkdf2:sha256:1000"),
            email_verified=verified,
        )
        db.session.add(user)
        db.session.commit()
        return user

    return _make_user


@pytest.fixture()
def login(client):
    """Put a user in the session without going through the login form."""
    def _login(user: User, c=None):
        c = c or client
        with c.session_transaction() as sess:
            sess["user_id"] = user.id
            sess["username"] = user.username
        return c

    return _login


@pytest.fixture()
def post_form(client):
    """POST a form with a valid CSRF token attached."""
    def _post_form(url: str, data: dict | None = None, c=None, **kwargs):
        c = c or client
        payload = {"csrf_token": CSRF_TOKEN, **(data or {})}
        return c.post(url, data=payload, **kwargs)

    return _post_form


@pytest.fixture()
def second_client(app):
    c = app.test_client()
    with c.session_transaction() as sess:
        sess["csrf_token"] = CSRF_TOKEN
    return c


@pytest.fixture()
def socket_client(app):
    """Factory for Socket.IO test clients that share a Flask test client's session."""
    created = []

    def _socket_client(flask_client):
        sc = socketio.test_client(app, flask_test_client=flask_client)
        created.append(sc)
        return sc

    yield _socket_client

    for sc in created:
        if sc.is_connected():
            sc.disconnect()
