"""Fill the database with demo users, a study room, chat history and two weeks of focus data.

Usage:
    python -m scripts.seed_demo

Log in afterwards as  demo / demo1234
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from werkzeug.security import generate_password_hash

from app import app
from main.db import db
from models.chat import ChatMessage
from models.focus import FocusLog, FocusSession
from models.todo import TodoItem
from models.user import User
from rooms.models import RoomMember
from rooms.service import create_room

DEMO_PASSWORD = "demo1234"
USERNAMES = ["demo", "maria", "giorgos", "eleni", "nikos"]

CHAT = [
    ("maria", "Starting the 25 min block now, phones away 📵"),
    ("giorgos", "Same, working through the DB normalization exercises"),
    ("eleni", "Does anyone have notes on BCNF?"),
    ("demo", "Ask FocusBot during the break, it explained it really well"),
    ("nikos", "Joined! Let's go 💪"),
]

TODOS = [
    ("Finish chapter 4: Transactions", True),
    ("Practice SQL joins (exercises 1-10)", False),
    ("Review lecture slides on indexing", False),
    ("Email supervisor about thesis draft", True),
]


def _user(username: str) -> User:
    user = User.query.filter_by(username=username).first()
    if user:
        return user
    user = User(
        username=username,
        email=f"{username}@example.com",
        password_hash=generate_password_hash(DEMO_PASSWORD),
        email_verified=True,
        daily_goal_minutes=120 if username == "demo" else None,
    )
    db.session.add(user)
    db.session.commit()
    return user


def _log_focus(user: User, room_id: int, seconds: int, when: datetime) -> None:
    session = FocusSession(
        room_id=room_id,
        started_by=user.id,
        status="ended",
        duration_seconds=seconds,
        started_at=when - timedelta(seconds=seconds),
        ended_at=when,
        created_at=when - timedelta(seconds=seconds),
    )
    db.session.add(session)
    db.session.flush()
    db.session.add(FocusLog(
        user_id=user.id, room_id=room_id, session_id=session.id, focused_seconds=seconds, created_at=when
    ))


def seed() -> None:
    rng = random.Random(42)
    users = {name: _user(name) for name in USERNAMES}

    if FocusLog.query.count():
        print("Demo data already present, nothing to do.")
        return

    room = create_room(users["demo"].id, "Thesis Sprint")
    other_rooms = [
        create_room(users["maria"].id, "Databases Exam Prep"),
        create_room(users["giorgos"].id, "Late Night Coding"),
    ]
    # Rooms must predate the focus history generated below.
    for r in (room, *other_rooms):
        r.created_at = datetime.utcnow() - timedelta(days=15)
    for name in USERNAMES[1:]:
        db.session.add(RoomMember(room_id=room.id, user_id=users[name].id))

    now = datetime.utcnow()
    for name, user in users.items():
        active_days = 14 if name == "demo" else rng.randint(6, 12)
        for days_ago in range(active_days):
            for _ in range(rng.randint(1, 4)):
                minutes = rng.choice([25, 25, 25, 50])
                when = now - timedelta(days=days_ago, hours=rng.randint(0, 6))
                _log_focus(user, room.id, minutes * 60, when)

    for i, (name, body) in enumerate(CHAT):
        db.session.add(ChatMessage(
            room_id=room.id,
            user_id=users[name].id,
            username=name,
            body=body,
            created_at=now - timedelta(minutes=len(CHAT) - i),
        ))

    for body, done in TODOS:
        db.session.add(TodoItem(user_id=users["demo"].id, body=body, done=done, is_public=True))

    db.session.commit()
    print(f"Seeded {len(users)} users, 3 rooms, {FocusLog.query.count()} focus logs.")
    print(f"Log in as: demo / {DEMO_PASSWORD}")


if __name__ == "__main__":
    with app.app_context():
        seed()
