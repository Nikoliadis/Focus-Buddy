"""SYNTHETIC chat messages for developing the NLP pipeline.

Real chat is scarce (and private), so this generates messages from templates. Each room
has a known topic; a good topic-term analysis should recover it. Nothing here is real
user text, and nothing derived from it describes real users.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import cleaning

TOPICS = {
    "databases": ["sql join", "normalization", "BCNF", "κανονικοποίηση", "foreign key", "index", "transactions",
                  "ερωτήματα sql", "query optimizer", "schema"],
    "algorithms": ["dijkstra", "graph", "complexity", "πολυπλοκότητα", "dynamic programming", "recursion",
                   "γράφοι", "big o", "sorting", "heap"],
    "thesis": ["chapter", "supervisor", "draft", "βιβλιογραφία", "κεφάλαιο", "references", "abstract",
               "επιβλέπων", "deadline", "methodology"],
}
TOPIC_TEMPLATES = [
    "Can someone explain {t}?", "I'm stuck on {t} again", "Does anyone have notes on {t}?",
    "Έχει κανείς σημειώσεις για {t};", "Δεν καταλαβαίνω το {t}", "finally understood {t}!",
    "{t} is harder than I thought", "Let's review {t} after the break", "Το {t} θα πέσει στις εξετάσεις;",
]
CHITCHAT = [
    "Starting the {n} min block now", "Break time ☕", "Back!", "Καλή επιτυχία σε όλους!",
    "Πάμε δυνατά 💪", "phones away, let's focus", "coffee break in {n}", "See you tomorrow",
    "Καλημέρα!", "good luck with the exam", "brb {n} minutes",
]
NOISE = ["ping me at {user}@example.com", "check https://example.org/notes", "call me 210 1234567", "@{user} ;)"]


def generate_chat(
    n_rooms: int = 6,
    users_per_room: int = 8,
    messages_per_room: int = 150,
    topic_share: float = 0.6,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Return cleaned chat messages (shaped like analytics.cleaning output) and room -> topic."""
    rng = np.random.default_rng(seed)
    topics = list(TOPICS)
    room_topic = {room: topics[(room - 1) % len(topics)] for room in range(1, n_rooms + 1)}
    start = pd.Timestamp("2026-02-01", tz="UTC")
    rows, message_id = [], 0
    for room, topic in room_topic.items():
        users = np.arange(users_per_room) + room * 100
        for _ in range(messages_per_room):
            message_id += 1
            r = rng.random()
            if r < topic_share:
                term = TOPICS[topic][rng.integers(len(TOPICS[topic]))]
                body = TOPIC_TEMPLATES[rng.integers(len(TOPIC_TEMPLATES))].format(t=term)
            elif r < 0.97:
                body = CHITCHAT[rng.integers(len(CHITCHAT))].format(n=int(rng.choice([5, 10, 25, 50])))
            else:
                body = NOISE[rng.integers(len(NOISE))].format(user=f"student{rng.integers(1000)}")
            rows.append({
                "message_id": message_id, "room_id": room, "user_id": int(rng.choice(users)), "body": body,
                "created_at": start + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 60))),
                "room_is_private": False,
            })
    messages, _ = cleaning.clean_chat_messages(pd.DataFrame(rows), now=start + pd.Timedelta(days=90))
    return messages, room_topic
