"""pgvector store against a real PostgreSQL with the vector extension (skipped elsewhere)."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
from sqlalchemy import text

from main.db import db
from models.chat import ChatMessage
from rooms.service import create_room

pytestmark = pytest.mark.skipif(
    not os.environ["DATABASE_URL"].startswith("postgresql"), reason="pgvector tests need PostgreSQL"
)

store = pytest.importorskip("ai.retrieval.store")
DIM = 32


class HashEmbedder:
    """Deterministic offline embedder: hashed bag of words, unit length."""

    dimension = DIM

    def __init__(self, name: str = "hash-v1"):
        self.name = name
        self.encoded = 0

    def encode(self, texts):
        self.encoded += len(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


@pytest.fixture(scope="module")
def vector_db(_app):
    with _app.app_context():
        try:
            with db.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as exc:  # noqa: BLE001 - any failure means "no pgvector here"
            pytest.skip(f"pgvector extension not available: {exc}")
        yield db.engine
        store.drop_schema(db.engine)  # before the app fixture drops chat_messages


@pytest.fixture()
def rooms(app, vector_db, make_user):
    alice, bob = make_user("alice"), make_user("bob")
    return create_room(alice.id, "Databases"), create_room(bob.id, "Algorithms")


class RandomEmbedder:
    """Dense pseudo-random unit vector per distinct text: no ties, like real embeddings."""

    dimension = DIM

    def __init__(self, name: str = "random-v1"):
        self.name = name

    def encode(self, texts):
        out = np.stack([np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
                        .normal(size=DIM) for t in texts]).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def bulk_messages(room, n, prefix):
    """Fast core INSERT for thousands of chat rows; returns their (id, room_id, body) frame."""
    import pandas as pd
    from sqlalchemy import insert, select

    db.session.execute(insert(ChatMessage), [
        {"room_id": room.id, "user_id": room.owner_id, "username": "u", "body": f"{prefix} {i}"} for i in range(n)
    ])
    db.session.commit()
    rows = db.session.execute(select(ChatMessage.id, ChatMessage.room_id, ChatMessage.body)
                              .where(ChatMessage.room_id == room.id)).all()
    return pd.DataFrame(rows, columns=["message_id", "room_id", "body"])


def add_messages(room, bodies, user_id=None):
    msgs = [ChatMessage(room_id=room.id, user_id=user_id or room.owner_id, username="u", body=b) for b in bodies]
    db.session.add_all(msgs)
    db.session.commit()
    return msgs


def frame(msgs):
    import pandas as pd

    return pd.DataFrame({"message_id": [m.id for m in msgs], "room_id": [m.room_id for m in msgs],
                         "body": [m.body for m in msgs]})


class TestSchema:
    def test_idempotent_with_hnsw_index(self, vector_db):
        store.create_schema(vector_db, DIM)
        store.create_schema(vector_db, DIM)

        with vector_db.connect() as conn:
            indexes = conn.execute(text(
                "SELECT indexdef FROM pg_indexes WHERE tablename = 'message_embeddings'")).scalars().all()
        assert any("hnsw" in i and "vector_cosine_ops" in i for i in indexes)

    def test_dimension_mismatch_is_rejected(self, vector_db):
        store.create_schema(vector_db, DIM)
        with pytest.raises(ValueError, match="dimension|stores"):
            store.create_schema(vector_db, DIM * 2)


class TestIndexing:
    def test_incremental_upsert(self, rooms, vector_db):
        databases, _ = rooms
        msgs = add_messages(databases, ["sql join help", "normalization notes", "coffee break"])
        embedder = HashEmbedder()

        first = store.index_messages(vector_db, frame(msgs), embedder)
        second = store.index_messages(vector_db, frame(msgs), embedder)

        assert (first.embedded, first.unchanged) == (3, 0)
        assert (second.embedded, second.unchanged) == (0, 3)   # nothing re-embedded
        assert embedder.encoded == 3

    def test_edited_message_is_re_embedded(self, rooms, vector_db):
        databases, _ = rooms
        msgs = add_messages(databases, ["sql join help", "coffee break"])
        embedder = HashEmbedder()
        store.index_messages(vector_db, frame(msgs), embedder)

        msgs[0].body = "sql index help"
        db.session.commit()
        result = store.index_messages(vector_db, frame(msgs), embedder)

        assert (result.embedded, result.unchanged) == (1, 1)

    def test_deleting_a_message_deletes_its_vector(self, rooms, vector_db):
        databases, _ = rooms
        msgs = add_messages(databases, ["to be deleted"])
        store.index_messages(vector_db, frame(msgs), HashEmbedder())

        db.session.delete(msgs[0])
        db.session.commit()

        assert store.stats(vector_db)["vectors"].sum() == 0  # ON DELETE CASCADE

    def test_models_are_stored_side_by_side(self, rooms, vector_db):
        databases, _ = rooms
        msgs = add_messages(databases, ["sql join help"])
        store.index_messages(vector_db, frame(msgs), HashEmbedder("hash-v1"))
        store.index_messages(vector_db, frame(msgs), HashEmbedder("hash-v2"))

        assert store.stats(vector_db).set_index("model")["vectors"].to_dict() == {"hash-v1": 1, "hash-v2": 1}

    def test_only_public_rooms_are_indexed_from_the_database(self, app, vector_db, make_user):
        owner = make_user("carol")
        public = create_room(owner.id, "Open")
        private = create_room(owner.id, "Secret", password="pw")
        add_messages(public, ["public sql question"])
        add_messages(private, ["private diary entry"])

        result = store.index_public_chat(vector_db, HashEmbedder())

        assert result.seen == 1


class TestSearch:
    def test_nearest_first_and_similarity_is_cosine(self, rooms, vector_db):
        databases, _ = rooms
        msgs = add_messages(databases, ["sql join help", "coffee break now", "join two sql tables"])
        embedder = HashEmbedder()
        store.index_messages(vector_db, frame(msgs), embedder)

        hits = store.search_text(vector_db, embedder, "sql join", [databases.id], k=3)

        assert hits["body"].iloc[0] == "sql join help"
        assert hits["distance"].is_monotonic_increasing
        q, v = embedder.encode(["sql join"])[0], embedder.encode(["sql join help"])[0]
        assert hits["similarity"].iloc[0] == pytest.approx(float(q @ v), abs=1e-5)

    def test_results_never_leave_the_allowed_rooms(self, rooms, vector_db):
        databases, algorithms = rooms
        add = add_messages(databases, ["sql join help"])
        other = add_messages(algorithms, ["sql join help exactly"])  # closer, but not allowed
        embedder = HashEmbedder()
        store.index_messages(vector_db, frame(add + other), embedder)

        hits = store.search_text(vector_db, embedder, "sql join help exactly", [databases.id], k=5)

        assert set(hits["room_id"]) == {databases.id}
        assert store.search_text(vector_db, embedder, "sql", [], k=5).empty  # no rooms -> nothing

    def test_filtered_hnsw_needs_the_iterative_scan(self, rooms, vector_db):
        """Allowed room holds ~5% of 20,000 vectors. With the HNSW plan and a small ef_search,
        post-filtering silently returns fewer than k rows; the iterative scan returns k."""
        import pandas as pd

        databases, algorithms = rooms
        embedder = RandomEmbedder()
        data = pd.concat([bulk_messages(algorithms, 19000, "crowd"), bulk_messages(databases, 1000, "allowed")])
        store.index_messages(vector_db, data, embedder)
        q = embedder.encode(["a query"])[0]

        plan = store.explain(vector_db, q, [databases.id], embedder.name, k=10)
        assert "hnsw" in plan, plan
        without = store.search(vector_db, q, [databases.id], embedder.name, k=10, ef_search=10, iterative=False)
        with_iterative = store.search(vector_db, q, [databases.id], embedder.name, k=10, ef_search=10)

        assert len(without) < 10                       # silently incomplete
        assert len(with_iterative) == 10
        assert set(with_iterative["room_id"]) == {databases.id}

    def test_recall_grows_with_ef_search(self, rooms, vector_db):
        """ANN recall@10 = share of the exact top 10 that HNSW also returns. A larger candidate
        list (ef_search) costs time and buys recall."""
        _, algorithms = rooms
        embedder = RandomEmbedder()
        store.index_messages(vector_db, bulk_messages(algorithms, 5000, "msg"), embedder)
        queries = embedder.encode([f"query {i}" for i in range(20)])

        def recall(ef):
            scores = []
            for q in queries:
                approx = set(store.search(vector_db, q, [algorithms.id], embedder.name, 10, ef_search=ef)["message_id"])
                exact = set(store.search(vector_db, q, [algorithms.id], embedder.name, 10, exact=True)["message_id"])
                scores.append(len(approx & exact) / 10)
            return float(np.mean(scores))

        low, high = recall(10), recall(200)
        assert high >= 0.95
        assert high >= low


def test_sqlite_is_refused(tmp_path):
    from sqlalchemy import create_engine

    with pytest.raises(RuntimeError, match="PostgreSQL"):
        store.create_schema(create_engine(f"sqlite:///{tmp_path}/x.db"))
