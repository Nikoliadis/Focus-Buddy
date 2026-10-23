"""Vector storage and similarity search with pgvector, inside the existing PostgreSQL.

    chat_messages (existing) ──► embed (nlp.embeddings) ──► message_embeddings (new)
                                                              message_id → chat_messages.id ON DELETE CASCADE
                                                              model, room_id, content_hash
                                                              embedding vector(384) + HNSW index
    query ──► embed ──► ORDER BY embedding <=> :q LIMIT k   (only rooms the caller may read)

Why the same PostgreSQL and not a separate vector database: the vectors live next to the
rows they describe, so one transaction, one backup, one access-control story; deleting a
message deletes its vector (foreign key cascade); and filters such as "only my rooms" are
plain SQL. A dedicated vector DB pays off at hundreds of millions of vectors or when you
need features pgvector lacks; FocusBuddy is nowhere near that.

This table is deliberately NOT part of the Flask models' metadata: ``db.create_all()`` at
app start would then require the pgvector extension, and SQLite (tests, local runs) or a
PostgreSQL without the extension would fail to boot. The schema is created explicitly by
``create_schema``.

Usage:
    python -m ai.retrieval.store index                 # embed public-room chat in DATABASE_URL
    python -m ai.retrieval.store search "notes on sql joins" --rooms 1,2
    python -m ai.retrieval.store stats
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    bindparam,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from nlp.embeddings import Embedder, embed_unique
from nlp.preprocessing import redact

DEFAULT_DIMENSION = 384
TABLE_NAME = "message_embeddings"
# HNSW graph parameters: m = links per node, ef_construction = candidate list while building.
# Higher = better recall, slower build, more memory. These are pgvector's defaults.
HNSW_M, HNSW_EF_CONSTRUCTION = 16, 64

metadata = MetaData()
# Stub of the existing table: lets SQLAlchemy resolve the foreign key and the join. It is
# never created or altered from here (create_all is called with tables=[...] only).
_chat_messages = Table("chat_messages", metadata, Column("id", Integer, primary_key=True),
                       Column("body", Text), Column("created_at", DateTime))


def _table(dimension: int) -> Table:
    if TABLE_NAME in metadata.tables:
        table = metadata.tables[TABLE_NAME]
        if table.c.embedding.type.dim != dimension:
            raise ValueError(f"{TABLE_NAME} is defined with dimension {table.c.embedding.type.dim}, "
                             f"not {dimension}")
        return table
    table = Table(
        TABLE_NAME, metadata,
        Column("message_id", Integer, ForeignKey("chat_messages.id", ondelete="CASCADE"), primary_key=True),
        # Several models can be stored side by side (needed when switching models: build the
        # new vectors, compare, then switch, without a gap).
        Column("model", String(200), primary_key=True),
        Column("room_id", Integer, nullable=False),
        Column("content_hash", String(64), nullable=False),
        Column("embedding", Vector(dimension), nullable=False),
        Column("embedded_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    )
    Index("ix_message_embeddings_room_model", table.c.room_id, table.c.model)
    Index(
        "ix_message_embeddings_hnsw", table.c.embedding,
        postgresql_using="hnsw",
        postgresql_with={"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    return table


def _stored_dimension(conn: Connection) -> int | None:
    row = conn.execute(text("""
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = to_regclass(:table) AND attname = 'embedding'
    """), {"table": TABLE_NAME}).first()
    return int(row[0]) if row else None


def create_schema(engine: Engine, dimension: int = DEFAULT_DIMENSION) -> Table:
    """Create the extension, table and indexes if missing. Idempotent."""
    if engine.dialect.name != "postgresql":
        raise RuntimeError("pgvector needs PostgreSQL")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        existing = _stored_dimension(conn)
        if existing is not None and existing != dimension:
            raise ValueError(f"{TABLE_NAME} stores {existing}-d vectors; this embedder produces {dimension}-d. "
                             "Use a separate table or drop it before switching dimension.")
        table = _table(dimension)
        metadata.create_all(conn, tables=[table])
    return table


def drop_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))
    if TABLE_NAME in metadata.tables:
        metadata.remove(metadata.tables[TABLE_NAME])


def content_hash(model: str, body: str) -> str:
    """Fingerprint of what was embedded: re-embed only when the (redacted) text or model changes."""
    return hashlib.sha256(f"{model}\x00{redact(body or '')}".encode()).hexdigest()


@dataclass(frozen=True)
class IndexStats:
    seen: int
    embedded: int
    unchanged: int


def index_messages(
    engine: Engine,
    messages: pd.DataFrame,
    embedder: Embedder,
    batch_size: int = 500,
) -> IndexStats:
    """Embed and upsert messages (columns: message_id, room_id, body). Incremental: rows whose
    content hash is already stored for this model are skipped."""
    table = create_schema(engine, embedder.dimension)
    if messages.empty:
        return IndexStats(0, 0, 0)
    frame = messages[["message_id", "room_id", "body"]].copy()
    frame["content_hash"] = [content_hash(embedder.name, b) for b in frame["body"]]

    with engine.connect() as conn:
        stored = dict(conn.execute(
            select(table.c.message_id, table.c.content_hash)
            .where(table.c.model == embedder.name, table.c.message_id.in_(frame["message_id"].tolist()))
        ).all())
    todo = frame[[stored.get(m) != h for m, h in zip(frame["message_id"], frame["content_hash"], strict=True)]]

    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        vectors = embed_unique(embedder, batch["body"].tolist())
        rows = [
            {"message_id": int(m), "model": embedder.name, "room_id": int(r), "content_hash": h, "embedding": v}
            for m, r, h, v in zip(batch["message_id"], batch["room_id"], batch["content_hash"], vectors,
                                  strict=True)
        ]
        stmt = insert(table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[table.c.message_id, table.c.model],
            set_={"room_id": stmt.excluded.room_id, "content_hash": stmt.excluded.content_hash,
                  "embedding": stmt.excluded.embedding, "embedded_at": func.now()},
        )
        with engine.begin() as conn:
            conn.execute(stmt)
    if len(todo):
        with engine.begin() as conn:  # fresh statistics, so the planner picks the right plan
            conn.execute(text(f"ANALYZE {TABLE_NAME}"))
    return IndexStats(seen=len(frame), embedded=len(todo), unchanged=len(frame) - len(todo))


def search(
    engine: Engine,
    query_vector: np.ndarray,
    room_ids: Sequence[int],
    model: str,
    k: int = 5,
    exact: bool = False,
    ef_search: int = 40,
    iterative: bool = True,
) -> pd.DataFrame:
    """Nearest messages by cosine distance, restricted to ``room_ids``.

    ``room_ids`` is required on purpose: the caller must pass the rooms the current user is
    allowed to read. There is no "search everything" default.

    Filtering + approximate index: when the planner uses HNSW, it first collects
    ``ef_search`` candidates and THEN applies the WHERE clause. If only a fraction of the
    candidates is in the allowed rooms, fewer than k rows come back, silently. Measured on
    20,000 vectors with ef_search=10: a filter matching 5% of rows returned 1 row instead of
    10. pgvector 0.8's iterative scan (``iterative=True``) keeps walking the graph until k
    rows pass the filter. For very selective filters the planner skips HNSW altogether and
    sorts the few matching rows exactly.
    ``exact=True`` disables index scans: slow, but the ground truth.
    """
    if not room_ids:
        return pd.DataFrame(columns=["message_id", "room_id", "body", "distance", "similarity"])
    table = _table(len(query_vector))
    distance = table.c.embedding.cosine_distance(np.asarray(query_vector, dtype=np.float32)).label("distance")
    stmt = (
        select(table.c.message_id, table.c.room_id, _chat_messages.c.body, distance)
        .join(_chat_messages, _chat_messages.c.id == table.c.message_id)
        .where(table.c.model == model, table.c.room_id.in_([int(r) for r in room_ids]))
        .order_by(distance)
        .limit(k)
    )
    with engine.begin() as conn:
        if exact:
            conn.execute(text("SET LOCAL enable_indexscan = off"))
        else:
            conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
            conn.execute(text(f"SET LOCAL hnsw.iterative_scan = {'strict_order' if iterative else 'off'}"))
        rows = conn.execute(stmt).all()
    out = pd.DataFrame(rows, columns=["message_id", "room_id", "body", "distance"])
    out["similarity"] = 1 - out["distance"]  # cosine distance = 1 - cosine similarity
    return out


def search_text(engine: Engine, embedder: Embedder, query: str, room_ids: Sequence[int], k: int = 5,
                **kwargs) -> pd.DataFrame:
    return search(engine, embedder.encode([query])[0], room_ids, embedder.name, k, **kwargs)


def explain(engine: Engine, query_vector: np.ndarray, room_ids: Sequence[int], model: str, k: int = 5) -> str:
    """The query plan, to see whether the HNSW index or an exact sort is used."""
    stmt = text(
        f"EXPLAIN SELECT message_id FROM {TABLE_NAME} "
        "WHERE model = :model AND room_id IN :rooms ORDER BY embedding <=> :q LIMIT :k"
    ).bindparams(bindparam("rooms", expanding=True), bindparam("q", type_=Vector(len(query_vector))))
    params = {"model": model, "rooms": [int(r) for r in room_ids], "k": k,
              "q": np.asarray(query_vector, dtype=np.float32)}
    with engine.connect() as conn:
        return "\n".join(row[0] for row in conn.execute(stmt, params))


def stats(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        if conn.execute(text("SELECT to_regclass(:t)"), {"t": TABLE_NAME}).scalar() is None:
            return pd.DataFrame(columns=["model", "vectors", "rooms"])
        rows = conn.execute(text(f"""
            SELECT model, count(*) AS vectors, count(DISTINCT room_id) AS rooms
            FROM {TABLE_NAME} GROUP BY model ORDER BY model
        """)).all()
    return pd.DataFrame(rows, columns=["model", "vectors", "rooms"])


def index_public_chat(engine: Engine, embedder: Embedder) -> IndexStats:
    """Index chat from public rooms only (data minimisation: private rooms are opt-out by default)."""
    from analytics import cleaning, data_loader

    with engine.connect() as conn:
        raw = data_loader.load_chat_messages(conn, include_private=False)
    messages, _ = cleaning.clean_chat_messages(raw)
    return index_messages(engine, messages[messages["quality"] == cleaning.OK], embedder)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    parser = argparse.ArgumentParser(description="pgvector index for FocusBuddy chat.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="embed public-room chat messages (incremental)")
    s = sub.add_parser("search", help="semantic search within the given rooms")
    s.add_argument("query")
    s.add_argument("--rooms", required=True, help="comma-separated room ids the caller may read")
    s.add_argument("-k", type=int, default=5)
    sub.add_parser("stats")
    args = parser.parse_args(argv)

    load_dotenv()
    from config import Config  # after load_dotenv

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
    if args.command == "stats":
        print(stats(engine).to_string(index=False))
        return 0

    from nlp.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    if args.command == "search" and embedder.name not in set(stats(engine)["model"]):
        print(f"No vectors for {embedder.name} yet: run `python -m ai.retrieval.store index` first.")
        return 2
    if args.command == "index":
        result = index_public_chat(engine, embedder)
        print(f"messages seen {result.seen}, embedded {result.embedded}, unchanged {result.unchanged}")
    else:
        rooms = [int(r) for r in args.rooms.split(",") if r.strip()]
        print(search_text(engine, embedder, args.query, rooms, args.k).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
