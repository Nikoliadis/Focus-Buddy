"""Retrieve knowledge chunks for a question.

The knowledge base is a few dozen chunks, so the vectors live in memory: embedding them
takes about a second, and a brute-force matrix product over 50 rows is exact and instant.
pgvector (Phase 16) is for data that grows, such as chat messages; using it here would add
a dependency without a benefit.

Two guards learned in Phase 16:
- ``min_similarity``: top-k always returns *something*, even for an unrelated question.
  Below the threshold a chunk is dropped; with nothing left, the bot must not pretend to
  know. The threshold is calibrated on labelled questions (ai.rag.evaluate).
- one chunk per section: overlapping windows of the same section are near-duplicates and
  would waste the context budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai.rag.knowledge import Chunk, load_knowledge
from nlp.embeddings import Embedder, top_k

# A deliberately LOW threshold (see `python -m ai.rag.evaluate`, multilingual MiniLM). It only
# removes clearly unrelated questions (0.09-0.22 for "capital of France", "pizza"). Near misses
# overlap with real questions: "How do I delete my account?" (unanswerable) scores 0.418 against
# "deleting a room", above "How do I switch to dark mode?" (answerable, 0.311). The best
# threshold on the 32 labelled questions (0.425) would sit right on top of those cases, i.e.
# overfit. Near misses are left to the second layer: the prompt tells the model to say it does
# not know when the sources do not answer the question.
DEFAULT_MIN_SIMILARITY = 0.30


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    similarity: float


class KnowledgeRetriever:
    def __init__(self, embedder: Embedder, chunks: list[Chunk] | None = None,
                 min_similarity: float = DEFAULT_MIN_SIMILARITY):
        self.embedder = embedder
        self.chunks = chunks if chunks is not None else load_knowledge()
        self.min_similarity = min_similarity
        self.vectors = embedder.encode([c.embedding_text for c in self.chunks])

    def scores(self, question: str) -> np.ndarray:
        return self.vectors @ self.embedder.encode([question])[0]

    def retrieve(self, question: str, k: int = 4, min_similarity: float | None = None) -> list[Retrieved]:
        threshold = self.min_similarity if min_similarity is None else min_similarity
        scores = self.scores(question)
        hits, seen_sections = [], set()
        for i in top_k(scores, len(self.chunks)):
            if scores[i] < threshold or len(hits) == k:
                break
            chunk = self.chunks[i]
            key = (chunk.doc, chunk.section)
            if key in seen_sections:
                continue
            seen_sections.add(key)
            hits.append(Retrieved(chunk, float(scores[i])))
        return hits
