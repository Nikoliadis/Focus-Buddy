"""Sentence embeddings: text -> dense vector whose direction encodes meaning.

    "normalization of a schema"      ─┐
    "κανονικοποίηση σχήματος βάσης"  ─┼─► multilingual transformer ─► 384-d unit vectors
    "coffee break"                   ─┘
    cosine(vec_a, vec_b) = a · b  (unit length)  ->  ~0.8 for the first two, ~0 for the third

Unlike TF-IDF (a sparse count of shared words), a transformer reads the whole sentence
and maps paraphrases and translations close together. Text is only *redacted* first (no
lowercasing or stopword removal): the model was trained on natural sentences and uses
word order, case and function words.

Usage:
    python -m nlp.embeddings --compare      # TF-IDF vs embeddings on labelled synthetic queries
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from nlp.preprocessing import redact

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised float32 vectors."""

    name: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Lazy-loading wrapper: the model (~470 MB) is downloaded and loaded on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto", batch_size: int = 64):
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            from ml.deep_learning.mlp import resolve_device

            self._model = SentenceTransformer(self.name, device=str(resolve_device(self.device)))
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        clean = [redact(t or "") for t in texts]  # personal data never reaches the model
        return self.model.encode(clean, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def model_is_cached(model_name: str = DEFAULT_MODEL) -> bool:
    """True if the model is already in the local Hugging Face cache (no download needed)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(model_name, "config.json")
    return isinstance(path, str)


def embed_unique(embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
    """Encode each distinct text once and map back. Chat repeats a lot ("Back!", "Break time"),
    and a transformer forward pass is the expensive step."""
    texts = list(texts)
    unique, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    return embedder.encode(list(unique))[inverse]


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of unit vectors: a single matrix multiplication."""
    return A @ B.T


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first. argpartition is O(n) instead of a full
    O(n log n) sort; only the k winners are sorted."""
    k = min(k, scores.size)
    if k == 0:
        return np.array([], dtype=int)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


class SemanticIndex:
    """In-memory vector index over messages (Phase 16 moves the vectors into PostgreSQL)."""

    def __init__(self, embedder: Embedder, messages: pd.DataFrame):
        self.embedder = embedder
        self.messages = messages.reset_index(drop=True)
        self.vectors = embed_unique(embedder, self.messages["body"].tolist())

    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        q = self.embedder.encode([query])[0]
        scores = self.vectors @ q
        idx = top_k(scores, k)
        hits = self.messages.iloc[idx][["message_id", "room_id", "body"]].copy()
        hits["cosine"] = scores[idx]
        return hits

    def search_ids(self, query: str, k: int) -> list[int]:
        return self.search(query, k)["message_id"].tolist()


def compare_with_tfidf(embedder: Embedder, seed: int = 0, k: int = 10) -> pd.DataFrame:
    """TF-IDF vs embeddings on the labelled synthetic queries. One row per query and method."""
    from nlp import retrieval_eval, synthetic, tfidf

    messages, _ = synthetic.generate_chat(seed=seed)
    tf = tfidf.fit_tfidf(messages)
    index = SemanticIndex(embedder, messages)
    queries = [(q, synthetic.relevant_ids(messages, kind, target)) for q, kind, target in synthetic.EVAL_QUERIES]

    def tfidf_search(query: str, n: int) -> list[int]:
        return tfidf.most_similar(tf, query, n)["message_id"].tolist()

    frames = []
    for method, search in [("tfidf", tfidf_search), ("embeddings", index.search_ids)]:
        frame = retrieval_eval.evaluate(search, queries, k)
        frame.insert(0, "method", method)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from nlp.synthetic import EVAL_QUERIES

    parser = argparse.ArgumentParser(description="Sentence embeddings for FocusBuddy chat.")
    parser.add_argument("--compare", action="store_true", help="TF-IDF vs embeddings on synthetic queries")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    embedder = SentenceTransformerEmbedder(args.model, args.device)
    if args.compare:
        results = compare_with_tfidf(embedder, k=args.k)
        print("=" * 76 + "\n  SYNTHETIC CHAT: generated text, not real messages\n" + "=" * 76)
        metrics = [c for c in results.columns if c not in ("method", "query", "relevant")]
        print(results.groupby("method", sort=False)[metrics].mean().round(3).to_string())
        print()
        wide = results.pivot(index="query", columns="method", values="mrr")[["tfidf", "embeddings"]]
        print("MRR per query:")
        print(wide.reindex([q for q, _, _ in EVAL_QUERIES]).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
