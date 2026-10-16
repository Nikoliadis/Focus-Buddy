"""Embeddings: vector maths and the index with a fake embedder (fast, offline), plus
semantic checks against the real multilingual model when it is in the local cache."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from nlp import retrieval_eval as rev
from nlp.embeddings import (
    DEFAULT_MODEL,
    SemanticIndex,
    cosine_matrix,
    embed_unique,
    model_is_cached,
    top_k,
)


class HashEmbedder:
    """Deterministic stand-in: bag of hashed lowercase words, L2-normalised. Records its inputs."""

    name, dimension = "hash", 64

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def messages(bodies):
    return pd.DataFrame({"message_id": range(1, len(bodies) + 1), "room_id": 1, "body": bodies})


class TestVectorMaths:
    def test_cosine_of_unit_vectors_is_the_dot_product(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(5, 8))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        expected = np.array([[a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) for b in A] for a in A])
        np.testing.assert_allclose(cosine_matrix(A, A), expected)
        np.testing.assert_allclose(np.diag(cosine_matrix(A, A)), 1)

    def test_top_k_matches_a_full_sort(self):
        scores = np.random.default_rng(1).normal(size=1000)
        np.testing.assert_array_equal(top_k(scores, 7), np.argsort(-scores)[:7])

    def test_top_k_edge_cases(self):
        assert top_k(np.array([0.2, 0.9]), 5).tolist() == [1, 0]
        assert top_k(np.array([]), 3).size == 0


class TestIndexWithFakeEmbedder:
    def test_each_distinct_text_is_encoded_once(self):
        embedder = HashEmbedder()
        vectors = embed_unique(embedder, ["Back!", "Break", "Back!", "Back!"])

        assert sorted(embedder.calls[0]) == ["Back!", "Break"]
        np.testing.assert_allclose(vectors[0], vectors[2])
        assert vectors.shape == (4, 64)

    def test_search_ranks_the_closest_message_first(self):
        index = SemanticIndex(HashEmbedder(), messages(["sql join help", "coffee break", "graph heap"]))
        hits = index.search("sql join", k=2)

        assert hits["body"].iloc[0] == "sql join help"
        assert hits["cosine"].is_monotonic_decreasing
        assert index.search_ids("graph", 1) == [3]


class TestRetrievalMetrics:
    ranked = ["a", "x", "b", "y", "c"]
    relevant = {"a", "b", "c", "d"}

    def test_hand_checked(self):
        assert rev.precision_at_k(self.ranked, self.relevant, 5) == pytest.approx(3 / 5)
        assert rev.recall_at_k(self.ranked, self.relevant, 5) == pytest.approx(3 / 4)
        assert rev.reciprocal_rank(self.ranked, self.relevant) == 1
        assert rev.reciprocal_rank(["x", "y", "b"], self.relevant) == pytest.approx(1 / 3)
        dcg = 1 / np.log2(2) + 1 / np.log2(4) + 1 / np.log2(6)
        ideal = sum(1 / np.log2(i + 2) for i in range(4))
        assert rev.ndcg_at_k(self.ranked, self.relevant, 5) == pytest.approx(dcg / ideal)

    def test_perfect_and_empty(self):
        assert rev.ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1)
        assert rev.reciprocal_rank(["x"], {"a"}) == 0
        assert np.isnan(rev.recall_at_k(["a"], set(), 1))

    def test_evaluate(self):
        frame = rev.evaluate(lambda q, k: ["a", "b"], [("q1", {"a"}), ("q2", {"z"})], k=2)
        assert frame["mrr"].tolist() == [1, 0]
        assert {"precision@2", "recall@2", "ndcg@2"} <= set(frame.columns)


def test_redaction_happens_before_embedding(monkeypatch):
    from nlp import embeddings

    seen = {}

    class FakeModel:
        def encode(self, texts, **kwargs):
            seen["texts"] = texts
            return np.ones((len(texts), 3), dtype=np.float32)

    embedder = embeddings.SentenceTransformerEmbedder()
    monkeypatch.setattr(embedder, "_model", FakeModel())
    embedder.encode(["mail maria@uni.gr", "ping @nikos"])

    assert "maria" not in " ".join(seen["texts"]) and "nikos" not in " ".join(seen["texts"])
    assert "__email__" in seen["texts"][0]


real_model = pytest.mark.skipif(not model_is_cached(), reason=f"{DEFAULT_MODEL} not in the local HF cache")


@real_model
class TestRealModel:
    @pytest.fixture(scope="class")
    def embedder(self):
        from nlp.embeddings import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(device="cpu")

    def test_cross_lingual_paraphrases_are_close(self, embedder):
        v = embedder.encode([
            "normalization of a database schema",
            "κανονικοποίηση σχήματος βάσης δεδομένων",
            "coffee break",
        ])
        sims = cosine_matrix(v, v)
        assert v.shape[1] == 384
        assert sims[0, 1] > 0.6
        assert sims[0, 1] > sims[0, 2] + 0.4

    def test_embeddings_beat_tfidf_on_labelled_queries(self, embedder):
        from nlp.embeddings import compare_with_tfidf

        means = compare_with_tfidf(embedder).groupby("method")[["mrr", "ndcg@10"]].mean()
        assert means.loc["embeddings", "mrr"] > means.loc["tfidf", "mrr"] + 0.2
        assert means.loc["embeddings", "ndcg@10"] > means.loc["tfidf", "ndcg@10"] + 0.2
