"""Retrieval quality metrics. Used for TF-IDF vs embeddings here, and for RAG later.

For one query, with ``ranked`` the retrieved ids in order and ``relevant`` the true answers:

    precision@k   share of the top k that are relevant
    recall@k      share of all relevant items found in the top k
    MRR           1 / rank of the first relevant item (0 if none): "how soon is the first good hit"
    nDCG@k        discounted cumulative gain: relevant hits count less the lower they are ranked
                  (1 / log2(rank + 1)), normalised by the best possible ordering, so 1 = perfect
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


def precision_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    top = list(ranked)[:k]
    return sum(item in relevant for item in top) / k if k else 0.0


def recall_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return float("nan")
    return sum(item in relevant for item in list(ranked)[:k]) / len(relevant)


def reciprocal_rank(ranked: Sequence, relevant: set) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    gains = [1.0 if item in relevant else 0.0 for item in list(ranked)[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def evaluate(
    search: Callable[[str, int], Sequence],
    queries: Iterable[tuple[str, set]],
    k: int = 10,
) -> pd.DataFrame:
    """One row per query; ``search(query, k)`` returns ranked ids."""
    rows = []
    for query, relevant in queries:
        ranked = list(search(query, k))
        rows.append({
            "query": query, "relevant": len(relevant),
            f"precision@{k}": precision_at_k(ranked, relevant, k),
            f"recall@{k}": recall_at_k(ranked, relevant, k),
            "mrr": reciprocal_rank(ranked, relevant),
            f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
        })
    return pd.DataFrame(rows)
