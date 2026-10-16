"""TF-IDF over chat messages, kept sparse end to end.

    chat ─► redact + normalise + tokenise ─► TfidfVectorizer ─► CSR matrix (messages × terms)
                                                                   │
                       important terms overall / per room ◄────────┤ sparse aggregations
                       most similar messages to a query   ◄────────┘ sparse dot products

TF-IDF weight of term t in message d:
    tf(t, d)   how often t occurs in d          (sublinear: 1 + log(count))
    idf(t)     ln((1 + n) / (1 + df(t))) + 1    df = number of messages containing t
    w(t, d)    tf · idf, then each row is L2-normalised
A term frequent in one message but rare across messages gets a high weight; words that
are everywhere ("break", "back") get a low one.

Most entries are zero: a message uses a handful of the vocabulary. Every operation here
works on the sparse representation; ``.toarray()`` is only ever applied to results of
size (n × 1) or (groups × k).

Usage:
    python -m nlp.tfidf --source synthetic
    python -m nlp.tfidf --source db           # public rooms only, read-only
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp.preprocessing import PLACEHOLDERS, tokenize

DEFAULT_MIN_USERS = 3


@dataclass
class TfidfResult:
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix      # rows aligned with `messages`
    messages: pd.DataFrame
    terms: np.ndarray

    def term_mask(self, min_users: int = DEFAULT_MIN_USERS, include_placeholders: bool = False) -> np.ndarray:
        """Terms safe to show: used by at least ``min_users`` distinct users (so one person's
        name or private phrase never surfaces), and optionally not a redaction placeholder."""
        mask = distinct_users_per_term(self.matrix, self.messages["user_id"]) >= min_users
        if not include_placeholders:
            mask &= ~np.isin(self.terms, PLACEHOLDERS)
        return mask


def fit_tfidf(
    messages: pd.DataFrame,
    min_df: int = 2,
    max_df: float = 0.5,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
) -> TfidfResult:
    """``min_df=2`` drops terms seen in a single message (typos, names); ``max_df=0.5`` drops
    terms in more than half of all messages (they distinguish nothing)."""
    if messages.empty:
        raise ValueError("no messages to analyse")
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, token_pattern=None, lowercase=False,  # tokenize() normalises itself
        min_df=min_df, max_df=max_df, ngram_range=ngram_range, sublinear_tf=sublinear_tf,
    )
    matrix = vectorizer.fit_transform(messages["body"]).tocsr()
    return TfidfResult(vectorizer, matrix, messages.reset_index(drop=True), vectorizer.get_feature_names_out())


def sparsity(matrix: sparse.spmatrix) -> dict[str, float]:
    """How sparse the matrix is and what densifying it would cost."""
    csr = matrix.tocsr()
    rows, cols = csr.shape
    sparse_bytes = csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes
    dense_bytes = rows * cols * csr.dtype.itemsize
    return {
        "rows": rows, "cols": cols, "nnz": int(csr.nnz),
        "density": csr.nnz / (rows * cols) if rows * cols else 0.0,
        "sparse_bytes": int(sparse_bytes), "dense_bytes": int(dense_bytes),
        "dense_to_sparse_ratio": dense_bytes / sparse_bytes if sparse_bytes else float("inf"),
    }


def _group_indicator(groups: pd.Series) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Sparse (n_groups × n_rows) 0/1 matrix: G @ X sums the rows of each group."""
    keys, idx = np.unique(groups.to_numpy(), return_inverse=True)
    G = sparse.csr_matrix((np.ones(len(idx)), (idx, np.arange(len(idx)))), shape=(len(keys), len(idx)))
    return G, keys


def distinct_users_per_term(matrix: sparse.csr_matrix, user_ids: pd.Series) -> np.ndarray:
    """For each term, how many distinct users used it at least once."""
    U, _ = _group_indicator(user_ids)
    used = U @ (matrix > 0).astype(np.float64)        # users × terms, still sparse
    return np.asarray((used > 0).sum(axis=0)).ravel()


def top_terms(result: TfidfResult, k: int = 15, min_users: int = DEFAULT_MIN_USERS) -> pd.DataFrame:
    """Terms with the highest mean TF-IDF over all messages."""
    mean = np.asarray(result.matrix.mean(axis=0)).ravel()   # 1 × terms, dense but tiny
    doc_freq = result.matrix.getnnz(axis=0)
    mask = result.term_mask(min_users)
    order = [i for i in np.argsort(-mean) if mask[i]][:k]
    return pd.DataFrame({"term": result.terms[order], "mean_tfidf": mean[order], "messages": doc_freq[order]})


def top_terms_by_group(
    result: TfidfResult, group_col: str = "room_id", k: int = 8, min_users: int = DEFAULT_MIN_USERS
) -> dict[object, pd.DataFrame]:
    """Most characteristic terms of each group (e.g. room): mean TF-IDF within the group."""
    G, keys = _group_indicator(result.messages[group_col])
    sizes = np.asarray(G.sum(axis=1)).ravel()
    means = sparse.diags(1 / sizes) @ (G @ result.matrix)   # groups × terms, sparse
    mask = result.term_mask(min_users)
    out = {}
    for row, key in enumerate(keys):
        r = means.getrow(row)
        keep = mask[r.indices]
        indices, values = r.indices[keep], r.data[keep]
        order = np.argsort(-values)[:k]
        out[key] = pd.DataFrame({"term": result.terms[indices[order]], "mean_tfidf": values[order]})
    return out


def distinctive_terms_by_group(
    result: TfidfResult, group_col: str = "room_id", k: int = 8, min_users: int = DEFAULT_MIN_USERS
) -> dict[object, pd.DataFrame]:
    """Terms that set each group apart: mean TF-IDF in the group minus the mean over all
    messages. Words common everywhere ("break", "tomorrow") score ~0 even if frequent in
    the group, which plain within-group means do not achieve."""
    G, keys = _group_indicator(result.messages[group_col])
    sizes = np.asarray(G.sum(axis=1)).ravel()
    means = sparse.diags(1 / sizes) @ (G @ result.matrix)
    overall = np.asarray(result.matrix.mean(axis=0)).ravel()
    mask = result.term_mask(min_users)
    out = {}
    for row, key in enumerate(keys):
        r = means.getrow(row)
        keep = mask[r.indices]
        indices, values = r.indices[keep], r.data[keep]
        lift = values - overall[indices]
        order = np.argsort(-lift)[:k]
        out[key] = pd.DataFrame({"term": result.terms[indices[order]], "distinctiveness": lift[order],
                                 "group_mean": values[order], "overall_mean": overall[indices[order]]})
    return out


def most_similar(result: TfidfResult, query: str, k: int = 5) -> pd.DataFrame:
    """Messages closest to ``query`` by cosine similarity. Rows are L2-normalised, so the
    dot product IS the cosine; the (n × 1) result is the only thing densified."""
    q = result.vectorizer.transform([query])
    scores = (result.matrix @ q.T).toarray().ravel()
    order = np.argsort(-scores)[:k]
    hits = result.messages.iloc[order][["message_id", "room_id", "body"]].copy()
    hits["cosine"] = scores[order]
    return hits[hits["cosine"] > 0]


def report_text(result: TfidfResult, data_source: str, room_topics: dict | None = None) -> str:
    s = sparsity(result.matrix)
    banner = "=" * 76
    lines = []
    if data_source != "db":
        lines += [banner, f"  {data_source.upper()} CHAT: generated text, not real messages", banner]
    lines += [
        f"Messages: {s['rows']}   vocabulary (unigrams + bigrams): {s['cols']}   "
        f"authors: {result.messages['user_id'].nunique()}",
        f"Sparse matrix: {s['nnz']} non-zeros, density {s['density']:.4%}, "
        f"{s['sparse_bytes'] / 1024:.0f} KB sparse vs {s['dense_bytes'] / 1024:.0f} KB dense "
        f"({s['dense_to_sparse_ratio']:.0f}x)",
        f"Privacy: terms shown only if used by >= {DEFAULT_MIN_USERS} distinct users; "
        "emails, URLs, phones and @mentions redacted before tokenising",
        "",
        "Top terms overall:",
        top_terms(result, 12).round(4).to_string(index=False),
        "",
        "Most distinctive terms per room (room mean minus overall mean):",
    ]
    for room, terms in distinctive_terms_by_group(result).items():
        topic = f" (generated topic: {room_topics[room]})" if room_topics else ""
        lines.append(f"  room {room}{topic}: " + ", ".join(terms["term"].head(6)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="TF-IDF analysis of FocusBuddy chat.")
    parser.add_argument("--source", choices=["synthetic", "db"], default="synthetic")
    parser.add_argument("--query", default="notes on normalization")
    args = parser.parse_args(argv)

    room_topics = None
    if args.source == "synthetic":
        from nlp.synthetic import generate_chat
        messages, room_topics = generate_chat()
    else:
        from analytics import cleaning, data_loader
        from analytics.quality import engine_from_env
        engine = engine_from_env()
        if engine is None:
            print("DATABASE_URL is not set.", file=sys.stderr)
            return 1
        with engine.connect() as conn:
            raw = data_loader.load_chat_messages(conn)  # public rooms only
        messages, _ = cleaning.clean_chat_messages(raw)
        messages = messages[messages["quality"] == cleaning.OK]
    try:
        result = fit_tfidf(messages)
    except ValueError as exc:  # empty corpus, or every term below min_df
        print(f"Not analysing: {exc}")
        return 2
    print(report_text(result, args.source, room_topics))
    print(f"\nMost similar to {args.query!r}:")
    print(most_similar(result, args.query).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
