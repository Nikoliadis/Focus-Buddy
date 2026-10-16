from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from nlp import tfidf
from nlp.preprocessing import normalize
from nlp.synthetic import TOPICS, generate_chat


def corpus(bodies: list[str], users: list[int] | None = None, rooms: list[int] | None = None) -> pd.DataFrame:
    n = len(bodies)
    return pd.DataFrame({
        "message_id": range(1, n + 1),
        "room_id": rooms or [1] * n,
        "user_id": users or list(range(n)),
        "body": bodies,
    })


@pytest.fixture(scope="module")
def synthetic():
    messages, topics = generate_chat(seed=4)
    return tfidf.fit_tfidf(messages), topics


class TestWeights:
    def test_hand_checked_idf_and_normalisation(self):
        docs = corpus(["sql join", "sql index", "graph heap"])
        result = tfidf.fit_tfidf(docs, min_df=1, max_df=1.0, ngram_range=(1, 1), sublinear_tf=False)
        vocab = dict(zip(result.terms, range(len(result.terms)), strict=True))

        n = 3
        idf = lambda df: np.log((1 + n) / (1 + df)) + 1  # noqa: E731
        row = result.matrix.getrow(0).toarray().ravel()   # one row, tiny
        expected = np.zeros(len(vocab))
        expected[vocab["sql"]], expected[vocab["join"]] = idf(2), idf(1)
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(row, expected)
        assert row[vocab["join"]] > row[vocab["sql"]]   # rarer term weighs more

    def test_rows_are_unit_length(self, synthetic):
        result, _ = synthetic
        norms = np.sqrt(np.asarray(result.matrix.multiply(result.matrix).sum(axis=1)).ravel())
        nonempty = norms > 0
        np.testing.assert_allclose(norms[nonempty], 1)

    def test_max_df_drops_ubiquitous_terms(self):
        docs = corpus(["focus sql", "focus join", "focus graph", "focus heap"])
        result = tfidf.fit_tfidf(docs, min_df=1, max_df=0.5, ngram_range=(1, 1))
        assert "focus" not in result.terms

    def test_empty_corpus(self):
        with pytest.raises(ValueError):
            tfidf.fit_tfidf(corpus([]))


class TestSparsity:
    def test_matrix_is_sparse_and_report_is_consistent(self, synthetic):
        result, _ = synthetic
        s = tfidf.sparsity(result.matrix)

        assert sparse.issparse(result.matrix)
        assert s["density"] < 0.05
        assert s["dense_bytes"] == s["rows"] * s["cols"] * 8
        assert s["dense_to_sparse_ratio"] > 10

    def test_analyses_never_densify_large_matrices(self, synthetic, monkeypatch):
        """Only (n × 1) or smaller results may be densified."""
        original = sparse.csr_matrix.toarray

        def guarded(self, *args, **kwargs):
            if min(self.shape) > 1:
                raise AssertionError(f"dense conversion of a {self.shape} sparse matrix")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(sparse.csr_matrix, "toarray", guarded)
        monkeypatch.setattr(sparse.csr_matrix, "todense", guarded)
        result, _ = synthetic

        tfidf.top_terms(result)
        tfidf.top_terms_by_group(result)
        tfidf.distinctive_terms_by_group(result)
        tfidf.most_similar(result, "dijkstra graph")
        tfidf.distinct_users_per_term(result.matrix, result.messages["user_id"])


class TestPrivacy:
    def test_a_term_used_by_one_person_is_never_shown(self):
        bodies = ["konstantinos says hi"] * 10 + ["sql join help", "sql index help", "sql query help"]
        users = [7] * 10 + [1, 2, 3]
        result = tfidf.fit_tfidf(corpus(bodies, users), min_df=1, max_df=1.0)

        shown = set(tfidf.top_terms(result, k=50, min_users=3)["term"])
        assert "konstantinos" not in shown       # frequent, but one author
        assert "sql" in shown

    def test_placeholders_hidden_by_default(self):
        bodies = [f"see https://x.org/{i} sql" for i in range(5)]
        result = tfidf.fit_tfidf(corpus(bodies), min_df=1, max_df=1.0)
        assert "__url__" in result.terms
        assert "__url__" not in set(tfidf.top_terms(result, k=50, min_users=1)["term"])

    def test_distinct_users_per_term(self):
        result = tfidf.fit_tfidf(corpus(["sql", "sql", "heap"], users=[1, 1, 2]), min_df=1, max_df=1.0,
                                 ngram_range=(1, 1))
        counts = dict(zip(result.terms, tfidf.distinct_users_per_term(result.matrix, result.messages["user_id"]),
                          strict=True))
        assert counts == {"sql": 1, "heap": 1}


class TestTopicRecovery:
    def test_distinctive_terms_recover_each_rooms_topic(self, synthetic):
        result, topics = synthetic
        topic_tokens = {t: {tok for phrase in words for tok in normalize(phrase).split()}
                        for t, words in TOPICS.items()}

        for room, terms in tfidf.distinctive_terms_by_group(result, k=6).items():
            own = topic_tokens[topics[room]]
            hits = sum(any(tok in own for tok in term.split()) for term in terms["term"])
            assert hits >= 4, (room, list(terms["term"]))

    def test_distinctive_beats_plain_within_room_mean(self, synthetic):
        """Plain means rank chit-chat high in some rooms; distinctiveness removes it."""
        result, _ = synthetic
        chitchat = {"break", "tomorrow", "back", "καλημερα"}
        plain = sum(t in chitchat for d in tfidf.top_terms_by_group(result, k=3).values() for t in d["term"])
        distinct = sum(t in chitchat for d in tfidf.distinctive_terms_by_group(result, k=3).values()
                       for t in d["term"])
        assert distinct < plain


class TestSimilarity:
    def test_identical_message_ranks_first(self, synthetic):
        result, _ = synthetic
        query = result.messages["body"].iloc[42]

        hits = tfidf.most_similar(result, query)

        assert hits["body"].iloc[0] == query
        assert hits["cosine"].iloc[0] == pytest.approx(1, abs=1e-6)
        assert (hits["cosine"].diff().dropna() <= 1e-12).all()  # sorted

    def test_lexical_overlap_beats_meaning(self, synthetic):
        """Known limitation (motivates embeddings): asking about 'dijkstra' in a template shared
        by many messages ranks other topics with the same wording above messages that are
        actually about dijkstra but phrased differently."""
        result, _ = synthetic
        query = "Does anyone have notes on dijkstra?"
        assert not (result.messages["body"] == query).any()

        top = tfidf.most_similar(result, query, k=3)["body"].str.lower()

        assert not top.str.contains("dijkstra").any()
        assert top.str.startswith("does anyone have notes on").all()

    def test_unknown_words_find_nothing(self, synthetic):
        result, _ = synthetic
        assert tfidf.most_similar(result, "zzzz qqqq").empty


def test_report_text(synthetic):
    result, topics = synthetic
    text = tfidf.report_text(result, "synthetic", topics)
    assert "SYNTHETIC CHAT" in text and "density" in text and "redacted" in text
