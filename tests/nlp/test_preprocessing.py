from __future__ import annotations

import pytest

from nlp.preprocessing import STOPWORDS, normalize, redact, strip_accents, tokenize


class TestRedaction:
    @pytest.mark.parametrize(
        ("text", "placeholder", "secret"),
        [
            ("mail me at maria.k+uni@example.com", "__email__", "maria"),
            ("notes at https://drive.example.com/abc?x=1", "__url__", "drive"),
            ("or www.example.org/page", "__url__", "example"),
            ("call 210 123 4567 later", "__phone__", "4567"),
            ("call +30 6912345678", "__phone__", "6912345678"),
            ("thanks @giorgos!", "__user__", "giorgos"),
        ],
    )
    def test_personal_data_is_replaced(self, text, placeholder, secret):
        out = redact(text)
        assert placeholder in out
        assert secret not in out

    def test_email_is_not_mistaken_for_a_mention(self):
        assert "__user__" not in redact("a@b.com")

    def test_redaction_happens_before_tokenising(self):
        tokens = tokenize("write to eleni@uni.gr or @eleni")
        assert "eleni" not in tokens and "uni" not in tokens
        assert "__email__" in tokens and "__user__" in tokens


class TestNormalisation:
    def test_greek_accents_case_and_final_sigma(self):
        assert normalize("Καλή Επιτυχία ΣΤΙΣ Εξετάσεις") == "καλη επιτυχια στισ εξετασεισ"

    def test_latin_accents(self):
        assert strip_accents("café naïve") == "cafe naive"

    def test_numbers_become_a_placeholder(self):
        assert normalize("25 min, then 5.5 more") == "__num__ min, then __num__ more"

    def test_none_and_whitespace(self):
        assert normalize(None) == ""
        assert normalize("  a \n\t b ") == "a b"


class TestTokenize:
    def test_stopwords_removed_in_all_three_languages(self):
        tokens = tokenize("Το SQL και the JOIN είναι kai join")
        assert tokens == ["sql", "join", "join"]

    def test_accented_stopword_matches(self):
        assert "ειναι" in STOPWORDS and tokenize("Είναι") == []

    def test_single_letters_digits_and_emoji_dropped(self):
        assert tokenize("a b 42 💪 ok") == ["__num__", "ok"]

    def test_same_word_with_and_without_accent_is_one_token(self):
        assert tokenize("Κανονικοποίηση") == tokenize("κανονικοποιηση")
