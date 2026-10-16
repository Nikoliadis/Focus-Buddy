"""Text normalisation and tokenisation for FocusBuddy chat (Greek, English and Greeklish).

Order matters:
1. Redact personal data first (emails, URLs, phone numbers, @mentions), so it can never
   reach a vocabulary, a model or a report.
2. Normalise: lowercase, strip accents (Greek tonos), unify final sigma, replace numbers.
3. Tokenise into words of letters and drop stopwords.

Stopwords are normalised with the same function as the text, so "και" and "Καί" match.
"""
from __future__ import annotations

import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
MENTION_RE = re.compile(r"(?<!\w)@\w+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

PLACEHOLDERS = ("__email__", "__url__", "__phone__", "__user__", "__num__")

# Letters only (any script), at least 2 long; or one of our placeholders.
TOKEN_RE = re.compile(r"__(?:email|url|phone|user|num)__|[^\W\d_]{2,}")


def strip_accents(text: str) -> str:
    """Remove combining marks: 'επιτυχία' -> 'επιτυχια', 'café' -> 'cafe'."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def redact(text: str) -> str:
    """Replace personal identifiers with placeholders. Emails before mentions: 'a@b.com'
    must not become '__user__.com'."""
    text = EMAIL_RE.sub(" __email__ ", text)
    text = URL_RE.sub(" __url__ ", text)
    text = PHONE_RE.sub(" __phone__ ", text)
    return MENTION_RE.sub(" __user__ ", text)


def normalize(text: str) -> str:
    text = redact(text or "")
    text = strip_accents(text.lower()).replace("ς", "σ")
    text = NUMBER_RE.sub(" __num__ ", text)
    return re.sub(r"\s+", " ", text).strip()


_GREEK_STOPWORDS = """
και να το τα την τη της τον των του τους στο στα στη στην στις στον σε με για απο από που πως
ως αν δεν μη μην θα ειναι είναι ηταν ήταν εχω έχω εχει έχει εχουμε έχουμε ενα ένα μια μία ενας
ένας ο η οι αυτο αυτό αυτα αυτά εγω εγώ εσυ εσύ εμεις εμείς μου σου μας σας τι ποιο ποια οτι ότι
αλλα αλλά ή η κι ομως όμως πολυ πολύ λιγο λίγο ακομα ακόμα τωρα τώρα εδω εδώ εκει εκεί πιο
""".split()
_GREEKLISH_STOPWORDS = "kai na to ta tin ti tis ton tou se me gia apo pou den tha einai exw ena mia".split()

STOPWORDS = frozenset(normalize(w) for w in (*ENGLISH_STOP_WORDS, *_GREEK_STOPWORDS, *_GREEKLISH_STOPWORDS))


def tokenize(text: str) -> list[str]:
    """Normalised tokens without stopwords. Used as the TfidfVectorizer tokenizer."""
    return [t for t in TOKEN_RE.findall(normalize(text)) if t not in STOPWORDS]
