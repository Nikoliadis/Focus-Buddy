"""Load the knowledge base and split it into retrievable chunks.

Chunking trade-off:
- too large: one chunk mixes several topics, its embedding is a blurry average, and it
  wastes the context budget on irrelevant sentences;
- too small: a chunk loses the context needed to understand it ("It is valid for 1 hour":
  what is?).

Strategy: split on markdown ``##`` sections (the author's own topic boundaries), split a
section further only if it exceeds ``max_words``, with ``overlap`` words repeated between
neighbouring pieces so a sentence cut at the border appears whole in one of them. Every
chunk carries its headings ("Account and profile > Forgotten password"), which are part
of the embedded text: a chunk then says what it is about even when its body does not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # "focusbuddy_account#forgotten-password#0"
    doc: str           # file stem
    title: str         # "# ..." heading of the document
    section: str       # "## ..." heading
    text: str          # body only

    @property
    def heading(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title

    @property
    def embedding_text(self) -> str:
        return f"{self.heading}\n{self.text}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _windows(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    if len(words) <= max_words:
        return [words]
    step = max_words - overlap
    return [words[i:i + max_words] for i in range(0, max(len(words) - overlap, 1), step)]


def chunk_markdown(markdown: str, doc: str, max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must be >= 0 and smaller than max_words")
    title, section, buffer, chunks = doc, "", [], []

    def flush():
        body = " ".join(" ".join(buffer).split())
        if body:
            for i, window in enumerate(_windows(body.split(), max_words, overlap)):
                chunks.append(Chunk(f"{doc}#{_slug(section) or 'intro'}#{i}", doc, title, section, " ".join(window)))
        buffer.clear()

    for line in markdown.splitlines():
        if line.startswith("# "):
            flush()
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section = line[3:].strip()
        else:
            buffer.append(line)
    flush()
    return chunks


def load_knowledge(directory: Path = KNOWLEDGE_DIR, **chunk_kwargs) -> list[Chunk]:
    chunks = []
    for path in sorted(directory.glob("*.md")):
        chunks += chunk_markdown(path.read_text(encoding="utf-8"), path.stem, **chunk_kwargs)
    if not chunks:
        raise FileNotFoundError(f"no knowledge documents in {directory}")
    return chunks
