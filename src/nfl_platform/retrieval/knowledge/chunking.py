"""Heading-aware chunking.

Markdown headings are an author's own statement of where one idea ends and the
next begins, so sections are the natural chunk boundary. A fixed-size sliding
window over the whole document would cut definitions in half, which is the
failure mode that makes small corpora retrieve badly.

Only oversized sections are split further, with overlap so a definition spanning
the split is still wholly present in one chunk.
"""

from __future__ import annotations

import hashlib
import re

from nfl_platform.retrieval.knowledge.corpus import KnowledgeDocument
from nfl_platform.retrieval.models import KnowledgeChunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# Chunk sizes are configured in tokens but measured here in characters, because
# counting real tokens would mean importing a tokeniser for a boundary decision
# that does not need that precision. English prose runs about four characters
# per token.
CHARS_PER_TOKEN = 4


def _sections(text: str) -> list[tuple[str, str]]:
    """Split a document body into (heading, body) pairs."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match is None:
            buffer.append(line)
            continue
        if any(item.strip() for item in buffer):
            sections.append((heading, "\n".join(buffer).strip()))
        heading = match.group(2).strip()
        buffer = []
    if any(item.strip() for item in buffer):
        sections.append((heading, "\n".join(buffer).strip()))
    return sections


def _split_oversized(body: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Break a long section on paragraph boundaries, keeping a little overlap."""
    if len(body) <= max_chars:
        return [body]

    paragraphs = [item.strip() for item in _PARAGRAPH_BREAK.split(body) if item.strip()]
    pieces: list[str] = []
    current: list[str] = []
    length = 0

    for paragraph in paragraphs:
        if current and length + len(paragraph) > max_chars:
            pieces.append("\n\n".join(current))
            tail = current[-1]
            # Carry the previous paragraph forward when it is small enough to be
            # context rather than a duplicate chunk in its own right.
            current = [tail] if len(tail) <= overlap_chars else []
            length = sum(len(item) for item in current)
        current.append(paragraph)
        length += len(paragraph)

    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_document(
    document: KnowledgeDocument,
    *,
    chunk_tokens: int = 350,
    chunk_overlap: int = 50,
) -> list[KnowledgeChunk]:
    """Split one document into indexable chunks with their metadata."""
    max_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * CHARS_PER_TOKEN

    chunks: list[KnowledgeChunk] = []
    seen_hashes: set[str] = set()

    for heading, body in _sections(document.text):
        for piece in _split_oversized(body, max_chars, overlap_chars):
            text = piece.strip()
            if not text:
                continue
            # The heading is prepended to the embedded text because a section
            # body often omits the term it defines ("Awarded for each reception"
            # only matches a PPR query once "PPR" is part of the chunk).
            embedded = (
                f"{document.title} — {heading}\n\n{text}"
                if heading
                else (f"{document.title}\n\n{text}")
            )
            content_hash = hashlib.sha256(embedded.encode("utf-8")).hexdigest()[:16]
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            index = len(chunks)
            chunks.append(
                KnowledgeChunk(
                    document_id=document.document_id,
                    chunk_id=f"{document.document_id}#{index:03d}",
                    title=document.title,
                    section=heading,
                    text=embedded,
                    document_type=document.document_type,
                    source=document.source,
                    updated=document.updated,
                    chunk_index=index,
                    content_hash=content_hash,
                )
            )
    return chunks
