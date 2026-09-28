"""Loading the curated markdown corpus.

Documents carry front-matter so that metadata -- what this is, where it came
from, when it was last touched -- travels with the text into the index and out
again to the model.

The front-matter parser is deliberately a handful of lines rather than PyYAML.
The schema is six flat string keys; adding a YAML dependency, and a YAML parser's
attack surface, to read ``title: Something`` would not be a trade worth making.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from nfl_platform.config.logging import get_logger
from nfl_platform.exceptions import CorpusValidationError

logger = get_logger(__name__)

FRONT_MATTER_DELIMITER = "---"
REQUIRED_FIELDS = ("document_id", "title", "document_type")
KNOWN_DOCUMENT_TYPES = frozenset({"reference", "glossary", "rules", "methodology", "research"})

_FIELD = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")


@dataclass(frozen=True)
class KnowledgeDocument:
    """One curated markdown document with its metadata."""

    document_id: str
    title: str
    document_type: str
    source: str
    text: str
    path: Path
    updated: str | None = None

    @property
    def content_hash(self) -> str:
        """Stable digest of the text, used to detect an index built from older content."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


def parse_front_matter(raw: str, path: Path) -> tuple[dict[str, str], str]:
    """Split a document into its front-matter mapping and its body."""
    lines = raw.lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != FRONT_MATTER_DELIMITER:
        raise CorpusValidationError(str(path), "missing '---' front-matter block")

    metadata: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONT_MATTER_DELIMITER:
            return metadata, "\n".join(lines[index + 1 :]).strip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _FIELD.match(line)
        if match is None:
            raise CorpusValidationError(str(path), f"unparsable front-matter line: {line!r}")
        metadata[match.group(1).lower()] = match.group(2).strip().strip("\"'")

    raise CorpusValidationError(str(path), "front-matter block was never closed")


def load_document(path: Path) -> KnowledgeDocument:
    """Read and validate one document."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusValidationError(str(path), f"could not be read: {exc}") from exc

    metadata, body = parse_front_matter(raw, path)

    missing = [field for field in REQUIRED_FIELDS if not metadata.get(field)]
    if missing:
        raise CorpusValidationError(str(path), f"missing required field(s): {', '.join(missing)}")
    if not body.strip():
        raise CorpusValidationError(str(path), "document body is empty")

    document_type = metadata["document_type"]
    if document_type not in KNOWN_DOCUMENT_TYPES:
        raise CorpusValidationError(
            str(path),
            f"unknown document_type {document_type!r}; "
            f"expected one of {', '.join(sorted(KNOWN_DOCUMENT_TYPES))}",
        )

    return KnowledgeDocument(
        document_id=metadata["document_id"],
        title=metadata["title"],
        document_type=document_type,
        source=metadata.get("source", "authored"),
        updated=metadata.get("updated") or None,
        text=body,
        path=path,
    )


def iter_document_paths(root: Path) -> Iterator[Path]:
    return iter(sorted(root.rglob("*.md")))


def load_corpus(root: Path) -> list[KnowledgeDocument]:
    """Load every document under ``root``, rejecting duplicate identifiers.

    A duplicate ``document_id`` is an error rather than a last-one-wins merge:
    two documents claiming the same id would make a citation ambiguous, and an
    ambiguous citation is worse than a missing one.
    """
    if not root.is_dir():
        raise CorpusValidationError(str(root), "corpus directory does not exist")

    documents: list[KnowledgeDocument] = []
    seen: dict[str, Path] = {}
    for path in iter_document_paths(root):
        document = load_document(path)
        previous = seen.get(document.document_id)
        if previous is not None:
            raise CorpusValidationError(
                str(path),
                f"document_id {document.document_id!r} is already used by {previous.name}",
            )
        seen[document.document_id] = path
        documents.append(document)

    if not documents:
        raise CorpusValidationError(str(root), "corpus contains no markdown documents")
    return documents


def corpus_fingerprint(documents: list[KnowledgeDocument]) -> str:
    """One digest over the whole corpus, so a stale index can be detected."""
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.document_id):
        digest.update(document.document_id.encode("utf-8"))
        digest.update(document.content_hash.encode("utf-8"))
    return digest.hexdigest()[:16]
