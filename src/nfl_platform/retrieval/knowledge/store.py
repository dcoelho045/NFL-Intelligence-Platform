"""Parquet-backed vector index with exact cosine search.

A deliberate non-choice of database. At this corpus size -- a few hundred chunks
-- a full scan is a few tens of milliseconds and returns exactly the top k, while
an approximate index would cost a build step, a dependency, and some recall to
avoid work that is not slow. The measurement, not the fashion, picks the design.

Storage is one Parquet file plus a JSON manifest. Float32 vectors at 1536
dimensions cost about 6 KB per chunk, so the whole index is on the order of a
megabyte and can be deleted and rebuilt for about a tenth of a cent.

Migrate to ``sqlite-vec`` or pgvector if the corpus passes roughly 5,000 chunks
or p95 search latency passes 100 ms. Until then this is the cheaper design.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nfl_platform.config.logging import get_logger
from nfl_platform.exceptions import KnowledgeIndexStaleError, KnowledgeIndexUnavailableError
from nfl_platform.retrieval.models import KnowledgeChunk, RetrievedChunk

logger = get_logger(__name__)

MANIFEST_SUFFIX = ".manifest.json"

_COLUMNS = (
    "chunk_id",
    "document_id",
    "title",
    "section",
    "text",
    "document_type",
    "source",
    "updated",
    "chunk_index",
    "content_hash",
    "embedding",
)


@dataclass(frozen=True)
class IndexManifest:
    """What the index was built from and with.

    Comparing vectors produced by two different models yields plausible
    nonsense rather than an error, so the model and width are recorded and
    checked on load. The corpus fingerprint turns "someone edited a document and
    forgot to rebuild" from an invisible problem into a warning.
    """

    model: str
    dimensions: int
    chunks: int
    documents: int
    corpus_fingerprint: str
    built_at: str
    embedding_tokens: int = 0
    estimated_cost_usd: float | None = None

    @property
    def signature(self) -> str:
        return f"{self.model}@{self.dimensions}d"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def build(
        cls,
        *,
        model: str,
        dimensions: int,
        chunks: int,
        documents: int,
        corpus_fingerprint: str,
        embedding_tokens: int = 0,
        estimated_cost_usd: float | None = None,
    ) -> IndexManifest:
        return cls(
            model=model,
            dimensions=dimensions,
            chunks=chunks,
            documents=documents,
            corpus_fingerprint=corpus_fingerprint,
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            embedding_tokens=embedding_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )


def manifest_path_for(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + MANIFEST_SUFFIX)


class VectorStore:
    """Reads and writes the chunk index."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._chunks: list[KnowledgeChunk] = []
        self._vectors: list[list[float]] = []
        self._manifest: IndexManifest | None = None
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def manifest(self) -> IndexManifest | None:
        return self._manifest

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def exists(self) -> bool:
        return self._path.is_file()

    def write(
        self,
        chunks: Sequence[KnowledgeChunk],
        vectors: Sequence[Sequence[float]],
        manifest: IndexManifest,
    ) -> Path:
        """Persist the index and its manifest."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Got {len(chunks)} chunks and {len(vectors)} vectors; they must match."
            )
        if not chunks:
            raise ValueError("Refusing to write an empty index.")

        frame = pl.DataFrame(
            {
                "chunk_id": [chunk.chunk_id for chunk in chunks],
                "document_id": [chunk.document_id for chunk in chunks],
                "title": [chunk.title for chunk in chunks],
                "section": [chunk.section for chunk in chunks],
                "text": [chunk.text for chunk in chunks],
                "document_type": [chunk.document_type for chunk in chunks],
                "source": [chunk.source for chunk in chunks],
                "updated": [chunk.updated for chunk in chunks],
                "chunk_index": [chunk.chunk_index for chunk in chunks],
                "content_hash": [chunk.content_hash for chunk in chunks],
                # Float32 halves the file for a precision the scores never notice.
                "embedding": [list(vector) for vector in vectors],
            },
            schema_overrides={"embedding": pl.List(pl.Float32), "updated": pl.String},
        )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(self._path, compression="zstd")
        manifest_path_for(self._path).write_text(
            json.dumps(manifest.as_dict(), indent=2) + "\n", encoding="utf-8"
        )

        self._chunks = list(chunks)
        self._vectors = [list(vector) for vector in vectors]
        self._manifest = manifest
        self._loaded = True
        return self._path

    def load(self, *, expected_signature: str | None = None) -> None:
        """Read the index into memory, refusing a mismatched or unusable one."""
        if self._loaded and expected_signature is None:
            return
        if not self._path.is_file():
            raise KnowledgeIndexUnavailableError(
                str(self._path),
                "the index file does not exist; build it with "
                "'python scripts/build_knowledge_index.py'",
            )

        try:
            frame = pl.read_parquet(self._path)
        except Exception as exc:
            raise KnowledgeIndexUnavailableError(str(self._path), f"unreadable: {exc}") from exc

        missing = [column for column in _COLUMNS if column not in frame.columns]
        if missing:
            raise KnowledgeIndexUnavailableError(
                str(self._path), f"missing column(s): {', '.join(missing)}"
            )
        if frame.height == 0:
            raise KnowledgeIndexUnavailableError(str(self._path), "the index is empty")

        manifest = self._read_manifest()
        if expected_signature and manifest and manifest.signature != expected_signature:
            raise KnowledgeIndexStaleError(str(self._path), expected_signature, manifest.signature)

        chunks: list[KnowledgeChunk] = []
        vectors: list[list[float]] = []
        for row in frame.iter_rows(named=True):
            chunks.append(
                KnowledgeChunk(
                    document_id=row["document_id"],
                    chunk_id=row["chunk_id"],
                    title=row["title"],
                    section=row["section"] or "",
                    text=row["text"],
                    document_type=row["document_type"],
                    source=row["source"],
                    updated=row["updated"],
                    chunk_index=int(row["chunk_index"]),
                    content_hash=row["content_hash"] or "",
                )
            )
            vectors.append([float(value) for value in row["embedding"]])

        self._chunks = chunks
        self._vectors = vectors
        self._manifest = manifest
        self._loaded = True

    def search(
        self, query_vector: Sequence[float], *, top_k: int, min_score: float = 0.0
    ) -> list[RetrievedChunk]:
        """Exact cosine top-k.

        Vectors are unit length on write, so similarity is a dot product.
        """
        self.load()
        if not self._vectors:
            return []

        width = len(self._vectors[0])
        if len(query_vector) != width:
            raise KnowledgeIndexStaleError(
                str(self._path),
                f"query width {len(query_vector)}",
                f"index width {width}",
            )

        scored = [
            (sum(a * b for a, b in zip(query_vector, vector, strict=True)), index)
            for index, vector in enumerate(self._vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            RetrievedChunk(chunk=self._chunks[index], score=score)
            for score, index in scored[:top_k]
            if score >= min_score
        ]

    def _read_manifest(self) -> IndexManifest | None:
        path = manifest_path_for(self._path)
        if not path.is_file():
            logger.warning("Knowledge index has no manifest", extra={"path": str(path)})
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return IndexManifest(**payload)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Knowledge index manifest is unreadable", extra={"error": str(exc)})
            return None
