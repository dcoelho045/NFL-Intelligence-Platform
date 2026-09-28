"""Build the local knowledge index.

Reads the curated corpus, chunks it, embeds the chunks, and writes a Parquet
index plus a manifest recording what it was built from and with.

The index is not committed. It is a derived artifact costing roughly a tenth of
a cent to rebuild, and a binary that churns on every corpus edit is worse than a
build step.

    python scripts/build_knowledge_index.py
    python scripts/build_knowledge_index.py --dry-run      # cost estimate only
    python scripts/build_knowledge_index.py --provider fake  # offline, no key
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_platform.config.settings import Settings, get_settings
from nfl_platform.retrieval.knowledge.chunking import chunk_document
from nfl_platform.retrieval.knowledge.corpus import (
    corpus_fingerprint,
    load_corpus,
)
from nfl_platform.retrieval.knowledge.embeddings import (
    build_embedding_client,
)
from nfl_platform.retrieval.knowledge.store import IndexManifest, VectorStore
from nfl_platform.retrieval.models import KnowledgeChunk

# Roughly four characters per token for English prose. Only used for the
# pre-flight estimate; the manifest records the provider's real token count.
CHARS_PER_TOKEN = 4


def collect_chunks(settings: Settings) -> tuple[list[KnowledgeChunk], int, str]:
    root = settings.resolved_knowledge_dir()
    documents = load_corpus(root)
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_tokens=settings.rag_chunk_tokens,
                chunk_overlap=settings.rag_chunk_overlap,
            )
        )
    return chunks, len(documents), corpus_fingerprint(documents)


def estimate_cost(model: str, chunks: list[KnowledgeChunk]) -> tuple[int, float | None]:
    from nfl_platform.retrieval.providers.openai_embeddings import estimate_embedding_cost_usd

    tokens = sum(len(chunk.text) for chunk in chunks) // CHARS_PER_TOKEN
    return tokens, estimate_embedding_cost_usd(model, tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="Override the corpus directory.")
    parser.add_argument("--output", type=Path, help="Override the index path.")
    parser.add_argument("--provider", choices=("openai", "fake"), help="Embedding provider.")
    parser.add_argument("--model", help="Embedding model.")
    parser.add_argument("--dimensions", type=int, help="Truncated embedding width.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report chunk count and estimated cost without embedding anything.",
    )
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.corpus:
        overrides["knowledge_dir"] = args.corpus
    if args.output:
        overrides["knowledge_index_path"] = args.output
    if args.provider:
        overrides["embedding_provider"] = args.provider
    if args.model:
        overrides["embedding_model"] = args.model
    if args.dimensions:
        overrides["embedding_dimensions"] = args.dimensions

    settings = get_settings().model_copy(update=overrides)

    chunks, document_count, fingerprint = collect_chunks(settings)
    characters = sum(len(chunk.text) for chunk in chunks)
    print(f"Corpus:    {settings.resolved_knowledge_dir()}")
    print(f"Documents: {document_count}")
    print(f"Chunks:    {len(chunks)}")
    print(f"Text:      {characters:,} characters")

    if args.dry_run:
        tokens, cost = estimate_cost(settings.embedding_model, chunks)
        print(f"Estimated: ~{tokens:,} tokens")
        print(f"Cost:      ~${cost:.5f}" if cost is not None else "Cost:      unpriced model")
        return 0

    embedder = build_embedding_client(settings)
    print(f"Embedding with {embedder.model} at {embedder.dimensions} dimensions...")
    result = embedder.embed([chunk.text for chunk in chunks])

    cost: float | None = None
    if settings.embedding_provider == "openai":
        from nfl_platform.retrieval.providers.openai_embeddings import (
            estimate_embedding_cost_usd,
        )

        cost = estimate_embedding_cost_usd(result.model, result.total_tokens)

    manifest = IndexManifest.build(
        model=embedder.model,
        dimensions=embedder.dimensions,
        chunks=len(chunks),
        documents=document_count,
        corpus_fingerprint=fingerprint,
        embedding_tokens=result.total_tokens,
        estimated_cost_usd=None if cost is None else round(cost, 6),
    )

    store = VectorStore(settings.resolved_knowledge_index_path())
    path = store.write(chunks, result.vectors, manifest)
    size_kb = path.stat().st_size / 1024

    print(f"Wrote:     {path} ({size_kb:.0f} KB)")
    print(f"Tokens:    {result.total_tokens:,}")
    if cost is not None:
        print(f"Cost:      ${cost:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
