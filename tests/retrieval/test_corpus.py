"""Corpus loading, front-matter parsing and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_platform.exceptions import CorpusValidationError
from nfl_platform.retrieval.knowledge.chunking import chunk_document
from nfl_platform.retrieval.knowledge.corpus import (
    KnowledgeDocument,
    corpus_fingerprint,
    load_corpus,
    load_document,
    parse_front_matter,
)
from support import KNOWLEDGE_FIXTURE_DIR

REAL_CORPUS = Path(__file__).resolve().parents[2] / "data" / "knowledge"

HEADER = "---\ndocument_id: d\ntitle: T\ndocument_type: reference\n---\n\n"


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestFrontMatter:
    def test_parses_fields_and_body(self, tmp_path: Path):
        metadata, body = parse_front_matter(
            "---\ndocument_id: a\ntitle: A Title\n---\n\n## Section\n\nBody.",
            tmp_path / "a.md",
        )
        assert metadata["document_id"] == "a"
        assert metadata["title"] == "A Title"
        assert body.startswith("## Section")

    def test_strips_surrounding_quotes(self, tmp_path: Path):
        metadata, _ = parse_front_matter(
            "---\ndocument_id: \"a\"\ntitle: 'B'\n---\nBody.", tmp_path / "a.md"
        )
        assert metadata["document_id"] == "a"
        assert metadata["title"] == "B"

    def test_missing_block_is_rejected(self, tmp_path: Path):
        with pytest.raises(CorpusValidationError, match="front-matter"):
            parse_front_matter("# Just a heading", tmp_path / "a.md")

    def test_unclosed_block_is_rejected(self, tmp_path: Path):
        with pytest.raises(CorpusValidationError, match="never closed"):
            parse_front_matter("---\ndocument_id: a\ntitle: T\n", tmp_path / "a.md")

    def test_unparsable_line_is_rejected(self, tmp_path: Path):
        with pytest.raises(CorpusValidationError, match="unparsable"):
            parse_front_matter("---\nnot a field\n---\nBody", tmp_path / "a.md")


class TestDocumentValidation:
    def test_loads_a_valid_document(self, tmp_path: Path):
        path = write(
            tmp_path / "doc.md",
            "---\ndocument_id: doc\ntitle: Doc\ndocument_type: reference\n"
            "source: authored\nupdated: 2026-01-01\n---\n\n## S\n\nBody text.",
        )
        document = load_document(path)
        assert document.document_id == "doc"
        assert document.document_type == "reference"
        assert document.updated == "2026-01-01"

    @pytest.mark.parametrize("field", ["document_id", "title", "document_type"])
    def test_missing_required_field_is_rejected(self, tmp_path: Path, field: str):
        fields = {"document_id": "d", "title": "T", "document_type": "reference"}
        del fields[field]
        block = "\n".join(f"{key}: {value}" for key, value in fields.items())
        path = write(tmp_path / "doc.md", f"---\n{block}\n---\n\nBody.")
        with pytest.raises(CorpusValidationError, match="missing required"):
            load_document(path)

    def test_unknown_document_type_is_rejected(self, tmp_path: Path):
        path = write(
            tmp_path / "doc.md",
            "---\ndocument_id: d\ntitle: T\ndocument_type: nonsense\n---\n\nBody.",
        )
        with pytest.raises(CorpusValidationError, match="unknown document_type"):
            load_document(path)

    def test_empty_body_is_rejected(self, tmp_path: Path):
        path = write(
            tmp_path / "doc.md",
            "---\ndocument_id: d\ntitle: T\ndocument_type: reference\n---\n\n   ",
        )
        with pytest.raises(CorpusValidationError, match="body is empty"):
            load_document(path)

    def test_source_defaults_to_authored(self, tmp_path: Path):
        path = write(
            tmp_path / "doc.md",
            "---\ndocument_id: d\ntitle: T\ndocument_type: reference\n---\n\nBody.",
        )
        assert load_document(path).source == "authored"


class TestCorpusLoading:
    def test_loads_every_document_in_the_fixture_corpus(self):
        documents = load_corpus(KNOWLEDGE_FIXTURE_DIR)
        assert {document.document_id for document in documents} == {
            "test-scoring",
            "test-usage",
            "test-conventions",
        }

    def test_duplicate_document_ids_are_rejected(self, tmp_path: Path):
        """A duplicate id would make a citation ambiguous, which is worse than missing."""
        block = "---\ndocument_id: same\ntitle: T\ndocument_type: reference\n---\n\nBody."
        write(tmp_path / "a.md", block)
        write(tmp_path / "b.md", block)
        with pytest.raises(CorpusValidationError, match="already used"):
            load_corpus(tmp_path)

    def test_missing_directory_is_rejected(self, tmp_path: Path):
        with pytest.raises(CorpusValidationError, match="does not exist"):
            load_corpus(tmp_path / "nope")

    def test_empty_directory_is_rejected(self, tmp_path: Path):
        with pytest.raises(CorpusValidationError, match="no markdown"):
            load_corpus(tmp_path)

    def test_fingerprint_is_stable_for_identical_content(self):
        first = corpus_fingerprint(load_corpus(KNOWLEDGE_FIXTURE_DIR))
        second = corpus_fingerprint(load_corpus(KNOWLEDGE_FIXTURE_DIR))
        assert first == second

    def test_fingerprint_changes_when_content_changes(self, tmp_path: Path):
        path = write(
            tmp_path / "a.md",
            "---\ndocument_id: a\ntitle: T\ndocument_type: reference\n---\n\nOriginal.",
        )
        before = corpus_fingerprint(load_corpus(tmp_path))
        write(
            path,
            "---\ndocument_id: a\ntitle: T\ndocument_type: reference\n---\n\nEdited.",
        )
        assert corpus_fingerprint(load_corpus(tmp_path)) != before


class TestRealCorpus:
    """The shipped corpus must stay valid and stay small."""

    def test_the_shipped_corpus_loads(self):
        assert len(load_corpus(REAL_CORPUS)) >= 10

    def test_every_shipped_document_records_when_it_was_updated(self):
        """Curated knowledge presented without a date looks current. It is not."""
        for document in load_corpus(REAL_CORPUS):
            assert document.updated, document.document_id

    def test_the_corpus_stays_small(self):
        """A large local archive is what this milestone explicitly avoids."""
        total = sum(len(document.text) for document in load_corpus(REAL_CORPUS))
        assert total < 250_000, f"corpus grew to {total} characters"

    def test_the_corpus_is_self_authored(self):
        for document in load_corpus(REAL_CORPUS):
            assert document.source in {"authored", "public-domain"}, document.document_id


class TestChunking:
    @pytest.fixture
    def document(self) -> KnowledgeDocument:
        return load_document(KNOWLEDGE_FIXTURE_DIR / "test-scoring.md")

    def test_headings_define_chunk_boundaries(self, document: KnowledgeDocument):
        chunks = chunk_document(document)
        assert {chunk.section for chunk in chunks} == {"PPR scoring", "Touchdown values"}

    def test_metadata_is_preserved_on_every_chunk(self, document: KnowledgeDocument):
        for chunk in chunk_document(document):
            assert chunk.document_id == "test-scoring"
            assert chunk.title == "Test Scoring Reference"
            assert chunk.document_type == "reference"
            assert chunk.source == "authored"
            assert chunk.updated == "2024-01-01"

    def test_chunk_ids_are_unique_and_stable(self, document: KnowledgeDocument):
        first = [chunk.chunk_id for chunk in chunk_document(document)]
        second = [chunk.chunk_id for chunk in chunk_document(document)]
        assert first == second
        assert len(set(first)) == len(first)

    def test_the_heading_is_embedded_with_the_body(self, document: KnowledgeDocument):
        """A body often omits the term it defines; without the heading it never matches."""
        ppr = next(chunk for chunk in chunk_document(document) if chunk.section == "PPR scoring")
        assert "PPR scoring" in ppr.text
        assert "Test Scoring Reference" in ppr.text

    def test_identical_sections_are_not_indexed_twice(self, tmp_path: Path):
        path = write(
            tmp_path / "dupe.md",
            "---\ndocument_id: d\ntitle: T\ndocument_type: reference\n---\n"
            "\n## A\n\nSame body.\n\n## A\n\nSame body.\n",
        )
        assert len(chunk_document(load_document(path))) == 1

    def test_oversized_sections_are_split(self, tmp_path: Path):
        paragraphs = "\n\n".join(f"Paragraph number {index} of text." for index in range(60))
        path = write(tmp_path / "long.md", f"{HEADER}## Long\n\n{paragraphs}")
        chunks = chunk_document(load_document(path), chunk_tokens=50, chunk_overlap=10)
        assert len(chunks) > 1
        assert all(chunk.section == "Long" for chunk in chunks)

    def test_split_chunks_stay_near_the_target_size(self, tmp_path: Path):
        paragraphs = "\n\n".join(f"Paragraph number {index} of text." for index in range(60))
        path = write(tmp_path / "long.md", f"{HEADER}## Long\n\n{paragraphs}")
        chunks = chunk_document(load_document(path), chunk_tokens=50, chunk_overlap=10)
        # Four characters per token, plus the prepended title and heading.
        assert all(len(chunk.text) < 50 * 4 * 2 for chunk in chunks)

    def test_content_hashes_are_recorded(self, document: KnowledgeDocument):
        assert all(chunk.content_hash for chunk in chunk_document(document))

    def test_a_document_with_no_headings_still_chunks(self, tmp_path: Path):
        path = write(tmp_path / "flat.md", f"{HEADER}Just prose, no headings.")
        chunks = chunk_document(load_document(path))
        assert len(chunks) == 1
        assert chunks[0].section == ""
