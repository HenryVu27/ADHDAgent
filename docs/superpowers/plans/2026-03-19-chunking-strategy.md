# Chunking Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two configurable chunking strategies (recursive+contextual, semantic) to the RAG pipeline, with parent document deduplication and strategy comparison support.

**Architecture:** A new `app/rag/chunker.py` module implements a `ChunkingStrategy` protocol with two implementations. `KnowledgeStore` delegates chunk creation to the configured strategy. Each strategy writes to its own Qdrant collection. The retriever deduplicates chunks by parent document after reranking, preserving the `full_doc` contract for the agent layer.

**Tech Stack:** Python 3.12, Qdrant, Gemini embeddings + Flash LLM (contextual headers), numpy

**Spec:** `docs/superpowers/specs/2026-03-19-chunking-strategy-design.md`

**Python env:** Always use `./adhd312/bin/python` for running tests and `./adhd312/bin/pip` for installing packages. System Python is 3.14 and has compatibility issues.

**Testing policy:** Never run integration tests that require `GEMINI_API_KEY`. All unit tests use mocks.

---

### Task 1: Add configuration settings

**Files:**
- Modify: `app/config.py:17-29` (add after existing RAG settings)

- [ ] **Step 1: Write failing test**

Create `tests/test_chunker.py` with a test that imports the new config values:

```python
# Tests for RAG chunking strategies

from app.config import settings


def test_chunking_config_defaults():
    assert settings.RAG_CHUNKING_STRATEGY == "none"
    assert settings.RAG_CHUNK_SIZE_TOKENS == 512
    assert settings.RAG_CHUNK_OVERLAP_TOKENS == 64
    assert settings.RAG_SEMANTIC_SIMILARITY_THRESHOLD == 0.75
    assert settings.RAG_CONTEXTUAL_HEADERS is True
    assert settings.RAG_PARENT_DEDUP is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_chunking_config_defaults -v`
Expected: FAIL with `AttributeError` — settings don't have these fields yet.

- [ ] **Step 3: Add settings to config.py**

In `app/config.py`, add after line 29 (`RAG_OUTCOME_JACCARD_THRESHOLD`):

```python
    # Chunking
    RAG_CHUNKING_STRATEGY: str = "none"  # "recursive_contextual" | "semantic" | "none"
    RAG_CHUNK_SIZE_TOKENS: int = 512
    RAG_CHUNK_OVERLAP_TOKENS: int = 64
    RAG_SEMANTIC_SIMILARITY_THRESHOLD: float = 0.75
    RAG_CONTEXTUAL_HEADERS: bool = True
    RAG_PARENT_DEDUP: bool = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_chunking_config_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_chunker.py
git commit -m "Add chunking strategy configuration settings"
```

---

### Task 2: Chunk dataclass and ChunkingStrategy protocol

**Files:**
- Create: `app/rag/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write failing test for Chunk dataclass**

Append to `tests/test_chunker.py`:

```python
from app.rag.chunker import Chunk


def test_chunk_dataclass():
    chunk = Chunk(
        chunk_id="doc1__step_0",
        parent_document_id="doc1",
        text="[Doc1] Step one text",
        raw_text="Step one text",
        context_header="[Doc1] ",
        chunk_type="step",
        chunk_index=0,
        metadata={"tags": ["homework"], "age_range": ["school_age"]},
    )
    assert chunk.chunk_id == "doc1__step_0"
    assert chunk.parent_document_id == "doc1"
    assert chunk.chunk_type == "step"
    assert chunk.chunk_index == 0
    assert chunk.metadata["tags"] == ["homework"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_chunk_dataclass -v`
Expected: FAIL — `app.rag.chunker` does not exist.

- [ ] **Step 3: Create chunker.py with Chunk dataclass and protocol**

Create `app/rag/chunker.py`:

```python
# Chunking strategies for the RAG knowledge base.
# Supports structured JSON docs (steps/key_points), sectioned text, and generic text.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str            # "{doc_id}__{type}_{index}"
    parent_document_id: str
    text: str                # chunk content with context header (if applicable)
    raw_text: str            # chunk content without header
    context_header: str      # empty for strategies that don't use headers
    chunk_type: str          # "description" | "step" | "key_point" | "text_segment"
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class ChunkingStrategy(Protocol):
    def chunk(self, document: dict) -> list[Chunk]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_chunk_dataclass -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/chunker.py tests/test_chunker.py
git commit -m "Add Chunk dataclass and ChunkingStrategy protocol"
```

---

### Task 3: NoneChunker (baseline — current behavior)

**Files:**
- Modify: `app/rag/chunker.py`
- Test: `tests/test_chunker.py`

This preserves the exact current one-chunk-per-doc behavior so we can verify zero regression when `KnowledgeStore` delegates to it.

- [ ] **Step 1: Write failing test**

Append to `tests/test_chunker.py`:

```python
from app.rag.chunker import Chunk, NoneChunker


def test_none_chunker_strategy_doc():
    doc = {
        "id": "test_strategy",
        "name": "Test Strategy",
        "description": "A test strategy for ADHD.",
        "document_type": "strategy",
        "tags": ["homework"],
        "age_range": ["school_age"],
        "evidence_level": "strong",
        "source": "Test Source",
        "citations": [],
        "steps": ["Step one.", "Step two.", "Step three."],
    }
    chunker = NoneChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "test_strategy__full_0"
    assert chunk.parent_document_id == "test_strategy"
    assert chunk.chunk_type == "full"
    assert "Test Strategy" in chunk.text
    assert "Step one." in chunk.text
    assert "Step two." in chunk.text
    assert chunk.context_header == ""
    assert chunk.metadata["tags"] == ["homework"]


def test_none_chunker_key_points_doc():
    doc = {
        "id": "test_fact",
        "name": "Test Fact",
        "description": "A fact about ADHD.",
        "document_type": "fact",
        "tags": ["adhd_basics"],
        "age_range": ["all"],
        "evidence_level": "moderate",
        "source": "Test",
        "citations": [],
        "key_points": ["Point A.", "Point B."],
    }
    chunker = NoneChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert "Point A." in chunks[0].text
    assert "Point B." in chunks[0].text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_none_chunker_strategy_doc tests/test_chunker.py::test_none_chunker_key_points_doc -v`
Expected: FAIL — `NoneChunker` not defined.

- [ ] **Step 3: Implement NoneChunker**

Add to `app/rag/chunker.py`:

```python
def _extract_metadata(doc: dict) -> dict:
    """Extract filtering metadata from a document dict."""
    return {
        "document_name": doc.get("name", ""),
        "tags": doc.get("tags", []),
        "source": doc.get("source", ""),
        "evidence_level": doc.get("evidence_level", ""),
        "document_type": doc.get("document_type", ""),
        "age_range": doc.get("age_range", []),
        "citations": doc.get("citations", []),
    }


class NoneChunker:
    """One chunk per document — replicates the original KnowledgeStore behavior."""

    def chunk(self, document: dict) -> list[Chunk]:
        doc_id = document.get("id", "")
        text_parts = [document.get("name", ""), document.get("description", "")]

        steps = document.get("steps", [])
        if steps:
            text_parts.append("Steps: " + " | ".join(steps))

        key_points = document.get("key_points", [])
        if key_points:
            text_parts.append("Key points: " + " | ".join(key_points))

        return [Chunk(
            chunk_id=f"{doc_id}__full_0",
            parent_document_id=doc_id,
            text=" ".join(text_parts),
            raw_text=" ".join(text_parts),
            context_header="",
            chunk_type="full",
            chunk_index=0,
            metadata=_extract_metadata(document),
        )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/chunker.py tests/test_chunker.py
git commit -m "Add NoneChunker baseline strategy"
```

---

### Task 4: RecursiveContextualChunker — structural splitting (tier 1)

**Files:**
- Modify: `app/rag/chunker.py`
- Test: `tests/test_chunker.py`

This implements splitting structured docs on steps/key_points. Contextual headers and recursive fallback are separate tasks.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_chunker.py`:

```python
from app.rag.chunker import Chunk, NoneChunker, RecursiveContextualChunker


def test_recursive_chunker_structured_steps():
    doc = {
        "id": "strat1",
        "name": "Test Strategy",
        "description": "Strategy description here.",
        "document_type": "strategy",
        "tags": ["homework"],
        "age_range": ["school_age"],
        "evidence_level": "strong",
        "source": "Test",
        "citations": [],
        "steps": ["Step one.", "Step two.", "Step three."],
    }
    chunker = RecursiveContextualChunker(contextual_headers=False)
    chunks = chunker.chunk(doc)
    # description + 3 steps = 4 chunks
    assert len(chunks) == 4
    # First chunk is description
    assert chunks[0].chunk_type == "description"
    assert chunks[0].chunk_id == "strat1__description_0"
    assert "Strategy description here." in chunks[0].text
    # Remaining are steps
    for i, c in enumerate(chunks[1:], start=1):
        assert c.chunk_type == "step"
        assert c.chunk_id == f"strat1__step_{i}"
        assert c.parent_document_id == "strat1"
        assert c.metadata["tags"] == ["homework"]
    assert "Step one." in chunks[1].text
    assert "Step three." in chunks[3].text
    # Document name is prefixed for embedding context
    assert "[Test Strategy]" in chunks[1].text


def test_recursive_chunker_structured_key_points():
    doc = {
        "id": "fact1",
        "name": "ADHD Fact",
        "description": "Facts about ADHD.",
        "document_type": "fact",
        "tags": ["adhd_basics"],
        "age_range": ["all"],
        "evidence_level": "moderate",
        "source": "Test",
        "citations": [],
        "key_points": ["Point A.", "Point B."],
    }
    chunker = RecursiveContextualChunker(contextual_headers=False)
    chunks = chunker.chunk(doc)
    # description + 2 key_points = 3 chunks
    assert len(chunks) == 3
    assert chunks[0].chunk_type == "description"
    assert chunks[1].chunk_type == "key_point"
    assert chunks[2].chunk_type == "key_point"
    assert "Point A." in chunks[1].text


def test_recursive_chunker_description_only():
    doc = {
        "id": "desc_only",
        "name": "Desc Doc",
        "description": "Just a description, no steps or key points.",
        "document_type": "guidance",
        "tags": [],
        "age_range": [],
        "evidence_level": "",
        "source": "",
        "citations": [],
    }
    chunker = RecursiveContextualChunker(contextual_headers=False)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "description"


def test_recursive_chunker_empty_doc():
    doc = {"id": "empty", "name": "", "description": ""}
    chunker = RecursiveContextualChunker(contextual_headers=False)
    chunks = chunker.chunk(doc)
    # Should still produce at least one chunk (description, even if empty)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "description"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_recursive_chunker_structured_steps tests/test_chunker.py::test_recursive_chunker_structured_key_points tests/test_chunker.py::test_recursive_chunker_description_only tests/test_chunker.py::test_recursive_chunker_empty_doc -v`
Expected: FAIL — `RecursiveContextualChunker` not defined.

- [ ] **Step 3: Implement RecursiveContextualChunker structural splitting**

Add to `app/rag/chunker.py`:

```python
class RecursiveContextualChunker:
    """Structural + recursive splitting with optional contextual headers.

    Tier 1: Structured docs (steps/key_points) -> one chunk per item.
    Tier 2: Sectioned text (headings) -> split on sections. (added in Task 5)
    Tier 3: Generic fallback -> recursive character split. (added in Task 5)
    """

    def __init__(
        self,
        chunk_size: int = 2048,
        chunk_overlap: int = 256,
        contextual_headers: bool = True,
        gemini_client=None,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._contextual_headers = contextual_headers
        self._gemini = gemini_client

    def chunk(self, document: dict) -> list[Chunk]:
        doc_id = document.get("id", "")
        doc_name = document.get("name", "")
        meta = _extract_metadata(document)

        steps = document.get("steps", [])
        key_points = document.get("key_points", [])

        if steps or key_points:
            return self._chunk_structured(document, doc_id, doc_name, meta)

        # Tier 2/3: text-based splitting (Task 5)
        # For now, fall back to single description chunk
        return self._chunk_description_only(document, doc_id, doc_name, meta)

    def _chunk_structured(
        self, doc: dict, doc_id: str, doc_name: str, meta: dict
    ) -> list[Chunk]:
        chunks = []

        # Description chunk
        description = doc.get("description", "")
        desc_text = f"[{doc_name}] {description}" if doc_name else description
        chunks.append(Chunk(
            chunk_id=f"{doc_id}__description_0",
            parent_document_id=doc_id,
            text=desc_text,
            raw_text=desc_text,
            context_header="",
            chunk_type="description",
            chunk_index=0,
            metadata=meta,
        ))

        # Step chunks
        for i, step in enumerate(doc.get("steps", []), start=1):
            step_text = f"[{doc_name}] {step}" if doc_name else step
            chunks.append(Chunk(
                chunk_id=f"{doc_id}__step_{i}",
                parent_document_id=doc_id,
                text=step_text,
                raw_text=step_text,
                context_header="",
                chunk_type="step",
                chunk_index=i,
                metadata=meta,
            ))

        # Key point chunks
        for i, kp in enumerate(doc.get("key_points", []), start=1):
            kp_text = f"[{doc_name}] {kp}" if doc_name else kp
            chunks.append(Chunk(
                chunk_id=f"{doc_id}__key_point_{i}",
                parent_document_id=doc_id,
                text=kp_text,
                raw_text=kp_text,
                context_header="",
                chunk_type="key_point",
                chunk_index=i,
                metadata=meta,
            ))

        return chunks

    def _chunk_description_only(
        self, doc: dict, doc_id: str, doc_name: str, meta: dict
    ) -> list[Chunk]:
        description = doc.get("description", "")
        desc_text = f"[{doc_name}] {description}" if doc_name else description
        return [Chunk(
            chunk_id=f"{doc_id}__description_0",
            parent_document_id=doc_id,
            text=desc_text,
            raw_text=desc_text,
            context_header="",
            chunk_type="description",
            chunk_index=0,
            metadata=meta,
        )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/chunker.py tests/test_chunker.py
git commit -m "Add RecursiveContextualChunker with structural splitting"
```

---

### Task 5: RecursiveContextualChunker — sectioned and recursive fallback (tiers 2-3)

**Files:**
- Modify: `app/rag/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_chunker.py`:

```python
def test_recursive_chunker_sectioned_text():
    doc = {
        "id": "sectioned1",
        "name": "Sectioned Guide",
        "description": "## Overview\nThis is the overview.\n\n## Details\nHere are the details of the guide.\n\n## Summary\nFinal summary.",
        "document_type": "guidance",
        "tags": ["guide"],
        "age_range": [],
        "evidence_level": "",
        "source": "",
        "citations": [],
    }
    chunker = RecursiveContextualChunker(contextual_headers=False)
    chunks = chunker.chunk(doc)
    # Should split on heading boundaries: 3 sections
    assert len(chunks) == 3
    assert all(c.chunk_type == "text_segment" for c in chunks)
    assert "Overview" in chunks[0].text
    assert "Details" in chunks[1].text
    assert "Summary" in chunks[2].text


def test_recursive_chunker_long_text_fallback():
    # Generate text longer than chunk_size to force recursive splitting
    long_text = ". ".join([f"Sentence number {i} with some extra words" for i in range(200)])
    doc = {
        "id": "longdoc",
        "name": "Long Document",
        "description": long_text,
        "document_type": "guidance",
        "tags": [],
        "age_range": [],
        "evidence_level": "",
        "source": "",
        "citations": [],
    }
    # Use small chunk size to force splitting
    chunker = RecursiveContextualChunker(chunk_size=200, chunk_overlap=50, contextual_headers=False)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert all(c.chunk_type == "text_segment" for c in chunks)
    assert all(c.parent_document_id == "longdoc" for c in chunks)
    # Each chunk should be within size limit (with some tolerance for boundary)
    for c in chunks:
        assert len(c.text) <= 300  # chunk_size + tolerance


def test_recursive_chunker_overlap():
    # Verify overlap between adjacent chunks
    sentences = [f"Unique sentence {i} about topic {i}." for i in range(50)]
    long_text = " ".join(sentences)
    doc = {
        "id": "overlap_test",
        "name": "Overlap Test",
        "description": long_text,
        "document_type": "guidance",
        "tags": [],
        "age_range": [],
        "evidence_level": "",
        "source": "",
        "citations": [],
    }
    chunker = RecursiveContextualChunker(chunk_size=200, chunk_overlap=80, contextual_headers=False)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 2
    # Check that adjacent chunks share some text (overlap)
    for i in range(len(chunks) - 1):
        # The end of chunk i should overlap with the start of chunk i+1
        # We can check by looking for shared words
        words_current = set(chunks[i].raw_text.split()[-10:])
        words_next = set(chunks[i + 1].raw_text.split()[:10])
        assert words_current & words_next, f"No overlap between chunks {i} and {i+1}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_recursive_chunker_sectioned_text tests/test_chunker.py::test_recursive_chunker_long_text_fallback tests/test_chunker.py::test_recursive_chunker_overlap -v`
Expected: FAIL — sectioned/recursive splitting not yet implemented.

- [ ] **Step 3: Implement sectioned and recursive splitting**

Update `RecursiveContextualChunker` in `app/rag/chunker.py`. Modify the `chunk` method's fallback path, and add `_chunk_sectioned` and `_chunk_recursive` methods:

```python
import re  # add to imports if not present

# In RecursiveContextualChunker.chunk(), replace the tier 2/3 comment block:
    def chunk(self, document: dict) -> list[Chunk]:
        doc_id = document.get("id", "")
        doc_name = document.get("name", "")
        meta = _extract_metadata(document)

        steps = document.get("steps", [])
        key_points = document.get("key_points", [])

        if steps or key_points:
            return self._chunk_structured(document, doc_id, doc_name, meta)

        description = document.get("description", "")
        if not description:
            return self._chunk_description_only(document, doc_id, doc_name, meta)

        # Tier 2: Sectioned text (headings)
        if re.search(r"^#{1,4}\s+", description, re.MULTILINE):
            return self._chunk_sectioned(description, doc_id, doc_name, meta)

        # Tier 3: Recursive split if text exceeds chunk size
        if len(description) > self._chunk_size:
            return self._chunk_recursive(description, doc_id, doc_name, meta)

        return self._chunk_description_only(document, doc_id, doc_name, meta)

# Add these methods to RecursiveContextualChunker:

    def _chunk_sectioned(
        self, text: str, doc_id: str, doc_name: str, meta: dict
    ) -> list[Chunk]:
        """Split text on markdown headings. Recurse if any section exceeds chunk_size."""
        sections = re.split(r"(?=^#{1,4}\s+)", text, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]

        chunks = []
        for i, section in enumerate(sections):
            if len(section) > self._chunk_size:
                # Recurse on oversized sections
                sub_chunks = self._split_recursive(section)
                for j, sub_text in enumerate(sub_chunks):
                    prefixed = f"[{doc_name}] {sub_text}" if doc_name else sub_text
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}__text_segment_{len(chunks)}",
                        parent_document_id=doc_id,
                        text=prefixed,
                        raw_text=sub_text,
                        context_header="",
                        chunk_type="text_segment",
                        chunk_index=len(chunks),
                        metadata=meta,
                    ))
            else:
                prefixed = f"[{doc_name}] {section}" if doc_name else section
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}__text_segment_{len(chunks)}",
                    parent_document_id=doc_id,
                    text=prefixed,
                    raw_text=section,
                    context_header="",
                    chunk_type="text_segment",
                    chunk_index=len(chunks),
                    metadata=meta,
                ))
        return chunks

    def _chunk_recursive(
        self, text: str, doc_id: str, doc_name: str, meta: dict
    ) -> list[Chunk]:
        """Generic recursive splitting for unstructured text."""
        pieces = self._split_recursive(text)
        chunks = []
        for i, piece in enumerate(pieces):
            prefixed = f"[{doc_name}] {piece}" if doc_name else piece
            chunks.append(Chunk(
                chunk_id=f"{doc_id}__text_segment_{i}",
                parent_document_id=doc_id,
                text=prefixed,
                raw_text=piece,
                context_header="",
                chunk_type="text_segment",
                chunk_index=i,
                metadata=meta,
            ))
        return chunks

    def _split_recursive(self, text: str) -> list[str]:
        """Recursively split text on paragraph -> newline -> sentence -> space boundaries.

        Returns non-overlapping pieces, then applies overlap in a second pass.
        """
        separators = ["\n\n", "\n", ". ", " "]
        raw_pieces = self._split_by_separators(text, separators, self._chunk_size)

        if self._chunk_overlap <= 0 or len(raw_pieces) <= 1:
            return raw_pieces

        # Apply overlap: append tail of previous piece to start of next
        overlapped = [raw_pieces[0]]
        for i in range(1, len(raw_pieces)):
            prev = raw_pieces[i - 1]
            overlap_text = prev[-self._chunk_overlap:]
            # Find a clean word boundary in the overlap
            space_idx = overlap_text.find(" ")
            if space_idx != -1:
                overlap_text = overlap_text[space_idx + 1:]
            overlapped.append(overlap_text + raw_pieces[i])
        return overlapped

    def _split_by_separators(
        self, text: str, separators: list[str], max_size: int
    ) -> list[str]:
        """Split text using the first separator that produces pieces under max_size."""
        if len(text) <= max_size:
            return [text]

        for sep in separators:
            parts = text.split(sep)
            if len(parts) <= 1:
                continue

            # Merge parts into pieces that fit within max_size
            pieces = []
            current = parts[0]
            for part in parts[1:]:
                candidate = current + sep + part
                if len(candidate) <= max_size:
                    current = candidate
                else:
                    if current.strip():
                        pieces.append(current.strip())
                    current = part
            if current.strip():
                pieces.append(current.strip())

            if pieces:
                # Recurse on any pieces that are still too large
                result = []
                remaining_seps = separators[separators.index(sep) + 1:]
                for piece in pieces:
                    if len(piece) > max_size and remaining_seps:
                        result.extend(self._split_by_separators(piece, remaining_seps, max_size))
                    else:
                        result.append(piece)
                return result

        # Last resort: return as-is (text is smaller than smallest split unit)
        return [text]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/chunker.py tests/test_chunker.py
git commit -m "Add sectioned and recursive splitting to RecursiveContextualChunker"
```

---

### Task 6: RecursiveContextualChunker — contextual headers

**Files:**
- Modify: `app/rag/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_chunker.py`:

```python
from unittest.mock import AsyncMock, patch
import asyncio


def test_contextual_headers_applied():
    """Test that contextual headers are prepended when enabled."""
    doc = {
        "id": "hdr_test",
        "name": "Header Test Strategy",
        "description": "Test description.",
        "document_type": "strategy",
        "tags": ["test"],
        "age_range": ["school_age"],
        "evidence_level": "strong",
        "source": "Test",
        "citations": [],
        "steps": ["Step one.", "Step two."],
    }

    mock_gemini = AsyncMock()
    mock_gemini.generate.return_value = "This chunk is from Header Test Strategy."

    chunker = RecursiveContextualChunker(contextual_headers=True, gemini_client=mock_gemini)
    chunks = chunker.chunk(doc)

    # Before add_contextual_headers is called, headers should be empty
    assert all(c.context_header == "" for c in chunks)

    # Now apply headers
    updated = asyncio.run(chunker.add_contextual_headers(chunks, doc))

    assert len(updated) == 3  # description + 2 steps
    for c in updated:
        assert c.context_header == "This chunk is from Header Test Strategy."
        assert c.text.startswith("This chunk is from Header Test Strategy. ")
    assert mock_gemini.generate.call_count == 3


def test_contextual_headers_disabled():
    """When contextual_headers=False, add_contextual_headers is a no-op."""
    chunks = [
        Chunk(
            chunk_id="x__step_1", parent_document_id="x",
            text="original", raw_text="original", context_header="",
            chunk_type="step", chunk_index=1, metadata={},
        )
    ]
    chunker = RecursiveContextualChunker(contextual_headers=False)
    updated = asyncio.run(chunker.add_contextual_headers(chunks, {}))
    assert updated[0].text == "original"
    assert updated[0].context_header == ""


def test_contextual_headers_graceful_failure():
    """If LLM call fails, chunk is kept without header."""
    doc = {
        "id": "fail_test",
        "name": "Fail Test",
        "description": "Desc.",
        "steps": ["Step."],
        "tags": [], "age_range": [], "evidence_level": "", "source": "", "citations": [],
    }

    mock_gemini = AsyncMock()
    mock_gemini.generate.side_effect = Exception("API error")

    chunker = RecursiveContextualChunker(contextual_headers=True, gemini_client=mock_gemini)
    chunks = chunker.chunk(doc)
    updated = asyncio.run(chunker.add_contextual_headers(chunks, doc))
    assert all(c.context_header == "" for c in updated)
    assert all("API error" not in c.text for c in updated)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_contextual_headers_applied tests/test_chunker.py::test_contextual_headers_disabled tests/test_chunker.py::test_contextual_headers_graceful_failure -v`
Expected: FAIL — `add_contextual_headers` not defined.

- [ ] **Step 3: Implement add_contextual_headers**

Add to `RecursiveContextualChunker` in `app/rag/chunker.py`:

```python
import asyncio  # add to imports

# Add this method to RecursiveContextualChunker:

    async def add_contextual_headers(
        self, chunks: list[Chunk], document: dict
    ) -> list[Chunk]:
        """Generate and prepend LLM contextual headers to chunks.

        Called separately from chunk() because it requires async LLM calls.
        Returns the same chunks with context_header and text updated.
        """
        if not self._contextual_headers or not self._gemini:
            return chunks

        doc_name = document.get("name", "")
        doc_type = document.get("document_type", "")
        total = len(chunks)

        async def _generate_header(chunk: Chunk) -> Chunk:
            prompt = (
                f"Write a 1-2 sentence context header for this chunk from a knowledge base document.\n"
                f"Document: '{doc_name}' (type: {doc_type})\n"
                f"Chunk {chunk.chunk_index + 1} of {total} (type: {chunk.chunk_type})\n"
                f"Chunk text: {chunk.raw_text[:500]}\n\n"
                f"Write ONLY the context header, nothing else. Be concise."
            )
            try:
                header = await self._gemini.generate(prompt)
                header = header.strip()
                chunk.context_header = header
                chunk.text = f"{header} {chunk.raw_text}"
            except Exception as e:
                logger.warning("Header generation failed for %s: %s", chunk.chunk_id, e)
            return chunk

        # Batch parallel generation
        tasks = [_generate_header(c) for c in chunks]
        return list(await asyncio.gather(*tasks))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/chunker.py tests/test_chunker.py
git commit -m "Add contextual header generation to RecursiveContextualChunker"
```

---

### Task 7: SemanticChunker

**Files:**
- Modify: `app/rag/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_chunker.py`:

```python
from app.rag.chunker import SemanticChunker
import numpy as np


def test_semantic_chunker_structured_doc():
    """Structured docs use steps/key_points as sentence units."""
    doc = {
        "id": "sem1",
        "name": "Semantic Test",
        "description": "A test strategy.",
        "document_type": "strategy",
        "tags": ["test"],
        "age_range": [],
        "evidence_level": "",
        "source": "",
        "citations": [],
        "steps": ["Step A about homework.", "Step B about homework.", "Step C about emotions."],
    }

    # Mock embeddings: steps A and B are similar, C is different
    mock_gemini = AsyncMock()
    embeddings = [
        [1.0, 0.0, 0.0],  # description
        [0.9, 0.1, 0.0],  # step A (similar to B)
        [0.85, 0.15, 0.0],  # step B (similar to A)
        [0.0, 0.0, 1.0],  # step C (different)
    ]
    mock_gemini.embed_batch = AsyncMock(return_value=embeddings)

    chunker = SemanticChunker(gemini_client=mock_gemini, similarity_threshold=0.5)
    chunks = asyncio.run(chunker.chunk_async(doc))

    # Should group A+B together (similar) and C separate, plus description
    # At minimum: boundary between B and C due to low similarity
    assert len(chunks) >= 2
    assert all(c.parent_document_id == "sem1" for c in chunks)
    assert all(c.chunk_type == "text_segment" for c in chunks)


def test_semantic_chunker_finds_boundaries():
    """Verify boundary detection with controlled embeddings."""
    from app.rag.chunker import SemanticChunker

    # 5 embeddings: first 3 similar, then a gap, then 2 similar
    embeddings = [
        [1.0, 0.0],
        [0.95, 0.05],
        [0.9, 0.1],
        [0.0, 1.0],  # big shift here
        [0.05, 0.95],
    ]

    chunker = SemanticChunker(similarity_threshold=0.5)
    boundaries = chunker._find_boundaries(embeddings, threshold=0.5)
    # Should find a boundary between index 2 and 3
    assert 3 in boundaries


def test_semantic_chunker_merge_small():
    """Small chunks below target size should be merged."""
    chunker = SemanticChunker(chunk_size=200, similarity_threshold=0.5)
    texts = ["Short A.", "Short B.", "Short C."]
    merged = chunker._merge_small_chunks(texts, target_size=200)
    # All three are tiny, should merge into one
    assert len(merged) == 1
    assert "Short A." in merged[0]
    assert "Short C." in merged[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_semantic_chunker_structured_doc tests/test_chunker.py::test_semantic_chunker_finds_boundaries tests/test_chunker.py::test_semantic_chunker_merge_small -v`
Expected: FAIL — `SemanticChunker` not defined.

- [ ] **Step 3: Implement SemanticChunker**

Add to `app/rag/chunker.py`:

```python
import numpy as np  # add to imports

class SemanticChunker:
    """Embedding-based boundary detection chunker.

    Splits on topic shifts detected by cosine similarity drops between adjacent sentences.
    """

    def __init__(
        self,
        gemini_client=None,
        similarity_threshold: float = 0.75,
        chunk_size: int = 2048,
    ):
        self._gemini = gemini_client
        self._threshold = similarity_threshold
        self._chunk_size = chunk_size

    def chunk(self, document: dict) -> list[Chunk]:
        """Synchronous fallback — cannot do semantic splitting without embeddings.

        Returns single chunk like NoneChunker. Use chunk_async() for real semantic splitting.
        """
        doc_id = document.get("id", "")
        doc_name = document.get("name", "")
        meta = _extract_metadata(document)
        text = self._extract_sentences_text(document)
        prefixed = f"[{doc_name}] {text}" if doc_name else text
        return [Chunk(
            chunk_id=f"{doc_id}__text_segment_0",
            parent_document_id=doc_id,
            text=prefixed,
            raw_text=text,
            context_header="",
            chunk_type="text_segment",
            chunk_index=0,
            metadata=meta,
        )]

    async def chunk_async(self, document: dict) -> list[Chunk]:
        """Async semantic chunking with embedding-based boundary detection."""
        doc_id = document.get("id", "")
        doc_name = document.get("name", "")
        meta = _extract_metadata(document)

        sentences = self._extract_sentences(document)
        if len(sentences) <= 1:
            text = sentences[0] if sentences else ""
            prefixed = f"[{doc_name}] {text}" if doc_name else text
            return [Chunk(
                chunk_id=f"{doc_id}__text_segment_0",
                parent_document_id=doc_id,
                text=prefixed,
                raw_text=text,
                context_header="",
                chunk_type="text_segment",
                chunk_index=0,
                metadata=meta,
            )]

        # Embed all sentences
        embeddings = await self._gemini.embed_batch(sentences)

        # Find boundaries
        boundaries = self._find_boundaries(embeddings, self._threshold)

        # Split sentences at boundaries
        groups = []
        start = 0
        for b in sorted(boundaries):
            groups.append(" ".join(sentences[start:b]))
            start = b
        groups.append(" ".join(sentences[start:]))
        groups = [g for g in groups if g.strip()]

        # Merge small chunks
        merged = self._merge_small_chunks(groups, self._chunk_size)

        chunks = []
        for i, text in enumerate(merged):
            prefixed = f"[{doc_name}] {text}" if doc_name else text
            chunks.append(Chunk(
                chunk_id=f"{doc_id}__text_segment_{i}",
                parent_document_id=doc_id,
                text=prefixed,
                raw_text=text,
                context_header="",
                chunk_type="text_segment",
                chunk_index=i,
                metadata=meta,
            ))
        return chunks

    def _extract_sentences(self, document: dict) -> list[str]:
        """Extract sentence units from a document.

        Structured docs: description + each step/key_point is a sentence.
        Unstructured: split description on sentence boundaries.
        """
        sentences = []
        description = document.get("description", "")
        steps = document.get("steps", [])
        key_points = document.get("key_points", [])

        if steps or key_points:
            if description:
                sentences.append(description)
            sentences.extend(steps)
            sentences.extend(key_points)
        else:
            # Split on sentence boundaries
            if description:
                parts = re.split(r'(?<=[.!?])\s+', description)
                sentences.extend([p for p in parts if p.strip()])

        return sentences or [description]

    def _extract_sentences_text(self, document: dict) -> str:
        """Join all sentence units into a single text."""
        return " ".join(self._extract_sentences(document))

    def _find_boundaries(
        self, embeddings: list[list[float]], threshold: float
    ) -> list[int]:
        """Find indices where cosine similarity between adjacent embeddings drops below threshold."""
        boundaries = []
        for i in range(1, len(embeddings)):
            a = np.asarray(embeddings[i - 1])
            b = np.asarray(embeddings[i])
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                boundaries.append(i)
                continue
            sim = float(np.dot(a, b) / (norm_a * norm_b))
            if sim < threshold:
                boundaries.append(i)
        return boundaries

    def _merge_small_chunks(self, texts: list[str], target_size: int) -> list[str]:
        """Merge adjacent chunks that are smaller than target_size."""
        if not texts:
            return texts
        merged = [texts[0]]
        for text in texts[1:]:
            if len(merged[-1]) + len(text) + 1 <= target_size:
                merged[-1] = merged[-1] + " " + text
            else:
                merged.append(text)
        return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/chunker.py tests/test_chunker.py
git commit -m "Add SemanticChunker with embedding-based boundary detection"
```

---

### Task 8: Integrate chunker into KnowledgeStore

**Files:**
- Modify: `app/rag/knowledge_store.py:38-108`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_chunker.py`:

```python
from app.rag.knowledge_store import KnowledgeStore
from app.rag.chunker import NoneChunker, RecursiveContextualChunker


def test_knowledge_store_with_none_chunker(monkeypatch):
    """KnowledgeStore with NoneChunker produces same chunk count as documents."""
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "none")
    store = KnowledgeStore()
    assert len(store.chunks) == len(store.documents)


def test_knowledge_store_with_recursive_chunker(monkeypatch):
    """KnowledgeStore with recursive chunker produces more chunks than documents."""
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "recursive_contextual")
    monkeypatch.setattr("app.config.settings.RAG_CONTEXTUAL_HEADERS", False)
    store = KnowledgeStore()
    assert len(store.chunks) > len(store.documents)
    # Each chunk dict should have document_id and document_name
    for chunk in store.chunks:
        assert "document_id" in chunk
        assert "document_name" in chunk
        assert "text" in chunk
        assert "tags" in chunk


def test_knowledge_store_collection_name_varies_by_strategy(monkeypatch):
    """Collection name should differ per strategy."""
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "none")
    store_none = KnowledgeStore()

    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "recursive_contextual")
    monkeypatch.setattr("app.config.settings.RAG_CONTEXTUAL_HEADERS", False)
    store_rc = KnowledgeStore()

    assert store_none._collection != store_rc._collection
    assert "recursive_contextual" in store_rc._collection
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_knowledge_store_with_none_chunker tests/test_chunker.py::test_knowledge_store_with_recursive_chunker tests/test_chunker.py::test_knowledge_store_collection_name_varies_by_strategy -v`
Expected: FAIL — `KnowledgeStore` doesn't use chunker yet.

- [ ] **Step 3: Modify KnowledgeStore to use configured chunker**

In `app/rag/knowledge_store.py`, make these changes:

1. Update `__init__` to accept and create a chunker, and derive collection name from strategy:

```python
# Add import at top:
from app.rag.chunker import Chunk, NoneChunker, RecursiveContextualChunker

# In __init__, update _collection to be strategy-aware:
    def __init__(self, knowledge_dir: Path | None = None):
        self.knowledge_dir = knowledge_dir or (
            Path(__file__).parent.parent / "knowledge"
        )
        self.documents: list[dict] = []
        self.chunks: list[dict] = []
        self._client: QdrantClient | None = None
        self._indexed = False
        self._vocab: dict[str, int] = {}
        self._doc_index: dict[str, dict] = {}

        # Strategy-specific collection name
        strategy = settings.RAG_CHUNKING_STRATEGY
        base = settings.QDRANT_COLLECTION
        self._collection = base if strategy == "none" else f"{base}_{strategy}"

        # Create chunker based on config
        self._chunker = self._create_chunker()

        self._load_documents()
```

2. Add `_create_chunker` method:

```python
    def _create_chunker(self):
        strategy = settings.RAG_CHUNKING_STRATEGY
        if strategy == "none":
            return NoneChunker()
        elif strategy == "recursive_contextual":
            return RecursiveContextualChunker(
                chunk_size=settings.RAG_CHUNK_SIZE_TOKENS * 4,  # tokens -> chars approx
                chunk_overlap=settings.RAG_CHUNK_OVERLAP_TOKENS * 4,
                contextual_headers=settings.RAG_CONTEXTUAL_HEADERS,
                gemini_client=None,  # set later in build_index if needed
            )
        elif strategy == "semantic":
            # SemanticChunker needs async, so sync chunk() returns single chunk
            # Real chunking happens in build_index via chunk_async
            from app.rag.chunker import SemanticChunker
            return SemanticChunker(
                similarity_threshold=settings.RAG_SEMANTIC_SIMILARITY_THRESHOLD,
                chunk_size=settings.RAG_CHUNK_SIZE_TOKENS * 4,
            )
        else:
            logger.warning("Unknown chunking strategy '%s', using 'none'", strategy)
            return NoneChunker()
```

3. Replace `_create_chunks` to use the chunker:

```python
    def _create_chunks(self):
        for doc in self.documents:
            chunk_objects = self._chunker.chunk(doc)
            for chunk_obj in chunk_objects:
                self.chunks.append({
                    "document_id": chunk_obj.parent_document_id,
                    "document_name": chunk_obj.metadata.get("document_name", ""),
                    "text": chunk_obj.text,
                    "tags": chunk_obj.metadata.get("tags", []),
                    "source": chunk_obj.metadata.get("source", ""),
                    "evidence_level": chunk_obj.metadata.get("evidence_level", ""),
                    "document_type": chunk_obj.metadata.get("document_type", ""),
                    "age_range": chunk_obj.metadata.get("age_range", []),
                    "citations": chunk_obj.metadata.get("citations", []),
                    "chunk_id": chunk_obj.chunk_id,
                    "chunk_type": chunk_obj.chunk_type,
                    "context_header": chunk_obj.context_header,
                })
```

Note: `full_doc` is no longer stored in chunks. The retriever will look it up via `get_document_by_id()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing knowledge store tests to verify no regression**

Run: `./adhd312/bin/python -m pytest tests/test_rag.py -v`

The existing tests may fail because they expect `full_doc` in chunks. We'll fix this in Task 9 when we update the retriever. For now, note any failures.

- [ ] **Step 6: Commit**

```bash
git add app/rag/knowledge_store.py tests/test_chunker.py
git commit -m "Integrate configurable chunker into KnowledgeStore"
```

---

### Task 9: Update retriever — full_doc lookup and parent dedup

**Files:**
- Modify: `app/rag/retriever.py:337-351` (`_chunk_to_result`)
- Modify: `app/rag/retriever.py:68-150` (`retrieve` — add dedup step)
- Modify: `app/models/schemas.py:73-87` (add `chunk_id`, `chunk_type`)
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Add chunk_id and chunk_type to RetrievalResult**

In `app/models/schemas.py`, add after line 87 (`full_doc` field):

```python
    chunk_id: str = ""
    chunk_type: str = ""
```

- [ ] **Step 2: Write failing tests for dedup**

Append to `tests/test_chunker.py`:

```python
from app.models.schemas import RetrievalResult
from app.rag.retriever import HybridRetriever


def test_parent_dedup_keeps_highest_score(monkeypatch):
    """Parent dedup should keep the highest-scoring chunk per document."""
    monkeypatch.setattr("app.config.settings.RAG_PARENT_DEDUP", True)
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "none")

    store = KnowledgeStore()
    retriever = HybridRetriever(knowledge_store=store)

    results = [
        RetrievalResult(
            document_id="doc1", document_name="Doc 1", content="chunk A",
            score=0.8, chunk_id="doc1__step_1", chunk_type="step",
        ),
        RetrievalResult(
            document_id="doc1", document_name="Doc 1", content="chunk B",
            score=0.9, chunk_id="doc1__step_2", chunk_type="step",
        ),
        RetrievalResult(
            document_id="doc2", document_name="Doc 2", content="chunk C",
            score=0.7, chunk_id="doc2__step_1", chunk_type="step",
        ),
    ]

    deduped = retriever._deduplicate_by_parent(results)
    assert len(deduped) == 2
    # doc1's highest score (0.9) should win
    doc1_result = next(r for r in deduped if r.document_id == "doc1")
    assert doc1_result.score == 0.9
    assert doc1_result.chunk_id == "doc1__step_2"
    # doc2 appears once
    doc2_result = next(r for r in deduped if r.document_id == "doc2")
    assert doc2_result.score == 0.7


def test_parent_dedup_disabled(monkeypatch):
    """When dedup is disabled, all results pass through."""
    monkeypatch.setattr("app.config.settings.RAG_PARENT_DEDUP", False)
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "none")

    store = KnowledgeStore()
    retriever = HybridRetriever(knowledge_store=store)

    results = [
        RetrievalResult(
            document_id="doc1", document_name="Doc 1", content="A", score=0.8,
        ),
        RetrievalResult(
            document_id="doc1", document_name="Doc 1", content="B", score=0.9,
        ),
    ]
    deduped = retriever._deduplicate_by_parent(results)
    assert len(deduped) == 2


def test_chunk_to_result_looks_up_full_doc(monkeypatch):
    """_chunk_to_result should look up full_doc from store, not chunk dict."""
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "none")
    store = KnowledgeStore()
    retriever = HybridRetriever(knowledge_store=store)

    # Pick a known document
    first_doc = store.documents[0]
    chunk = {
        "document_id": first_doc["id"],
        "document_name": first_doc["name"],
        "text": "some chunk text",
        "tags": first_doc.get("tags", []),
        "source": first_doc.get("source", ""),
        "evidence_level": first_doc.get("evidence_level", ""),
        "document_type": first_doc.get("document_type", ""),
        "age_range": first_doc.get("age_range", []),
        "citations": first_doc.get("citations", []),
        "chunk_id": f"{first_doc['id']}__full_0",
        "chunk_type": "full",
    }
    result = retriever._chunk_to_result(chunk, 0.5, "keyword")
    assert result.full_doc == first_doc
    assert result.chunk_id == f"{first_doc['id']}__full_0"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py::test_parent_dedup_keeps_highest_score tests/test_chunker.py::test_parent_dedup_disabled tests/test_chunker.py::test_chunk_to_result_looks_up_full_doc -v`
Expected: FAIL — `_deduplicate_by_parent` does not exist, `_chunk_to_result` doesn't handle `chunk_id`.

- [ ] **Step 4: Update _chunk_to_result to look up full_doc from store**

In `app/rag/retriever.py`, replace `_chunk_to_result` (lines 337-351):

```python
    def _chunk_to_result(self, chunk: dict, score: float, match_type: str) -> RetrievalResult:
        doc_id = chunk["document_id"]
        full_doc = self._store.get_document_by_id(doc_id) or {}
        return RetrievalResult(
            document_id=doc_id,
            document_name=chunk["document_name"],
            content=chunk["text"],
            score=score,
            match_type=match_type,
            source=chunk.get("source", ""),
            tags=chunk.get("tags", []),
            evidence_level=chunk.get("evidence_level", ""),
            document_type=chunk.get("document_type", ""),
            age_range=chunk.get("age_range", []),
            citations=chunk.get("citations", []),
            full_doc=full_doc,
            chunk_id=chunk.get("chunk_id", ""),
            chunk_type=chunk.get("chunk_type", ""),
        )
```

- [ ] **Step 5: Add _deduplicate_by_parent method**

Add to `HybridRetriever` class in `app/rag/retriever.py`:

```python
    def _deduplicate_by_parent(
        self, candidates: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Keep only the highest-scoring chunk per parent document."""
        if not settings.RAG_PARENT_DEDUP:
            return candidates

        best: dict[str, RetrievalResult] = {}
        for r in candidates:
            if r.document_id not in best or r.score > best[r.document_id].score:
                best[r.document_id] = r

        deduped = list(best.values())
        deduped.sort(key=lambda r: r.score, reverse=True)
        return deduped
```

- [ ] **Step 6: Wire dedup into the retrieve() pipeline**

In `app/rag/retriever.py`, in the `retrieve` method, add the dedup call after the relevance threshold step (after line 116) and before outcome boost (line 119):

```python
        # Step 3d: Parent document deduplication
        candidates = self._deduplicate_by_parent(candidates)
```

Also add dedup to `_keyword_fallback` — call `self._deduplicate_by_parent(results)` before the final `return results[:top_k]`:

```python
        results.sort(key=lambda r: r.score, reverse=True)
        results = self._deduplicate_by_parent(results)
        return results[:top_k]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `./adhd312/bin/python -m pytest tests/test_chunker.py -v`
Expected: All PASS

- [ ] **Step 8: Run all existing tests to verify no regression**

Run: `./adhd312/bin/python -m pytest tests/test_rag.py tests/test_agent_tools.py -v`
Expected: All PASS. If any fail due to `full_doc` changes, fix them.

- [ ] **Step 9: Commit**

```bash
git add app/rag/retriever.py app/models/schemas.py tests/test_chunker.py
git commit -m "Add parent document deduplication and full_doc lookup to retriever"
```

---

### Task 10: Wire chunker into app startup (main.py)

**Files:**
- Modify: `app/main.py:55-64`
- Modify: `app/rag/knowledge_store.py` (build_index update for semantic/contextual)

- [ ] **Step 1: Add async enrichment to build_index (before embedding)**

In `app/rag/knowledge_store.py`, restructure `build_index` so async chunk enrichment
happens **before** embedding, not after. Insert a new `_async_enrich_chunks` method that
runs between `_create_chunks` and the embedding step.

Add this method to `KnowledgeStore`:

```python
    async def _async_enrich_chunks(self, gemini_client):
        """Run async chunking enrichment before index build.

        For semantic strategy: re-chunk with embedding-based boundary detection.
        For recursive_contextual: add LLM contextual headers to chunk texts.
        """
        from app.rag.chunker import Chunk, RecursiveContextualChunker, SemanticChunker

        strategy = settings.RAG_CHUNKING_STRATEGY

        if strategy == "semantic" and isinstance(self._chunker, SemanticChunker):
            self._chunker._gemini = gemini_client
            new_chunks = []
            for doc in self.documents:
                chunk_objects = await self._chunker.chunk_async(doc)
                for co in chunk_objects:
                    new_chunks.append(self._chunk_obj_to_dict(co))
            self.chunks = new_chunks
            self._build_vocabulary()
            logger.info("Semantic chunking: %d chunks from %d documents", len(self.chunks), len(self.documents))

        elif (
            strategy == "recursive_contextual"
            and settings.RAG_CONTEXTUAL_HEADERS
            and isinstance(self._chunker, RecursiveContextualChunker)
        ):
            self._chunker._gemini = gemini_client
            enriched_chunks = []
            for doc in self.documents:
                doc_id = doc.get("id", "")
                doc_chunk_dicts = [c for c in self.chunks if c["document_id"] == doc_id]
                chunk_objs = [
                    Chunk(
                        chunk_id=c["chunk_id"],
                        parent_document_id=c["document_id"],
                        text=c["text"],
                        raw_text=c["text"],  # raw_text before header
                        context_header="",
                        chunk_type=c["chunk_type"],
                        chunk_index=i,
                        metadata={},
                    )
                    for i, c in enumerate(doc_chunk_dicts)
                ]
                updated_objs = await self._chunker.add_contextual_headers(chunk_objs, doc)
                for orig_dict, updated_obj in zip(doc_chunk_dicts, updated_objs):
                    orig_dict["text"] = updated_obj.text
                    orig_dict["context_header"] = updated_obj.context_header
                    enriched_chunks.append(orig_dict)
            self.chunks = enriched_chunks
            self._build_vocabulary()
            logger.info("Contextual headers applied to %d chunks", len(self.chunks))

    @staticmethod
    def _chunk_obj_to_dict(co: "Chunk") -> dict:
        return {
            "document_id": co.parent_document_id,
            "document_name": co.metadata.get("document_name", ""),
            "text": co.text,
            "tags": co.metadata.get("tags", []),
            "source": co.metadata.get("source", ""),
            "evidence_level": co.metadata.get("evidence_level", ""),
            "document_type": co.metadata.get("document_type", ""),
            "age_range": co.metadata.get("age_range", []),
            "citations": co.metadata.get("citations", []),
            "chunk_id": co.chunk_id,
            "chunk_type": co.chunk_type,
            "context_header": co.context_header,
        }
```

Then in `build_index`, add the enrichment call **before** the embedding step (before `texts = [chunk["text"] for chunk in self.chunks]`):

```python
        # Async chunk enrichment (semantic re-chunking or contextual headers)
        await self._async_enrich_chunks(gemini_client)
```

This is clean: no recursive calls, no double-prefixed text, runs once before embedding.

- [ ] **Step 2: Update main.py to log chunking strategy**

In `app/main.py`, add after the Qdrant index build log (line 62):

```python
            logger.info("Chunking strategy: %s", settings.RAG_CHUNKING_STRATEGY)
```

- [ ] **Step 3: Run all tests**

Run: `./adhd312/bin/python -m pytest tests/ -v -m "not integration"`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add app/rag/knowledge_store.py app/main.py
git commit -m "Wire chunker into app startup with async semantic and contextual support"
```

---

### Task 11: Run full test suite and fix regressions

**Files:**
- Possibly modify: any test file that breaks

- [ ] **Step 1: Run the full test suite**

Run: `./adhd312/bin/python -m pytest tests/ -v -m "not integration"`

- [ ] **Step 2: Fix any failures**

Common expected issues:
- Tests that access `chunk["full_doc"]` directly — update to use `store.get_document_by_id()`
- Tests that assert `len(store.chunks) == len(store.documents)` — update if strategy is not "none"
- Chunk dict shape changes (new fields like `chunk_id`, `chunk_type`)

Fix each failure, keeping changes minimal.

- [ ] **Step 3: Run tests again to confirm all pass**

Run: `./adhd312/bin/python -m pytest tests/ -v -m "not integration"`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add app/ tests/
git commit -m "Fix test regressions from chunking integration"
```

---

### Task 12: Evaluation script for strategy comparison

**Files:**
- Create: `eval/runners/chunking_comparison.py`

- [ ] **Step 1: Create the evaluation script**

Create `eval/runners/chunking_comparison.py`:

```python
"""Compare retrieval quality across chunking strategies.

Builds all three collections (none, recursive_contextual, semantic),
runs the same queries against each, and reports Recall@k and MRR.

Requires GEMINI_API_KEY. Run: ./adhd312/bin/python eval/runners/chunking_comparison.py
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Test queries with expected document IDs
EVAL_QUERIES = [
    {"query": "How can I help my child with homework?", "expected": ["schoolbased_interventions_for_adhd"]},
    {"query": "daily report card for school", "expected": ["schoolbased_interventions_for_adhd"]},
    {"query": "managing emotions and frustration", "expected": []},  # fill with actual doc IDs
    {"query": "positive reinforcement techniques", "expected": ["parent_training_in_behavior_therapy"]},
    {"query": "what is ADHD", "expected": []},  # fill with actual doc IDs
]


async def evaluate_strategy(strategy: str, queries: list[dict], top_k: int = 5) -> dict:
    """Build index for a strategy and evaluate retrieval quality."""
    os.environ["RAG_CHUNKING_STRATEGY"] = strategy
    os.environ["RAG_CONTEXTUAL_HEADERS"] = "false"  # skip headers for fair comparison

    # Re-import to pick up new settings
    import importlib
    import app.config
    importlib.reload(app.config)

    from app.config import settings
    from app.llm.client import GeminiClient
    from app.rag.knowledge_store import KnowledgeStore
    from app.rag.retriever import HybridRetriever

    gemini = GeminiClient()
    store = KnowledgeStore()
    await store.build_index(gemini)

    retriever = HybridRetriever(knowledge_store=store, gemini_client=gemini)

    results = {"strategy": strategy, "queries": [], "recall_at_k": 0.0, "mrr": 0.0}
    total_recall = 0.0
    total_rr = 0.0
    evaluated = 0

    for q in queries:
        if not q["expected"]:
            continue
        evaluated += 1
        response = await retriever.retrieve(q["query"], top_k=top_k, skip_rewrite=True)
        retrieved_ids = [r.document_id for r in response.results]

        # Recall@k
        hits = sum(1 for eid in q["expected"] if eid in retrieved_ids)
        recall = hits / len(q["expected"])
        total_recall += recall

        # MRR
        rr = 0.0
        for eid in q["expected"]:
            if eid in retrieved_ids:
                rank = retrieved_ids.index(eid) + 1
                rr = max(rr, 1.0 / rank)
        total_rr += rr

        results["queries"].append({
            "query": q["query"],
            "expected": q["expected"],
            "retrieved": retrieved_ids[:top_k],
            "recall": recall,
            "rr": rr,
        })

    if evaluated > 0:
        results["recall_at_k"] = total_recall / evaluated
        results["mrr"] = total_rr / evaluated

    logger.info(
        "Strategy=%s  Recall@%d=%.3f  MRR=%.3f",
        strategy, top_k, results["recall_at_k"], results["mrr"],
    )
    return results


async def main():
    strategies = ["none", "recursive_contextual", "semantic"]
    all_results = []

    for strategy in strategies:
        logger.info("--- Evaluating strategy: %s ---", strategy)
        result = await evaluate_strategy(strategy, EVAL_QUERIES)
        all_results.append(result)

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Strategy':<25} {'Recall@5':<12} {'MRR':<12}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['strategy']:<25} {r['recall_at_k']:<12.3f} {r['mrr']:<12.3f}")
    print("=" * 60)

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "results" / "chunking_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Verify the script loads without errors (don't run the full eval)**

Run: `./adhd312/bin/python -c "import eval.runners.chunking_comparison; print('OK')"`
Expected: `OK` (no import errors)

- [ ] **Step 3: Commit**

```bash
git add eval/runners/chunking_comparison.py
git commit -m "Add chunking strategy comparison evaluation script"
```
