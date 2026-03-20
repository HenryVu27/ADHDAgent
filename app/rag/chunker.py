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
    """One chunk per document -- replicates the original KnowledgeStore behavior."""

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
