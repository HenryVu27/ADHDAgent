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
