# Chunking strategies for the RAG knowledge base.
# Supports structured JSON docs (steps/key_points), sectioned text, and generic text.

from __future__ import annotations

import asyncio
import logging
import re
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

        tasks = [_generate_header(c) for c in chunks]
        return list(await asyncio.gather(*tasks))

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

    def _chunk_sectioned(
        self, text: str, doc_id: str, doc_name: str, meta: dict
    ) -> list[Chunk]:
        """Split text on markdown headings. Recurse if any section exceeds chunk_size."""
        sections = re.split(r"(?=^#{1,4}\s+)", text, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]

        chunks = []
        for section in sections:
            if len(section) > self._chunk_size:
                sub_chunks = self._split_recursive(section)
                for sub_text in sub_chunks:
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

        # Last resort: return as-is
        return [text]


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
        """Synchronous fallback -- cannot do semantic splitting without embeddings.

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
        import numpy as np

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
        import numpy as np

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
