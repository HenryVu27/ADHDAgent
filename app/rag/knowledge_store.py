# Qdrant-based Knowledge Store
# Loads JSON docs, chunks them, embeds with Gemini, stores in Qdrant for
# hybrid search (dense + sparse vectors with server-side RRF fusion).

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.rag.colbert_index import ColBERTIndex

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Modifier,
    MultiVectorConfig,
    MultiVectorComparator,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.models.schemas import RetrievalFilters

logger = logging.getLogger(__name__)


class KnowledgeStore:
    # Manages the ADHD knowledge base with Qdrant vector storage

    def __init__(
        self,
        knowledge_dir: Path | None = None,
        sparse_mode: Literal["tfidf", "bm25"] = "tfidf",
        collection_name: str | None = None,
    ):
        self.knowledge_dir = knowledge_dir or (
            Path(__file__).parent.parent / "knowledge"
        )
        self._sparse_mode = sparse_mode
        self.documents: list[dict] = []
        self.chunks: list[dict] = []
        self._client: QdrantClient | None = None
        self._indexed = False
        self._vocab: dict[str, int] = {}
        self._doc_index: dict[str, dict] = {}
        self._avg_doc_len: float = 0.0

        # Strategy-specific collection name
        strategy = settings.RAG_CHUNKING_STRATEGY
        base = collection_name or settings.QDRANT_COLLECTION
        self._collection = base if strategy == "none" else f"{base}_{strategy}"

        # Create chunker based on config
        self._chunker = self._create_chunker()

        self._load_documents()

    # Load all JSON knowledge files
    def _load_documents(self):
        for json_file in sorted(self.knowledge_dir.glob("**/*.json")):
            try:
                with open(json_file) as f:
                    docs = json.load(f)
                    self.documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} docs from {json_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

        self._build_doc_index()
        self._create_chunks()
        self._build_vocabulary()
        logger.info(f"Knowledge store: {len(self.documents)} documents, {len(self.chunks)} chunks, vocab={len(self._vocab)}")

    # Build O(1) lookup index by document ID
    def _build_doc_index(self):
        for doc in self.documents:
            doc_id = doc.get("id", "")
            if doc_id:
                self._doc_index[doc_id] = doc

    def _create_chunker(self):
        from app.rag.chunker import NoneChunker, RecursiveContextualChunker

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
            from app.rag.chunker import SemanticChunker
            return SemanticChunker(
                similarity_threshold=settings.RAG_SEMANTIC_SIMILARITY_THRESHOLD,
                chunk_size=settings.RAG_CHUNK_SIZE_TOKENS * 4,
            )
        else:
            logger.warning("Unknown chunking strategy '%s', using 'none'", strategy)
            return NoneChunker()

    # Create chunks using the configured chunker
    def _create_chunks(self):
        from app.rag.chunker import Chunk as ChunkObj

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

    # Build token->index mapping for sparse vectors
    def _build_vocabulary(self):
        self._vocab = {}
        next_id = 0
        doc_lengths: list[int] = []
        for chunk in self.chunks:
            tokens = self._tokenize(chunk["text"])
            doc_lengths.append(len(tokens))
            for token in tokens:
                if token not in self._vocab:
                    self._vocab[token] = next_id
                    next_id += 1
        self._avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0

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
                        raw_text=c["text"],
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
    def _chunk_obj_to_dict(co) -> dict:
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

    # Convert text to sparse vector using TF or BM25 weights
    # (Qdrant applies IDF server-side via Modifier.IDF)
    def _text_to_sparse(self, text: str) -> SparseVector:
        tokens = self._tokenize(text)
        counts = Counter(tokens)

        indices = []
        values = []

        if self._sparse_mode == "bm25":
            k1, b = 1.2, 0.75
            doc_len = len(tokens)
            for token, count in sorted(counts.items()):
                if token in self._vocab:
                    indices.append(self._vocab[token])
                    tf_sat = (count * (k1 + 1)) / (count + k1 * (1 - b + b * doc_len / self._avg_doc_len))
                    values.append(tf_sat)
        else:
            for token, count in sorted(counts.items()):
                if token in self._vocab:
                    indices.append(self._vocab[token])
                    values.append(float(count))

        if not indices:
            return SparseVector(indices=[0], values=[0.0])

        return SparseVector(indices=indices, values=values)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    # Build Qdrant collection with dense (Gemini) + sparse (TF/BM25) vectors
    # and optionally ColBERT multi-vectors.
    # Idempotent: skips rebuild if collection already has correct point count
    async def build_index(
        self,
        gemini_client,
        colbert_index: ColBERTIndex | None = None,
    ):
        if not self.chunks:
            logger.warning("No chunks to index")
            return

        if settings.QDRANT_URL == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
            )

        if self._collection_is_current():
            self._indexed = True
            logger.info(
                f"Collection '{self._collection}' already exists with "
                f"{len(self.chunks)} points — skipping rebuild"
            )
            return

        # Async chunk enrichment (semantic re-chunking or contextual headers)
        await self._async_enrich_chunks(gemini_client)

        texts = [chunk["text"] for chunk in self.chunks]
        logger.info(f"Embedding {len(texts)} chunks (sparse_mode={self._sparse_mode})...")
        raw_embeddings = await gemini_client.embed_batch(texts, timeout=settings.RAG_EMBED_TIMEOUT_S)
        dim = len(raw_embeddings[0])

        # ColBERT multi-vector embeddings (optional)
        colbert_embeddings: list[list[list[float]]] | None = None
        if colbert_index is not None:
            logger.info("Generating ColBERT per-token embeddings for %d chunks...", len(self.chunks))
            colbert_embeddings = colbert_index.embed_chunks(self.chunks)

        # Build collection schema
        vectors_config: dict = {
            "dense": VectorParams(size=dim, distance=Distance.COSINE),
        }
        if colbert_embeddings is not None:
            from app.rag.colbert_index import ColBERTIndex as _CI
            vectors_config["colbert"] = VectorParams(
                size=_CI.DIM,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            )

        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=vectors_config,
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF),
            },
        )

        points = []
        for i, (embedding, chunk) in enumerate(zip(raw_embeddings, self.chunks)):
            sparse_vec = self._text_to_sparse(chunk["text"])
            vectors: dict = {
                "dense": embedding,
                "sparse": sparse_vec,
            }
            if colbert_embeddings is not None:
                vectors["colbert"] = colbert_embeddings[i]

            points.append(PointStruct(
                id=i,
                vector=vectors,
                payload={
                    "document_id": chunk["document_id"],
                    "document_name": chunk["document_name"],
                    "text": chunk["text"],
                    "tags": chunk["tags"],
                    "source": chunk["source"],
                    "evidence_level": chunk["evidence_level"],
                    "document_type": chunk["document_type"],
                    "age_range": chunk["age_range"],
                    "citations": chunk["citations"],
                },
            ))

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

        self._create_payload_indexes()
        self._indexed = True
        logger.info(
            f"Qdrant collection built: {len(points)} vectors, "
            f"dim={dim}, sparse={self._sparse_mode}, "
            f"colbert={'yes' if colbert_embeddings else 'no'}, "
            f"vocab={len(self._vocab)}"
        )

    # Check if collection exists with correct schema and point count
    def _collection_is_current(self) -> bool:
        try:
            collections = self._client.get_collections().collections
            exists = any(c.name == self._collection for c in collections)
            if not exists:
                return False

            info = self._client.get_collection(self._collection)

            vectors_config = info.config.params.vectors
            if not (isinstance(vectors_config, dict) and "dense" in vectors_config):
                logger.info(f"Collection '{self._collection}' uses old vector schema — rebuilding")
                return False

            sparse_config = info.config.params.sparse_vectors
            if not sparse_config or "sparse" not in sparse_config:
                logger.info(f"Collection '{self._collection}' missing sparse vectors — rebuilding")
                return False

            if info.points_count == len(self.chunks):
                return True

            logger.info(
                f"Collection point count ({info.points_count}) differs "
                f"from chunk count ({len(self.chunks)}) — rebuilding"
            )
            return False
        except Exception as e:
            logger.debug(f"Collection check failed: {e}")
            return False

    # Create keyword indexes on filterable payload fields
    def _create_payload_indexes(self):
        indexed_fields = ["tags", "document_type", "evidence_level", "age_range", "source"]
        for field in indexed_fields:
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.warning(f"Failed to create index for '{field}': {e}")

        logger.info(f"Payload indexes created: {indexed_fields}")

    # Qdrant hybrid search: dense + sparse (+ optional colbert) with RRF fusion
    def search_hybrid(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
        colbert_prefetch: Prefetch | None = None,
    ) -> list[tuple[int, float, dict]]:
        if not self._client or not self._indexed:
            return []

        qdrant_filter = self._build_filter(filters) if filters else None
        sparse_vec = self._text_to_sparse(query_text)
        prefetch_limit = min(top_k * 3, len(self.chunks))

        prefetches = [
            Prefetch(
                query=query_vector,
                using="dense",
                limit=prefetch_limit,
            ),
            Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=prefetch_limit,
            ),
        ]
        if colbert_prefetch is not None:
            prefetches.append(colbert_prefetch)

        results = self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetches,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=qdrant_filter,
            limit=top_k,
        ).points

        return [
            (point.id, point.score, point.payload)
            for point in results
        ]

    # Dense-only vector search (fallback if sparse unavailable)
    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[tuple[int, float, dict]]:
        if not self._client or not self._indexed:
            return []

        qdrant_filter = self._build_filter(filters) if filters else None

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            query_filter=qdrant_filter,
        ).points

        return [
            (point.id, point.score, point.payload)
            for point in results
        ]

    # Convert RetrievalFilters to Qdrant Filter
    def _build_filter(self, filters: RetrievalFilters) -> Filter | None:
        conditions = []

        if filters.document_type:
            conditions.append(
                FieldCondition(key="document_type", match=MatchValue(value=filters.document_type))
            )
        if filters.tags:
            conditions.append(
                FieldCondition(key="tags", match=MatchAny(any=filters.tags))
            )
        if filters.age_range:
            conditions.append(
                FieldCondition(key="age_range", match=MatchAny(any=[filters.age_range, "all"]))
            )
        if filters.evidence_level:
            conditions.append(
                FieldCondition(key="evidence_level", match=MatchValue(value=filters.evidence_level))
            )
        if filters.source:
            conditions.append(
                FieldCondition(key="source", match=MatchValue(value=filters.source))
            )

        if not conditions:
            return None
        return Filter(must=conditions)

    # Find documents matching any of the given tags
    def search_by_tags(self, tags: list[str]) -> list[dict]:
        results = []
        tag_set = {t.lower() for t in tags}
        for doc in self.documents:
            doc_tags = {t.lower() for t in doc.get("tags", [])}
            if tag_set & doc_tags:
                results.append(doc)
        return results

    # Return all unique tags/topics
    def get_all_topics(self) -> list[str]:
        topics = set()
        for doc in self.documents:
            topics.update(doc.get("tags", []))
        return sorted(topics)

    # Look up a document by its ID. Returns None if not found.
    def get_document_by_id(self, doc_id: str) -> dict | None:
        return self._doc_index.get(doc_id)

    # Return full documents for all related_ids of the given document.
    def get_related_docs(self, doc_id: str) -> list[dict]:
        doc = self._doc_index.get(doc_id)
        if not doc:
            return []
        related_ids = doc.get("related_ids", [])
        results = []
        for rid in related_ids:
            related_doc = self._doc_index.get(rid)
            if related_doc:
                results.append(related_doc)
        return results

    @property
    def is_indexed(self) -> bool:
        return self._indexed

    @property
    def has_sparse(self) -> bool:
        return self._indexed and len(self._vocab) > 0
