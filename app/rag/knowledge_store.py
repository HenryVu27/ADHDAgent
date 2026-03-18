# Qdrant-based Knowledge Store
# Loads JSON docs, chunks them, embeds with Gemini, stores in Qdrant for
# hybrid search (dense + sparse vectors with server-side RRF fusion).

import json
import logging
import re
from collections import Counter
from pathlib import Path

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

    def __init__(self, knowledge_dir: Path | None = None):
        self.knowledge_dir = knowledge_dir or (
            Path(__file__).parent.parent / "knowledge"
        )
        self.documents: list[dict] = []
        self.chunks: list[dict] = []
        self._client: QdrantClient | None = None
        self._collection = settings.QDRANT_COLLECTION
        self._indexed = False
        self._vocab: dict[str, int] = {}
        self._doc_index: dict[str, dict] = {}

        self._load_documents()

    # Load all JSON knowledge files
    def _load_documents(self):
        for json_file in sorted(self.knowledge_dir.glob("*.json")):
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

    # One chunk per document — concatenate name, description, steps, key_points
    def _create_chunks(self):
        for doc in self.documents:
            text_parts = [doc.get("name", ""), doc.get("description", "")]

            steps = doc.get("steps", [])
            if steps:
                text_parts.append("Steps: " + " | ".join(steps))

            key_points = doc.get("key_points", [])
            if key_points:
                text_parts.append("Key points: " + " | ".join(key_points))

            self.chunks.append({
                "document_id": doc.get("id", ""),
                "document_name": doc.get("name", ""),
                "text": " ".join(text_parts),
                "tags": doc.get("tags", []),
                "source": doc.get("source", ""),
                "evidence_level": doc.get("evidence_level", ""),
                "document_type": doc.get("document_type", ""),
                "age_range": doc.get("age_range", []),
                "citations": doc.get("citations", []),
                "full_doc": doc,
            })

    # Build token->index mapping for sparse vectors
    def _build_vocabulary(self):
        self._vocab = {}
        next_id = 0
        for chunk in self.chunks:
            for token in self._tokenize(chunk["text"]):
                if token not in self._vocab:
                    self._vocab[token] = next_id
                    next_id += 1

    # Convert text to sparse vector using TF weights (Qdrant applies IDF server-side)
    def _text_to_sparse(self, text: str) -> SparseVector:
        tokens = self._tokenize(text)
        counts = Counter(tokens)

        indices = []
        values = []
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

    # Build Qdrant collection with dense (Gemini) + sparse (TF) vectors
    # Idempotent: skips rebuild if collection already has correct point count
    async def build_index(self, gemini_client):
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

        texts = [chunk["text"] for chunk in self.chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        raw_embeddings = await gemini_client.embed_batch(texts, timeout=settings.RAG_EMBED_TIMEOUT_S)
        dim = len(raw_embeddings[0])

        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": VectorParams(size=dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF),
            },
        )

        points = []
        for i, (embedding, chunk) in enumerate(zip(raw_embeddings, self.chunks)):
            sparse_vec = self._text_to_sparse(chunk["text"])
            points.append(PointStruct(
                id=i,
                vector={
                    "dense": embedding,
                    "sparse": sparse_vec,
                },
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
            f"dim={dim}, vocab={len(self._vocab)}"
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

    # Qdrant hybrid search: dense + sparse with RRF fusion
    def search_hybrid(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        filters: RetrievalFilters | None = None,
    ) -> list[tuple[int, float, dict]]:
        if not self._client or not self._indexed:
            return []

        qdrant_filter = self._build_filter(filters) if filters else None
        sparse_vec = self._text_to_sparse(query_text)
        prefetch_limit = min(top_k * 3, len(self.chunks))

        results = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
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
            ],
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
