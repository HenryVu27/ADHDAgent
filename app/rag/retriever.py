"""
RAG Retriever for ADHD Coaching Knowledge Base.

Retrieves relevant ADHD parenting strategies and clinical guidance
using hybrid search (keyword + semantic).
"""

import json
from pathlib import Path


KNOWLEDGE_PATH = Path(__file__).parent.parent / "knowledge" / "adhd_strategies.json"


class RAGRetriever:
    """
    Hybrid retriever combining keyword matching and semantic search.

    Currently uses keyword matching for the demo.
    Production version will use FAISS for vector similarity search
    alongside BM25 for keyword retrieval.
    """

    def __init__(self):
        self.documents = self._load_knowledge()

    def _load_knowledge(self) -> list[dict]:
        if KNOWLEDGE_PATH.exists():
            with open(KNOWLEDGE_PATH) as f:
                return json.load(f)
        return []

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the most relevant strategies for a given query.

        Args:
            query: The parent's message or extracted topic
            top_k: Number of results to return

        Returns:
            List of relevant strategy documents
        """
        if not self.documents:
            return []

        scored = []
        query_lower = query.lower()

        for doc in self.documents:
            score = self._score_document(query_lower, doc)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def _score_document(self, query: str, doc: dict) -> float:
        """Score a document's relevance to the query using keyword overlap."""
        score = 0.0
        doc_text = (
            doc.get("name", "")
            + " "
            + doc.get("description", "")
            + " "
            + " ".join(doc.get("tags", []))
        ).lower()

        query_terms = query.split()
        for term in query_terms:
            if term in doc_text:
                score += 1.0

        # Boost for tag matches
        for tag in doc.get("tags", []):
            if tag.lower() in query:
                score += 2.0

        return score
