"""
Knowledge Base Manager.

Handles loading, indexing, and managing the ADHD coaching
knowledge base. In production, this will manage FAISS indexes
and document embeddings.
"""

import json
from pathlib import Path


class KnowledgeBase:
    """Manages the vetted ADHD parenting knowledge base."""

    def __init__(self, knowledge_dir: Path | None = None):
        self.knowledge_dir = knowledge_dir or (
            Path(__file__).parent.parent / "knowledge"
        )
        self.documents: list[dict] = []
        self._load()

    def _load(self):
        """Load all knowledge documents."""
        strategies_path = self.knowledge_dir / "adhd_strategies.json"
        if strategies_path.exists():
            with open(strategies_path) as f:
                self.documents = json.load(f)

    def search(self, tags: list[str]) -> list[dict]:
        """Find strategies matching any of the given tags."""
        results = []
        tag_set = set(t.lower() for t in tags)
        for doc in self.documents:
            doc_tags = set(t.lower() for t in doc.get("tags", []))
            if tag_set & doc_tags:
                results.append(doc)
        return results

    def get_all_topics(self) -> list[str]:
        """Return all unique tags/topics in the knowledge base."""
        topics = set()
        for doc in self.documents:
            topics.update(doc.get("tags", []))
        return sorted(topics)
