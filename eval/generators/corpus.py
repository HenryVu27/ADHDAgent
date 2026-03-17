"""Load raw knowledge documents without importing from app/.

Replicates the chunking logic from app/rag/knowledge_store.py so the
eval pipeline can generate test data from the same text representation
that the production retriever actually indexes.
"""

import json
from pathlib import Path

from eval.config import KNOWLEDGE_DIR


def load_chunks(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[dict]:
    """Load all JSON knowledge files and return one chunk per document.

    Chunk text is assembled the same way as KnowledgeStore._create_chunks()
    so eval queries test the actual indexed text.
    """
    documents: list[dict] = []
    for json_file in sorted(knowledge_dir.glob("*.json")):
        with open(json_file) as f:
            documents.extend(json.load(f))

    chunks = []
    for doc in documents:
        text_parts = [doc.get("name", ""), doc.get("description", "")]

        steps = doc.get("steps", [])
        if steps:
            text_parts.append("Steps: " + " | ".join(steps))

        key_points = doc.get("key_points", [])
        if key_points:
            text_parts.append("Key points: " + " | ".join(key_points))

        chunks.append({
            "document_id": doc["id"],
            "document_name": doc.get("name", ""),
            "document_type": doc.get("document_type", ""),
            "tags": doc.get("tags", []),
            "evidence_level": doc.get("evidence_level", ""),
            "age_range": doc.get("age_range", []),
            "text": " ".join(text_parts),
        })

    return chunks
