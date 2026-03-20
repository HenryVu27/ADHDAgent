import pytest
from scripts.ingest.pipeline.deduplicator import Deduplicator


class TestDeduplicator:
    def test_doi_exact_dedup(self):
        dedup = Deduplicator(title_threshold=0.85)
        docs = [
            {"id": "pmc_1", "name": "Study A", "citations": [{"detail": "10.1234/a"}]},
            {"id": "oalex_2", "name": "Study A Copy", "citations": [{"detail": "10.1234/a"}]},
        ]
        result = dedup.deduplicate(docs)
        assert len(result) == 1
        assert result[0]["id"] == "pmc_1"  # First seen wins

    def test_title_similarity_dedup(self):
        dedup = Deduplicator(title_threshold=0.85)
        docs = [
            {"id": "pmc_1", "name": "ADHD Parent Training: A Randomized Trial", "citations": [{"detail": ""}]},
            {"id": "oalex_2", "name": "ADHD Parent Training A Randomized Trial", "citations": [{"detail": ""}]},
        ]
        result = dedup.deduplicate(docs)
        assert len(result) == 1

    def test_different_titles_kept(self):
        dedup = Deduplicator(title_threshold=0.85)
        docs = [
            {"id": "pmc_1", "name": "ADHD Parent Training", "citations": [{"detail": ""}]},
            {"id": "pmc_2", "name": "Executive Function in Adolescents", "citations": [{"detail": ""}]},
        ]
        result = dedup.deduplicate(docs)
        assert len(result) == 2

    def test_empty_input(self):
        dedup = Deduplicator(title_threshold=0.85)
        assert dedup.deduplicate([]) == []

    def test_jaccard_similarity(self):
        dedup = Deduplicator(title_threshold=0.85)
        assert dedup._jaccard_similarity("ADHD Parent Training", "adhd parent training") == 1.0
        assert dedup._jaccard_similarity("ADHD", "Quantum Physics") == 0.0
