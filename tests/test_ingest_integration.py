import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from scripts.ingest.pipeline.transformer import SchemaTransformer
from scripts.ingest.pipeline.deduplicator import Deduplicator
from scripts.ingest.output.writer import BatchWriter
from scripts.ingest.models import ParsedDocument


class TestEndToEnd:
    def test_full_transform_dedup_write_pipeline(self, tmp_path):
        """Test transformer -> deduplicator -> writer flow without network calls."""
        transformer = SchemaTransformer()

        # Two docs from different sources with same DOI (should dedup)
        parsed_pubmed = ParsedDocument(
            source_id="PMC111",
            source_name="pubmed",
            title="ADHD Parent Training RCT",
            abstract="A randomized controlled trial of parent training for children with ADHD.",
            authors=["Smith J", "Doe A"],
            doi="10.1234/same",
            publication_year=2023,
            publication_type="Randomized Controlled Trial",
            mesh_terms=["ADHD"],
        )
        parsed_oalex = ParsedDocument(
            source_id="W222",
            source_name="openalex",
            title="ADHD Parent Training RCT",
            abstract="A randomized controlled trial.",
            doi="10.1234/same",
            publication_year=2023,
            work_type="journal-article",
            concepts=["ADHD"],
        )
        parsed_eric = ParsedDocument(
            source_id="ED333",
            source_name="eric",
            title="Classroom ADHD Strategies Guide",
            abstract="Strategies for teachers.",
            subject_terms=["ADHD", "Classroom Strategies"],
            peer_reviewed=True,
        )

        # Transform
        doc1 = transformer.transform(parsed_pubmed)
        doc2 = transformer.transform(parsed_oalex)
        doc3 = transformer.transform(parsed_eric)

        # Dedup (pubmed first = higher priority)
        dedup = Deduplicator(title_threshold=0.85)
        all_docs = [doc1, doc2, doc3]
        deduped = dedup.deduplicate(all_docs)

        assert len(deduped) == 2  # pubmed + eric, openalex deduped by DOI
        ids = [d["id"] for d in deduped]
        assert "pmc_PMC111" in ids
        assert "eric_ED333" in ids
        assert "oalex_W222" not in ids

        # Write
        writer = BatchWriter(output_dir=str(tmp_path), batch_size=500)
        writer.write("pubmed", [d for d in deduped if d["id"].startswith("pmc_")])
        writer.write("eric", [d for d in deduped if d["id"].startswith("eric_")])

        pubmed_files = list((tmp_path / "pubmed").glob("*.json"))
        assert len(pubmed_files) == 1
        with open(pubmed_files[0]) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["id"] == "pmc_PMC111"
        assert data[0]["evidence_level"] == "strong"

    def test_schema_matches_existing_knowledge_format(self, tmp_path):
        """Verify output schema has all required fields matching existing curated docs."""
        transformer = SchemaTransformer()
        parsed = ParsedDocument(
            source_id="PMC999",
            source_name="pubmed",
            title="Test Study",
            abstract="Test abstract for ADHD in children.",
            doi="10.1234/test",
            publication_year=2024,
        )
        doc = transformer.transform(parsed)

        required_fields = [
            "id", "name", "description", "document_type", "tags", "age_range",
            "evidence_level", "source", "citations", "steps", "key_points",
            "contraindications", "related_ids",
        ]
        for field in required_fields:
            assert field in doc, f"Missing required field: {field}"

        # Type checks
        assert isinstance(doc["tags"], list)
        assert isinstance(doc["age_range"], list)
        assert isinstance(doc["citations"], list)
        assert isinstance(doc["steps"], list)
        assert isinstance(doc["key_points"], list)
        assert isinstance(doc["contraindications"], list)
        assert isinstance(doc["related_ids"], list)
        assert doc["document_type"] in ("fact", "guidance", "strategy")
        assert doc["evidence_level"] in ("strong", "moderate", "emerging", "expert_consensus")
