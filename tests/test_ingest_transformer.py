import pytest
from scripts.ingest.models import ParsedDocument
from scripts.ingest.pipeline.transformer import SchemaTransformer


class TestAgeRangeInference:
    def test_preschool_keywords(self):
        t = SchemaTransformer()
        result = t.infer_age_range("Preschool ADHD interventions", "Study on toddler behavior")
        assert "preschool" in result

    def test_school_age_keywords(self):
        t = SchemaTransformer()
        result = t.infer_age_range("Elementary school ADHD", "Children ages 6-12")
        assert "school_age" in result

    def test_adolescent_keywords(self):
        t = SchemaTransformer()
        result = t.infer_age_range("Teen ADHD management", "Adolescent behavior")
        assert "adolescent" in result

    def test_multiple_ranges(self):
        t = SchemaTransformer()
        result = t.infer_age_range("ADHD in children and adolescents", "Preschool to teen")
        assert len(result) >= 2

    def test_no_match_returns_all(self):
        t = SchemaTransformer()
        result = t.infer_age_range("ADHD neuroimaging study", "Brain activation patterns")
        assert result == ["all"]


class TestEvidenceLevel:
    def test_rct_is_strong(self):
        t = SchemaTransformer()
        assert t.infer_evidence_level(publication_type="Randomized Controlled Trial") == "strong"

    def test_review_type_is_strong(self):
        t = SchemaTransformer()
        assert t.infer_evidence_level(work_type="review") == "strong"

    def test_journal_article_is_moderate(self):
        t = SchemaTransformer()
        assert t.infer_evidence_level(work_type="journal-article") == "moderate"

    def test_peer_reviewed_is_moderate(self):
        t = SchemaTransformer()
        assert t.infer_evidence_level(peer_reviewed=True) == "moderate"

    def test_default_is_emerging(self):
        t = SchemaTransformer()
        assert t.infer_evidence_level() == "emerging"


class TestTransform:
    def test_pubmed_document(self):
        t = SchemaTransformer()
        parsed = ParsedDocument(
            source_id="PMC123456",
            source_name="pubmed",
            title="Parent Training for ADHD in Preschoolers",
            abstract="A randomized controlled trial of parent training for preschool children with ADHD.",
            authors=["Smith J", "Doe A"],
            doi="10.1234/test",
            publication_year=2023,
            publication_type="Randomized Controlled Trial",
            mesh_terms=["Attention Deficit Disorder with Hyperactivity", "Parent-Child Relations"],
        )
        result = t.transform(parsed)
        assert result["id"] == "pmc_PMC123456"
        assert result["name"] == "Parent Training for ADHD in Preschoolers"
        assert result["document_type"] in ("fact", "guidance")
        assert "preschool" in result["age_range"]
        assert result["evidence_level"] == "strong"
        assert result["source"] == "Smith J et al. (2023)"
        assert len(result["key_points"]) > 0
        assert result["steps"] == []
        assert result["contraindications"] == []
        assert result["related_ids"] == []

    def test_openalex_document(self):
        t = SchemaTransformer()
        parsed = ParsedDocument(
            source_id="W123456",
            source_name="openalex",
            title="Executive Function Training in School-Age Children",
            abstract="A review of executive function interventions for school-age children with ADHD.",
            doi="10.5678/test",
            publication_year=2024,
            work_type="review",
            concepts=["ADHD", "Executive Function", "Children"],
        )
        result = t.transform(parsed)
        assert result["id"] == "oalex_W123456"
        assert result["evidence_level"] == "strong"
        assert "school_age" in result["age_range"]

    def test_semantic_scholar_with_tldr(self):
        t = SchemaTransformer()
        parsed = ParsedDocument(
            source_id="12345",
            source_name="semantic_scholar",
            title="ADHD and Family Functioning",
            abstract="This study examines the impact of ADHD on family dynamics.",
            tldr="ADHD significantly impacts family functioning and parent-child relationships.",
            doi="10.9999/test",
            publication_year=2022,
            work_type="journal-article",
        )
        result = t.transform(parsed)
        assert result["id"] == "s2_12345"
        assert result["description"] == "ADHD significantly impacts family functioning and parent-child relationships."
        assert result["key_points"][0] == "ADHD significantly impacts family functioning and parent-child relationships."

    def test_government_page_with_lists(self):
        t = SchemaTransformer()
        parsed = ParsedDocument(
            source_id="cdc-adhd-treatment",
            source_name="government",
            title="Treatment of ADHD",
            abstract="",
            html_content="Treatment options for children with ADHD include behavioral therapy and medication.",
            has_lists=True,
            list_items=["Try behavioral therapy first", "Set up a daily routine", "Work with the school"],
        )
        result = t.transform(parsed)
        assert result["id"] == "gov_cdc-adhd-treatment"
        assert result["document_type"] == "strategy"
        assert result["steps"] == ["Try behavioral therapy first", "Set up a daily routine", "Work with the school"]
        assert result["evidence_level"] == "expert_consensus"
