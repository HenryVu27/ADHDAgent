import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from scripts.ingest.sources.pubmed import PubMedConnector
from scripts.ingest.sources.openalex import OpenAlexConnector
from scripts.ingest.sources.semantic_scholar import SemanticScholarConnector
from scripts.ingest.sources.eric import ERICConnector
from scripts.ingest.sources.government import GovernmentConnector
from scripts.ingest.sources.base import SourceConnector
from scripts.ingest.models import RawDocument


class TestPubMedConnector:
    def test_implements_protocol(self):
        connector = PubMedConnector(api_key="test", rps=10.0)
        assert isinstance(connector, SourceConnector)

    def test_parse_jats_xml(self):
        connector = PubMedConnector(api_key="test", rps=10.0)
        xml_data = """
        <article>
          <front>
            <article-meta>
              <article-id pub-id-type="pmc">PMC123456</article-id>
              <article-id pub-id-type="doi">10.1234/test</article-id>
              <title-group>
                <article-title>ADHD Parent Training Study</article-title>
              </title-group>
              <contrib-group>
                <contrib contrib-type="author">
                  <name><surname>Smith</surname><given-names>John</given-names></name>
                </contrib>
                <contrib contrib-type="author">
                  <name><surname>Doe</surname><given-names>Jane</given-names></name>
                </contrib>
              </contrib-group>
              <pub-date pub-type="epub">
                <year>2023</year>
              </pub-date>
              <abstract>
                <p>This RCT examines parent training for preschool children with ADHD.</p>
              </abstract>
              <kwd-group kwd-group-type="MeSH">
                <kwd>Attention Deficit Disorder with Hyperactivity</kwd>
                <kwd>Parent-Child Relations</kwd>
              </kwd-group>
            </article-meta>
          </front>
        </article>
        """
        raw = RawDocument(
            source_id="PMC123456",
            source_name="pubmed",
            raw_data={"xml": xml_data},
            format="xml",
        )
        parsed = connector.parse(raw)
        assert parsed.title == "ADHD Parent Training Study"
        assert parsed.doi == "10.1234/test"
        assert "Smith John" in parsed.authors
        assert parsed.publication_year == 2023
        assert "Attention Deficit Disorder with Hyperactivity" in parsed.mesh_terms
        assert "This RCT examines" in parsed.abstract

    @pytest.mark.asyncio
    async def test_fetch_calls_api(self):
        connector = PubMedConnector(api_key="test", rps=10.0)

        mock_search_response = '{"esearchresult": {"idlist": ["PMC111", "PMC222"]}}'
        mock_fetch_response = """
        <pmc-articleset>
          <article>
            <front><article-meta>
              <article-id pub-id-type="pmc">PMC111</article-id>
              <title-group><article-title>Study 1</article-title></title-group>
              <abstract><p>Abstract 1</p></abstract>
            </article-meta></front>
          </article>
        </pmc-articleset>
        """

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_resp_search = AsyncMock()
            mock_resp_search.text = AsyncMock(return_value=mock_search_response)
            mock_resp_search.status = 200

            mock_resp_fetch = AsyncMock()
            mock_resp_fetch.text = AsyncMock(return_value=mock_fetch_response)
            mock_resp_fetch.status = 200

            mock_session.get = AsyncMock(side_effect=[mock_resp_search, mock_resp_fetch])

            results = await connector.fetch("ADHD", max_docs=10)
            assert len(results) >= 1


class TestOpenAlexConnector:
    def test_implements_protocol(self):
        connector = OpenAlexConnector(email="test@test.com", rps=10.0)
        assert isinstance(connector, SourceConnector)

    def test_reconstruct_abstract(self):
        connector = OpenAlexConnector(email="test@test.com", rps=10.0)
        inverted = {"ADHD": [0], "affects": [1], "children": [2], "globally": [3]}
        result = connector._reconstruct_abstract(inverted)
        assert result == "ADHD affects children globally"

    def test_reconstruct_abstract_empty(self):
        connector = OpenAlexConnector(email="test@test.com", rps=10.0)
        assert connector._reconstruct_abstract(None) == ""
        assert connector._reconstruct_abstract({}) == ""

    def test_parse_work(self):
        connector = OpenAlexConnector(email="test@test.com", rps=10.0)
        raw = RawDocument(
            source_id="W123",
            source_name="openalex",
            raw_data={
                "id": "https://openalex.org/W123",
                "title": "ADHD in School-Age Children",
                "abstract_inverted_index": {"ADHD": [0], "study": [1]},
                "authorships": [
                    {"author": {"display_name": "Smith J"}},
                    {"author": {"display_name": "Doe A"}},
                ],
                "doi": "https://doi.org/10.1234/test",
                "publication_year": 2024,
                "type": "journal-article",
                "concepts": [
                    {"display_name": "ADHD", "score": 0.9},
                    {"display_name": "Pediatrics", "score": 0.8},
                ],
            },
            format="json",
        )
        parsed = connector.parse(raw)
        assert parsed.title == "ADHD in School-Age Children"
        assert parsed.abstract == "ADHD study"
        assert parsed.doi == "10.1234/test"
        assert parsed.work_type == "journal-article"
        assert "ADHD" in parsed.concepts


class TestSemanticScholarConnector:
    def test_implements_protocol(self):
        connector = SemanticScholarConnector(api_key="test", rps=10.0)
        assert isinstance(connector, SourceConnector)

    def test_parse_paper_with_tldr(self):
        connector = SemanticScholarConnector(api_key="test", rps=10.0)
        raw = RawDocument(
            source_id="12345",
            source_name="semantic_scholar",
            raw_data={
                "corpusId": 12345,
                "title": "ADHD Family Dynamics",
                "abstract": "This paper examines family dynamics in ADHD households.",
                "tldr": {"text": "ADHD disrupts family functioning."},
                "authors": [{"name": "Smith J"}],
                "externalIds": {"DOI": "10.1234/test"},
                "year": 2023,
                "fieldsOfStudy": ["Psychology", "Medicine"],
                "citationCount": 42,
                "openAccessPdf": {"url": "https://example.com/paper.pdf"},
            },
            format="json",
        )
        parsed = connector.parse(raw)
        assert parsed.title == "ADHD Family Dynamics"
        assert parsed.tldr == "ADHD disrupts family functioning."
        assert parsed.doi == "10.1234/test"
        assert "Psychology" in parsed.fields_of_study

    def test_parse_paper_without_tldr(self):
        connector = SemanticScholarConnector(api_key="test", rps=10.0)
        raw = RawDocument(
            source_id="99999",
            source_name="semantic_scholar",
            raw_data={
                "corpusId": 99999,
                "title": "Some Study",
                "abstract": "An abstract.",
                "tldr": None,
                "authors": [],
                "externalIds": {},
                "year": 2022,
                "fieldsOfStudy": None,
                "citationCount": 0,
                "openAccessPdf": None,
            },
            format="json",
        )
        parsed = connector.parse(raw)
        assert parsed.tldr == ""
        assert parsed.fields_of_study == []


class TestERICConnector:
    def test_implements_protocol(self):
        connector = ERICConnector(rps=5.0)
        assert isinstance(connector, SourceConnector)

    def test_parse_document(self):
        connector = ERICConnector(rps=5.0)
        raw = RawDocument(
            source_id="ED123456",
            source_name="eric",
            raw_data={
                "id": "ED123456",
                "title": "ADHD Classroom Strategies for Elementary Teachers",
                "description": "This guide provides strategies for teachers working with ADHD students.",
                "subject": ["ADHD", "Classroom Strategies", "Elementary Education"],
                "peerreviewed": True,
                "url": "https://eric.ed.gov/?id=ED123456",
            },
            format="json",
        )
        parsed = connector.parse(raw)
        assert parsed.title == "ADHD Classroom Strategies for Elementary Teachers"
        assert parsed.peer_reviewed is True
        assert "ADHD" in parsed.subject_terms
        assert parsed.source_id == "ED123456"


class TestGovernmentConnector:
    def test_implements_protocol(self):
        connector = GovernmentConnector(rps=2.0)
        assert isinstance(connector, SourceConnector)

    def test_parse_html_page(self):
        connector = GovernmentConnector(rps=2.0)
        raw = RawDocument(
            source_id="cdc-adhd-treatment",
            source_name="government",
            raw_data={
                "url": "https://www.cdc.gov/adhd/treatment.html",
                "html": """
                <html><body>
                <h1>Treatment of ADHD</h1>
                <p>ADHD can be managed with behavioral therapy.</p>
                <ul>
                  <li>Try behavioral therapy first</li>
                  <li>Set up a daily routine</li>
                </ul>
                </body></html>
                """,
            },
            format="html",
        )
        parsed = connector.parse(raw)
        assert parsed.title == "Treatment of ADHD"
        assert parsed.has_lists is True
        assert len(parsed.list_items) == 2
        assert "Try behavioral therapy first" in parsed.list_items
        assert "cdc" in parsed.url.lower()

    def test_parse_html_page_no_lists(self):
        connector = GovernmentConnector(rps=2.0)
        raw = RawDocument(
            source_id="nimh-adhd-overview",
            source_name="government",
            raw_data={
                "url": "https://www.nimh.nih.gov/adhd/overview.html",
                "html": """
                <html><body>
                <h1>ADHD Overview</h1>
                <p>ADHD is a neurodevelopmental disorder.</p>
                <p>It affects attention and behavior.</p>
                </body></html>
                """,
            },
            format="html",
        )
        parsed = connector.parse(raw)
        assert parsed.has_lists is False
        assert parsed.list_items == []
        assert "ADHD is a neurodevelopmental disorder" in parsed.html_content
