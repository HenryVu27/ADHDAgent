# Data Ingestion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone CLI pipeline that fetches ADHD-related documents from 5 authoritative sources, filters for relevance via Gemini Flash, deduplicates across sources, and writes JSON files matching the existing knowledge base schema.

**Architecture:** Each source has its own async connector module. A shared pipeline handles transformation (to JSON schema), LLM-based relevance filtering, and cross-source deduplication. Output is JSON files in `app/knowledge/<source>/` subdirectories. The pipeline is fully isolated from the app runtime -- no `app.*` imports.

**Tech Stack:** Python 3.12, aiohttp, LlamaIndex (web readers), google-genai (Gemini Flash), nltk (sentence tokenization), beautifulsoup4

**Spec:** `docs/superpowers/specs/2026-03-19-ingestion-pipeline-design.md`

---

## File Map

```
scripts/
  __init__.py
  ingest/
    __init__.py
    main.py                     # CLI entry point (argparse)
    config.py                   # Pydantic Settings config (loads .env)
    gemini_client.py            # Standalone Gemini Flash wrapper
    rate_limiter.py             # asyncio.Semaphore-based rate limiter
    models.py                   # RawDocument, ParsedDocument dataclasses

    sources/
      __init__.py
      base.py                   # SourceConnector Protocol
      pubmed.py                 # PubMed E-Utilities connector
      openalex.py               # OpenAlex REST API connector
      semantic_scholar.py       # Semantic Scholar Graph API connector
      eric.py                   # ERIC REST API connector
      government.py             # CDC/NIH/NIMH HTML crawler

    pipeline/
      __init__.py
      transformer.py            # ParsedDocument -> knowledge base JSON schema
      filter.py                 # Gemini Flash batch relevance scoring
      deduplicator.py           # DOI + title similarity dedup

    output/
      __init__.py
      writer.py                 # Batch JSON file writer
      cache.py                  # Raw response cache for resumability

tests/
  test_ingest_transformer.py    # Transformer unit tests
  test_ingest_filter.py         # Filter unit tests
  test_ingest_dedup.py          # Deduplicator unit tests
  test_ingest_sources.py        # Source connector unit tests (mocked HTTP)
  test_ingest_writer.py         # Writer unit tests
  test_ingest_integration.py    # End-to-end pipeline test (mocked APIs)
```

---

### Task 1: Project scaffolding and config

**Files:**
- Create: `scripts/__init__.py`, `scripts/ingest/__init__.py`, `scripts/ingest/config.py`, `scripts/ingest/models.py`, `scripts/ingest/rate_limiter.py`, `scripts/ingest/requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create directory structure and `__init__.py` files**

```bash
mkdir -p scripts/ingest/sources scripts/ingest/pipeline scripts/ingest/output
touch scripts/__init__.py scripts/ingest/__init__.py scripts/ingest/sources/__init__.py scripts/ingest/pipeline/__init__.py scripts/ingest/output/__init__.py
```

- [ ] **Step 2: Write `scripts/ingest/requirements.txt`**

```
llama-index-core>=0.12.0
llama-index-readers-web>=0.3.0
beautifulsoup4>=4.12.0
aiohttp>=3.9.0
tenacity>=9.0.0
google-genai>=1.0.0
nltk>=3.9.0
pydantic-settings>=2.0.0
```

- [ ] **Step 3: Install dependencies**

Run: `./adhd312/Scripts/pip.exe install -r scripts/ingest/requirements.txt`

- [ ] **Step 4: Write `scripts/ingest/config.py`**

```python
from pydantic_settings import BaseSettings


class IngestConfig(BaseSettings):
    # API keys
    ncbi_api_key: str = ""
    s2_api_key: str = ""
    openalex_email: str = ""
    gemini_api_key: str = ""

    # Pipeline
    relevance_threshold: int = 6
    batch_size: int = 500
    dedup_title_threshold: float = 0.85
    filter_batch_size: int = 20

    # Rate limiting (requests per second)
    pubmed_rps: float = 10.0
    openalex_rps: float = 10.0
    s2_rps: float = 10.0
    eric_rps: float = 5.0
    gov_rps: float = 2.0

    # Paths
    output_dir: str = "app/knowledge"
    raw_cache_dir: str = "scripts/ingest/.cache"

    # Default queries
    default_queries: list[str] = [
        "ADHD parenting strategies",
        "attention deficit hyperactivity disorder child behavior",
        "behavioral intervention pediatric ADHD",
        "executive function child intervention",
        "parent training ADHD",
        "ADHD school accommodations",
        "ADHD family support",
    ]

    model_config = {"env_file": ".env", "env_prefix": "INGEST_"}
```

- [ ] **Step 5: Write `scripts/ingest/models.py`**

```python
from dataclasses import dataclass, field


@dataclass
class RawDocument:
    """Raw document as fetched from a source API."""
    source_id: str
    source_name: str
    raw_data: dict
    format: str  # "xml", "json", "html"


@dataclass
class ParsedDocument:
    """Document after parsing, before schema transformation."""
    source_id: str
    source_name: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    publication_year: int = 0
    publication_type: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    subject_terms: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)
    tldr: str = ""
    work_type: str = ""  # OpenAlex/S2 type field (review, journal-article, etc.)
    peer_reviewed: bool = False
    url: str = ""
    html_content: str = ""  # For government pages
    has_lists: bool = False  # For government pages with step-like content
    list_items: list[str] = field(default_factory=list)
```

- [ ] **Step 6: Write `scripts/ingest/rate_limiter.py`**

```python
import asyncio


class RateLimiter:
    """Async rate limiter using semaphore with timed release."""

    def __init__(self, rps: float):
        self._semaphore = asyncio.Semaphore(max(1, int(rps)))
        self._interval = 1.0 / rps

    async def acquire(self):
        await self._semaphore.acquire()
        loop = asyncio.get_event_loop()
        loop.call_later(self._interval, self._semaphore.release)
```

- [ ] **Step 7: Add `.cache` to `.gitignore`**

Append `scripts/ingest/.cache/` to `.gitignore`.

- [ ] **Step 8: Commit**

```bash
git add scripts/ .gitignore
git commit -m "Add ingestion pipeline scaffolding, config, models, rate limiter"
```

---

### Task 2: Gemini Flash client and relevance filter

**Files:**
- Create: `scripts/ingest/gemini_client.py`, `scripts/ingest/pipeline/filter.py`
- Test: `tests/test_ingest_filter.py`

- [ ] **Step 1: Write the failing test for `gemini_client.py`**

```python
# tests/test_ingest_filter.py
import pytest
from unittest.mock import AsyncMock, patch
from scripts.ingest.gemini_client import GeminiClient
from scripts.ingest.pipeline.filter import RelevanceFilter


@pytest.mark.asyncio
async def test_gemini_client_score_batch():
    client = GeminiClient(api_key="test-key")
    mock_response = '[{"index": 1, "score": 8, "reason": "relevant"}]'
    with patch.object(client, "_call_flash", new_callable=AsyncMock, return_value=mock_response):
        results = await client.score_relevance([{
            "title": "ADHD Parent Training",
            "abstract": "A study on parent training for ADHD.",
            "tags": ["ADHD", "parenting"],
        }])
    assert len(results) == 1
    assert results[0]["score"] == 8


@pytest.mark.asyncio
async def test_relevance_filter_keeps_above_threshold():
    docs = [
        {"id": "doc1", "name": "High relevance", "description": "ADHD parenting"},
        {"id": "doc2", "name": "Low relevance", "description": "Quantum physics"},
    ]
    mock_client = AsyncMock()
    mock_client.score_relevance.return_value = [
        {"index": 1, "score": 8, "reason": "relevant"},
        {"index": 2, "score": 2, "reason": "not relevant"},
    ]
    filt = RelevanceFilter(client=mock_client, threshold=6, batch_size=20)
    kept = await filt.filter(docs)
    assert len(kept) == 1
    assert kept[0]["id"] == "doc1"


@pytest.mark.asyncio
async def test_relevance_filter_empty_input():
    mock_client = AsyncMock()
    filt = RelevanceFilter(client=mock_client, threshold=6, batch_size=20)
    kept = await filt.filter([])
    assert kept == []
    mock_client.score_relevance.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_filter.py -v`
Expected: FAIL (modules not found)

- [ ] **Step 3: Write `scripts/ingest/gemini_client.py`**

```python
import json
import google.genai as genai
from tenacity import retry, stop_after_attempt, wait_exponential


class GeminiClient:
    """Standalone Gemini Flash client for relevance scoring."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_flash(self, prompt: str) -> str:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text

    async def score_relevance(self, docs: list[dict]) -> list[dict]:
        """Score a batch of documents for ADHD parenting relevance (0-10)."""
        if not docs:
            return []

        doc_lines = []
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")[:500]
            tags = ", ".join(doc.get("tags", []))
            doc_lines.append(f"[{i}] Title: {title} | Abstract: {abstract} | Tags: {tags}")

        prompt = (
            "Rate each document's relevance to an ADHD parenting coaching chatbot (0-10):\n"
            "- 9-10: Directly about ADHD parenting strategies, child behavior management, or family support\n"
            "- 7-8: About ADHD in children/adolescents with practical implications for parents\n"
            "- 5-6: About ADHD generally (neuroscience, adult ADHD, pharmacology) or general parenting\n"
            "- 3-4: Tangentially related (general child psychology, education theory)\n"
            "- 0-2: Not relevant\n\n"
            "Documents:\n" + "\n".join(doc_lines) + "\n\n"
            'Return JSON array: [{"index": int, "score": int, "reason": str}, ...]'
        )

        raw = await self._call_flash(prompt)
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
```

- [ ] **Step 4: Write `scripts/ingest/pipeline/filter.py`**

```python
from scripts.ingest.gemini_client import GeminiClient


class RelevanceFilter:
    """Filters documents by Gemini Flash relevance score."""

    def __init__(self, client: GeminiClient, threshold: int = 6, batch_size: int = 20):
        self._client = client
        self._threshold = threshold
        self._batch_size = batch_size

    async def filter(self, docs: list[dict]) -> list[dict]:
        """Return only documents scoring >= threshold."""
        if not docs:
            return []

        kept = []
        for i in range(0, len(docs), self._batch_size):
            batch = docs[i:i + self._batch_size]
            batch_for_scoring = [
                {
                    "title": d.get("name", ""),
                    "abstract": d.get("description", ""),
                    "tags": d.get("tags", []),
                }
                for d in batch
            ]
            scores = await self._client.score_relevance(batch_for_scoring)

            score_map = {s["index"]: s["score"] for s in scores}
            for j, doc in enumerate(batch, 1):
                if score_map.get(j, 0) >= self._threshold:
                    kept.append(doc)

        return kept
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_filter.py -v`
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add scripts/ingest/gemini_client.py scripts/ingest/pipeline/filter.py tests/test_ingest_filter.py
git commit -m "Add Gemini Flash client and relevance filter with tests"
```

---

### Task 3: Schema transformer

**Files:**
- Create: `scripts/ingest/pipeline/transformer.py`
- Test: `tests/test_ingest_transformer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingest_transformer.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_transformer.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write `scripts/ingest/pipeline/transformer.py`**

```python
import re
import nltk
from scripts.ingest.models import ParsedDocument

nltk.download("punkt_tab", quiet=True)


class SchemaTransformer:
    """Transforms ParsedDocument into the knowledge base JSON schema."""

    # Age range keyword patterns
    _AGE_PATTERNS = {
        "preschool": re.compile(
            r"\b(preschool|pre-k|pre-school|early childhood|toddler|ages?\s*3[-\s]?5|young children)\b", re.I
        ),
        "school_age": re.compile(
            r"\b(school[ -]?age|elementary|ages?\s*6[-\s]?12|child(?:ren)?(?!\s+and\s+adolescent))\b", re.I
        ),
        "adolescent": re.compile(
            r"\b(adolescen|teen|ages?\s*13[-\s]?17|high\s+school|youth|young\s+adult)\b", re.I
        ),
    }

    # Publication types that indicate strong evidence
    _STRONG_PUB_TYPES = {
        "randomized controlled trial", "meta-analysis", "systematic review",
        "review", "clinical trial", "practice guideline",
    }

    # Publication types for fact vs guidance
    _FACT_PUB_TYPES = {
        "randomized controlled trial", "meta-analysis", "systematic review",
        "clinical trial", "observational study", "cohort study",
    }

    # ERIC subject terms that indicate strategy documents
    _STRATEGY_TERMS = {"strategies", "interventions", "programs", "techniques", "methods", "training"}

    def infer_age_range(self, title: str, abstract: str) -> list[str]:
        text = f"{title} {abstract}"
        ranges = []
        for age_key, pattern in self._AGE_PATTERNS.items():
            if pattern.search(text):
                ranges.append(age_key)
        return ranges if ranges else ["all"]

    def infer_evidence_level(
        self,
        publication_type: str = "",
        work_type: str = "",
        peer_reviewed: bool = False,
        is_government: bool = False,
    ) -> str:
        if is_government:
            return "expert_consensus"

        pub_lower = publication_type.lower()
        work_lower = work_type.lower()

        if pub_lower in self._STRONG_PUB_TYPES or work_lower in {"review", "meta-analysis"}:
            return "strong"
        if work_lower == "journal-article" or peer_reviewed:
            return "moderate"
        if pub_lower:
            return "moderate"
        return "emerging"

    def _infer_document_type(self, parsed: ParsedDocument) -> str:
        if parsed.source_name == "government":
            return "strategy" if parsed.has_lists else "guidance"
        if parsed.source_name == "eric":
            terms_lower = {t.lower() for t in parsed.subject_terms}
            if terms_lower & self._STRATEGY_TERMS:
                return "strategy"
            return "guidance"
        # PubMed, OpenAlex, Semantic Scholar
        pub_lower = parsed.publication_type.lower()
        if pub_lower in self._FACT_PUB_TYPES:
            return "fact"
        if "guideline" in pub_lower or "recommendation" in pub_lower:
            return "guidance"
        return "fact"

    def _build_tags(self, parsed: ParsedDocument) -> list[str]:
        tags = []
        if parsed.mesh_terms:
            tags.extend(t.lower().replace(" ", "_") for t in parsed.mesh_terms)
        if parsed.concepts:
            tags.extend(t.lower().replace(" ", "_") for t in parsed.concepts)
        if parsed.fields_of_study:
            tags.extend(t.lower().replace(" ", "_") for t in parsed.fields_of_study)
        if parsed.subject_terms:
            tags.extend(t.lower().replace(" ", "_") for t in parsed.subject_terms)
        return list(dict.fromkeys(tags))  # deduplicate, preserve order

    def _build_source(self, parsed: ParsedDocument) -> str:
        if parsed.source_name == "government":
            if "cdc" in parsed.url.lower():
                return "CDC"
            if "nimh" in parsed.url.lower():
                return "NIMH"
            return "NIH"
        if parsed.authors:
            first = parsed.authors[0]
            suffix = " et al." if len(parsed.authors) > 1 else ""
            year = f" ({parsed.publication_year})" if parsed.publication_year else ""
            return f"{first}{suffix}{year}"
        return parsed.source_name

    def _build_key_points(self, parsed: ParsedDocument) -> list[str]:
        if parsed.source_name == "government" and not parsed.has_lists:
            # Split HTML content into paragraphs
            text = parsed.html_content or parsed.abstract
            if not text:
                return []
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            return paragraphs[:10]

        points = []
        if parsed.tldr:
            points.append(parsed.tldr)
        if parsed.abstract:
            sentences = nltk.sent_tokenize(parsed.abstract)
            points.extend(sentences)
        return points[:10]

    def _build_id(self, parsed: ParsedDocument) -> str:
        prefixes = {
            "pubmed": "pmc_",
            "openalex": "oalex_",
            "semantic_scholar": "s2_",
            "eric": "eric_",
            "government": "gov_",
        }
        prefix = prefixes.get(parsed.source_name, "")
        return f"{prefix}{parsed.source_id}"

    def _build_description(self, parsed: ParsedDocument) -> str:
        if parsed.tldr:
            return parsed.tldr
        text = parsed.abstract or parsed.html_content or ""
        return text[:200].strip()

    def transform(self, parsed: ParsedDocument) -> dict:
        return {
            "id": self._build_id(parsed),
            "name": parsed.title,
            "description": self._build_description(parsed),
            "document_type": self._infer_document_type(parsed),
            "tags": self._build_tags(parsed),
            "age_range": self.infer_age_range(parsed.title, parsed.abstract or parsed.html_content or ""),
            "evidence_level": self.infer_evidence_level(
                publication_type=parsed.publication_type,
                work_type=parsed.work_type,
                peer_reviewed=parsed.peer_reviewed,
                is_government=parsed.source_name == "government",
            ),
            "source": self._build_source(parsed),
            "citations": [
                {
                    "source_file": "",
                    "source_name": self._build_source(parsed),
                    "detail": parsed.doi or parsed.url or parsed.source_id,
                }
            ],
            "steps": parsed.list_items if parsed.has_lists else [],
            "key_points": self._build_key_points(parsed),
            "contraindications": [],
            "related_ids": [],
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_transformer.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest/pipeline/transformer.py tests/test_ingest_transformer.py
git commit -m "Add schema transformer with age range, evidence level, and document type inference"
```

---

### Task 4: Deduplicator

**Files:**
- Create: `scripts/ingest/pipeline/deduplicator.py`
- Test: `tests/test_ingest_dedup.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingest_dedup.py
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
        # Identical after normalization
        assert dedup._jaccard_similarity("ADHD Parent Training", "adhd parent training") == 1.0
        # Completely different
        assert dedup._jaccard_similarity("ADHD", "Quantum Physics") == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_dedup.py -v`
Expected: FAIL

- [ ] **Step 3: Write `scripts/ingest/pipeline/deduplicator.py`**

```python
import re


class Deduplicator:
    """Deduplicates documents across sources by DOI and title similarity."""

    def __init__(self, title_threshold: float = 0.85):
        self._title_threshold = title_threshold

    def _normalize_title(self, title: str) -> set[str]:
        text = re.sub(r"[^\w\s]", "", title.lower())
        return set(text.split())

    def _jaccard_similarity(self, title_a: str, title_b: str) -> float:
        tokens_a = self._normalize_title(title_a)
        tokens_b = self._normalize_title(title_b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def _extract_doi(self, doc: dict) -> str:
        citations = doc.get("citations", [])
        for c in citations:
            detail = c.get("detail", "")
            if detail.startswith("10."):
                return detail
        return ""

    def deduplicate(self, docs: list[dict]) -> list[dict]:
        """Remove duplicates. Input should be ordered by source priority
        (PubMed first, then Semantic Scholar, OpenAlex, ERIC, CDC/NIH)
        so first-seen wins."""
        if not docs:
            return []

        seen_dois: set[str] = set()
        seen_titles: list[tuple[str, str]] = []  # (normalized_title_str, id)
        kept: list[dict] = []

        for doc in docs:
            doi = self._extract_doi(doc)
            if doi:
                if doi in seen_dois:
                    continue
                seen_dois.add(doi)

            # Title similarity check
            title = doc.get("name", "")
            is_dup = False
            for seen_title, _ in seen_titles:
                if self._jaccard_similarity(title, seen_title) >= self._title_threshold:
                    is_dup = True
                    break

            if not is_dup:
                seen_titles.append((title, doc["id"]))
                kept.append(doc)

        return kept
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_dedup.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest/pipeline/deduplicator.py tests/test_ingest_dedup.py
git commit -m "Add cross-source deduplicator with DOI and title similarity"
```

---

### Task 5: JSON writer and raw cache

**Files:**
- Create: `scripts/ingest/output/writer.py`, `scripts/ingest/output/cache.py`
- Test: `tests/test_ingest_writer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingest_writer.py
import json
import pytest
from pathlib import Path
from scripts.ingest.output.writer import BatchWriter
from scripts.ingest.output.cache import RawCache


class TestBatchWriter:
    def test_writes_batch_files(self, tmp_path):
        writer = BatchWriter(output_dir=str(tmp_path), batch_size=2)
        docs = [
            {"id": "doc1", "name": "Doc 1"},
            {"id": "doc2", "name": "Doc 2"},
            {"id": "doc3", "name": "Doc 3"},
        ]
        writer.write("pubmed", docs)

        pubmed_dir = tmp_path / "pubmed"
        assert pubmed_dir.exists()
        files = sorted(pubmed_dir.glob("*.json"))
        assert len(files) == 2  # 2 docs + 1 doc = 2 files

        with open(files[0]) as f:
            batch1 = json.load(f)
        assert len(batch1) == 2

        with open(files[1]) as f:
            batch2 = json.load(f)
        assert len(batch2) == 1

    def test_overwrites_existing_directory(self, tmp_path):
        writer = BatchWriter(output_dir=str(tmp_path), batch_size=500)
        # Write once
        writer.write("pubmed", [{"id": "old"}])
        # Write again (should overwrite)
        writer.write("pubmed", [{"id": "new"}])

        files = list((tmp_path / "pubmed").glob("*.json"))
        assert len(files) == 1
        with open(files[0]) as f:
            data = json.load(f)
        assert data[0]["id"] == "new"

    def test_empty_docs_creates_empty_dir(self, tmp_path):
        writer = BatchWriter(output_dir=str(tmp_path), batch_size=500)
        writer.write("pubmed", [])
        assert (tmp_path / "pubmed").exists()
        assert list((tmp_path / "pubmed").glob("*.json")) == []


class TestRawCache:
    def test_save_and_load(self, tmp_path):
        cache = RawCache(cache_dir=str(tmp_path))
        cache.save("pubmed", "doc1", {"title": "Test"})
        result = cache.load("pubmed", "doc1")
        assert result == {"title": "Test"}

    def test_load_missing_returns_none(self, tmp_path):
        cache = RawCache(cache_dir=str(tmp_path))
        assert cache.load("pubmed", "missing") is None

    def test_has(self, tmp_path):
        cache = RawCache(cache_dir=str(tmp_path))
        cache.save("pubmed", "doc1", {"title": "Test"})
        assert cache.has("pubmed", "doc1")
        assert not cache.has("pubmed", "doc2")

    def test_list_cached_ids(self, tmp_path):
        cache = RawCache(cache_dir=str(tmp_path))
        cache.save("pubmed", "doc1", {})
        cache.save("pubmed", "doc2", {})
        ids = cache.list_ids("pubmed")
        assert sorted(ids) == ["doc1", "doc2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_writer.py -v`
Expected: FAIL

- [ ] **Step 3: Write `scripts/ingest/output/writer.py`**

```python
import json
import shutil
from pathlib import Path


class BatchWriter:
    """Writes documents as batched JSON files to app/knowledge/<source>/."""

    def __init__(self, output_dir: str, batch_size: int = 500):
        self._output_dir = Path(output_dir)
        self._batch_size = batch_size

    def write(self, source_name: str, docs: list[dict]) -> list[Path]:
        """Write docs to source subdirectory. Overwrites existing files for this source."""
        source_dir = self._output_dir / source_name

        # Overwrite: remove existing source directory
        if source_dir.exists():
            shutil.rmtree(source_dir)
        source_dir.mkdir(parents=True, exist_ok=True)

        written_files = []
        for i in range(0, max(len(docs), 1), self._batch_size):
            batch = docs[i:i + self._batch_size]
            if not batch:
                break
            batch_num = (i // self._batch_size) + 1
            file_path = source_dir / f"{source_name}_batch_{batch_num:03d}.json"
            with open(file_path, "w") as f:
                json.dump(batch, f, indent=2)
            written_files.append(file_path)

        return written_files
```

- [ ] **Step 4: Write `scripts/ingest/output/cache.py`**

```python
import json
from pathlib import Path


class RawCache:
    """Caches raw API responses for resumability."""

    def __init__(self, cache_dir: str):
        self._cache_dir = Path(cache_dir)

    def _path(self, source: str, doc_id: str) -> Path:
        return self._cache_dir / source / f"{doc_id}.json"

    def save(self, source: str, doc_id: str, data: dict) -> None:
        path = self._path(source, doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, source: str, doc_id: str) -> dict | None:
        path = self._path(source, doc_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def has(self, source: str, doc_id: str) -> bool:
        return self._path(source, doc_id).exists()

    def list_ids(self, source: str) -> list[str]:
        source_dir = self._cache_dir / source
        if not source_dir.exists():
            return []
        return [p.stem for p in source_dir.glob("*.json")]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_writer.py -v`
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add scripts/ingest/output/writer.py scripts/ingest/output/cache.py tests/test_ingest_writer.py
git commit -m "Add batch JSON writer and raw response cache"
```

---

### Task 6: Source connector protocol and PubMed connector

**Files:**
- Create: `scripts/ingest/sources/base.py`, `scripts/ingest/sources/pubmed.py`
- Test: `tests/test_ingest_sources.py`

- [ ] **Step 1: Write `scripts/ingest/sources/base.py`**

```python
from typing import Protocol, runtime_checkable
from scripts.ingest.models import RawDocument, ParsedDocument


@runtime_checkable
class SourceConnector(Protocol):
    source_name: str

    async def fetch(self, query: str, max_docs: int) -> list[RawDocument]:
        ...

    def parse(self, raw: RawDocument) -> ParsedDocument:
        ...
```

- [ ] **Step 2: Write the failing tests for PubMed**

```python
# tests/test_ingest_sources.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from scripts.ingest.sources.pubmed import PubMedConnector
from scripts.ingest.sources.base import SourceConnector


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
        from scripts.ingest.models import RawDocument
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_sources.py -v`
Expected: FAIL

- [ ] **Step 4: Write `scripts/ingest/sources/pubmed.py`**

```python
import json
import xml.etree.ElementTree as ET
import aiohttp
from scripts.ingest.models import RawDocument, ParsedDocument
from scripts.ingest.rate_limiter import RateLimiter

E_UTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedConnector:
    source_name = "pubmed"

    def __init__(self, api_key: str, rps: float = 10.0):
        self._api_key = api_key
        self._limiter = RateLimiter(rps)

    async def fetch(self, query: str, max_docs: int = 10000) -> list[RawDocument]:
        raw_docs = []
        async with aiohttp.ClientSession() as session:
            # Step 1: Search for PMC IDs
            ids = await self._search(session, query, max_docs)
            # Step 2: Fetch XML in batches of 200
            for i in range(0, len(ids), 200):
                batch_ids = ids[i:i + 200]
                xml_text = await self._fetch_batch(session, batch_ids)
                raw_docs.extend(self._split_articles(xml_text))
        return raw_docs

    async def _search(self, session: aiohttp.ClientSession, query: str, max_docs: int) -> list[str]:
        await self._limiter.acquire()
        params = {
            "db": "pmc",
            "term": query,
            "retmax": min(max_docs, 100000),
            "rettype": "json",
            "api_key": self._api_key,
        }
        resp = await session.get(f"{E_UTILS_BASE}/esearch.fcgi", params=params)
        text = await resp.text()
        data = json.loads(text)
        return data.get("esearchresult", {}).get("idlist", [])

    async def _fetch_batch(self, session: aiohttp.ClientSession, ids: list[str]) -> str:
        await self._limiter.acquire()
        params = {
            "db": "pmc",
            "id": ",".join(ids),
            "rettype": "xml",
            "api_key": self._api_key,
        }
        resp = await session.get(f"{E_UTILS_BASE}/efetch.fcgi", params=params)
        return await resp.text()

    def _split_articles(self, xml_text: str) -> list[RawDocument]:
        docs = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return docs

        for article in root.iter("article"):
            pmc_id = ""
            meta = article.find(".//article-meta")
            if meta is not None:
                for aid in meta.findall("article-id"):
                    if aid.get("pub-id-type") == "pmc":
                        pmc_id = aid.text or ""
                        break

            if not pmc_id:
                continue

            docs.append(RawDocument(
                source_id=pmc_id,
                source_name="pubmed",
                raw_data={"xml": ET.tostring(article, encoding="unicode")},
                format="xml",
            ))
        return docs

    def parse(self, raw: RawDocument) -> ParsedDocument:
        xml_str = raw.raw_data.get("xml", "")
        root = ET.fromstring(xml_str)
        meta = root.find(".//article-meta")

        title = ""
        abstract = ""
        authors = []
        doi = ""
        year = 0
        mesh_terms = []
        pub_type = ""

        if meta is not None:
            # Title
            title_el = meta.find(".//article-title")
            if title_el is not None:
                title = "".join(title_el.itertext()).strip()

            # DOI
            for aid in meta.findall("article-id"):
                if aid.get("pub-id-type") == "doi":
                    doi = aid.text or ""

            # Authors
            for contrib in meta.findall(".//contrib[@contrib-type='author']"):
                name_el = contrib.find("name")
                if name_el is not None:
                    surname = name_el.findtext("surname", "")
                    given = name_el.findtext("given-names", "")
                    authors.append(f"{surname} {given}".strip())

            # Year
            for pub_date in meta.findall("pub-date"):
                year_el = pub_date.find("year")
                if year_el is not None and year_el.text:
                    year = int(year_el.text)
                    break

            # Abstract
            abs_el = meta.find("abstract")
            if abs_el is not None:
                abstract = " ".join(abs_el.itertext()).strip()

            # MeSH terms
            for kwd_group in meta.findall(".//kwd-group"):
                if "MeSH" in (kwd_group.get("kwd-group-type", "")):
                    for kwd in kwd_group.findall("kwd"):
                        if kwd.text:
                            mesh_terms.append(kwd.text.strip())

        return ParsedDocument(
            source_id=raw.source_id,
            source_name="pubmed",
            title=title,
            abstract=abstract,
            authors=authors,
            doi=doi,
            publication_year=year,
            publication_type=pub_type,
            mesh_terms=mesh_terms,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_sources.py -v`
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add scripts/ingest/sources/base.py scripts/ingest/sources/pubmed.py tests/test_ingest_sources.py
git commit -m "Add source connector protocol and PubMed connector"
```

---

### Task 7: OpenAlex connector

**Files:**
- Create: `scripts/ingest/sources/openalex.py`
- Modify: `tests/test_ingest_sources.py`

- [ ] **Step 1: Add failing tests for OpenAlex**

Append to `tests/test_ingest_sources.py`:

```python
from scripts.ingest.sources.openalex import OpenAlexConnector


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
        from scripts.ingest.models import RawDocument
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_sources.py::TestOpenAlexConnector -v`
Expected: FAIL

- [ ] **Step 3: Write `scripts/ingest/sources/openalex.py`**

```python
import aiohttp
from scripts.ingest.models import RawDocument, ParsedDocument
from scripts.ingest.rate_limiter import RateLimiter

OPENALEX_BASE = "https://api.openalex.org"


class OpenAlexConnector:
    source_name = "openalex"

    def __init__(self, email: str, rps: float = 10.0):
        self._email = email
        self._limiter = RateLimiter(rps)

    def _reconstruct_abstract(self, inverted_index: dict | None) -> str:
        if not inverted_index:
            return ""
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(w for _, w in word_positions)

    async def fetch(self, query: str, max_docs: int = 10000) -> list[RawDocument]:
        raw_docs = []
        cursor = "*"
        async with aiohttp.ClientSession() as session:
            while cursor and len(raw_docs) < max_docs:
                await self._limiter.acquire()
                params = {
                    "search": query,
                    "per_page": 200,
                    "cursor": cursor,
                    "mailto": self._email,
                }
                resp = await session.get(f"{OPENALEX_BASE}/works", params=params)
                data = await resp.json()

                works = data.get("results", [])
                if not works:
                    break

                for work in works:
                    work_id = work.get("id", "").split("/")[-1]
                    if not work_id:
                        continue
                    raw_docs.append(RawDocument(
                        source_id=work_id,
                        source_name="openalex",
                        raw_data=work,
                        format="json",
                    ))

                cursor = data.get("meta", {}).get("next_cursor")

                if len(raw_docs) >= max_docs:
                    break

        return raw_docs[:max_docs]

    def parse(self, raw: RawDocument) -> ParsedDocument:
        data = raw.raw_data

        doi = data.get("doi", "") or ""
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]

        authors = [
            a["author"]["display_name"]
            for a in data.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]

        concepts = [
            c["display_name"]
            for c in sorted(data.get("concepts", []), key=lambda x: x.get("score", 0), reverse=True)[:5]
            if c.get("display_name")
        ]

        return ParsedDocument(
            source_id=raw.source_id,
            source_name="openalex",
            title=data.get("title", "") or "",
            abstract=self._reconstruct_abstract(data.get("abstract_inverted_index")),
            authors=authors,
            doi=doi,
            publication_year=data.get("publication_year", 0) or 0,
            work_type=data.get("type", "") or "",
            concepts=concepts,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_sources.py::TestOpenAlexConnector -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest/sources/openalex.py tests/test_ingest_sources.py
git commit -m "Add OpenAlex connector with abstract reconstruction"
```

---

### Task 8: Semantic Scholar connector

**Files:**
- Create: `scripts/ingest/sources/semantic_scholar.py`
- Modify: `tests/test_ingest_sources.py`

- [ ] **Step 1: Add failing tests for Semantic Scholar**

Append to `tests/test_ingest_sources.py`:

```python
from scripts.ingest.sources.semantic_scholar import SemanticScholarConnector


class TestSemanticScholarConnector:
    def test_implements_protocol(self):
        connector = SemanticScholarConnector(api_key="test", rps=10.0)
        assert isinstance(connector, SourceConnector)

    def test_parse_paper_with_tldr(self):
        connector = SemanticScholarConnector(api_key="test", rps=10.0)
        from scripts.ingest.models import RawDocument
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
        from scripts.ingest.models import RawDocument
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_sources.py::TestSemanticScholarConnector -v`
Expected: FAIL

- [ ] **Step 3: Write `scripts/ingest/sources/semantic_scholar.py`**

```python
import aiohttp
from scripts.ingest.models import RawDocument, ParsedDocument
from scripts.ingest.rate_limiter import RateLimiter

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,abstract,tldr,authors,externalIds,year,fieldsOfStudy,citationCount,openAccessPdf"


class SemanticScholarConnector:
    source_name = "semantic_scholar"

    def __init__(self, api_key: str, rps: float = 10.0):
        self._api_key = api_key
        self._limiter = RateLimiter(rps)

    async def fetch(self, query: str, max_docs: int = 5000) -> list[RawDocument]:
        raw_docs = []
        async with aiohttp.ClientSession() as session:
            offset = 0
            while offset < max_docs:
                await self._limiter.acquire()
                limit = min(100, max_docs - offset)
                params = {
                    "query": query,
                    "limit": limit,
                    "offset": offset,
                    "fields": S2_FIELDS,
                }
                headers = {}
                if self._api_key:
                    headers["x-api-key"] = self._api_key

                resp = await session.get(
                    f"{S2_BASE}/paper/search", params=params, headers=headers
                )
                data = await resp.json()

                papers = data.get("data", [])
                if not papers:
                    break

                for paper in papers:
                    corpus_id = str(paper.get("corpusId", ""))
                    if not corpus_id:
                        continue
                    raw_docs.append(RawDocument(
                        source_id=corpus_id,
                        source_name="semantic_scholar",
                        raw_data=paper,
                        format="json",
                    ))

                offset += limit
                if data.get("next") is None:
                    break

        return raw_docs[:max_docs]

    def parse(self, raw: RawDocument) -> ParsedDocument:
        data = raw.raw_data

        doi = ""
        ext_ids = data.get("externalIds") or {}
        if isinstance(ext_ids, dict):
            doi = ext_ids.get("DOI", "") or ""

        tldr = ""
        tldr_obj = data.get("tldr")
        if isinstance(tldr_obj, dict):
            tldr = tldr_obj.get("text", "") or ""

        authors = [
            a["name"] for a in (data.get("authors") or [])
            if a.get("name")
        ]

        fields = data.get("fieldsOfStudy") or []

        return ParsedDocument(
            source_id=raw.source_id,
            source_name="semantic_scholar",
            title=data.get("title", "") or "",
            abstract=data.get("abstract", "") or "",
            tldr=tldr,
            authors=authors,
            doi=doi,
            publication_year=data.get("year", 0) or 0,
            fields_of_study=fields,
            url=(data.get("openAccessPdf") or {}).get("url", "") or "",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_sources.py::TestSemanticScholarConnector -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest/sources/semantic_scholar.py tests/test_ingest_sources.py
git commit -m "Add Semantic Scholar connector with TLDR extraction"
```

---

### Task 9: ERIC connector

**Files:**
- Create: `scripts/ingest/sources/eric.py`
- Modify: `tests/test_ingest_sources.py`

- [ ] **Step 1: Add failing tests for ERIC**

Append to `tests/test_ingest_sources.py`:

```python
from scripts.ingest.sources.eric import ERICConnector


class TestERICConnector:
    def test_implements_protocol(self):
        connector = ERICConnector(rps=5.0)
        assert isinstance(connector, SourceConnector)

    def test_parse_document(self):
        connector = ERICConnector(rps=5.0)
        from scripts.ingest.models import RawDocument
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_sources.py::TestERICConnector -v`
Expected: FAIL

- [ ] **Step 3: Write `scripts/ingest/sources/eric.py`**

```python
import aiohttp
from scripts.ingest.models import RawDocument, ParsedDocument
from scripts.ingest.rate_limiter import RateLimiter

ERIC_BASE = "https://api.ies.ed.gov/eric/"


class ERICConnector:
    source_name = "eric"

    def __init__(self, rps: float = 5.0):
        self._limiter = RateLimiter(rps)

    async def fetch(self, query: str, max_docs: int = 20000) -> list[RawDocument]:
        raw_docs = []
        async with aiohttp.ClientSession() as session:
            start = 0
            rows = 200
            while start < max_docs:
                await self._limiter.acquire()
                params = {
                    "search": query,
                    "rows": rows,
                    "start": start,
                    "format": "json",
                }
                resp = await session.get(ERIC_BASE, params=params)
                data = await resp.json()

                docs = data.get("response", {}).get("docs", [])
                if not docs:
                    break

                for doc in docs:
                    doc_id = doc.get("id", "")
                    if not doc_id:
                        continue
                    raw_docs.append(RawDocument(
                        source_id=doc_id,
                        source_name="eric",
                        raw_data=doc,
                        format="json",
                    ))

                start += rows

        return raw_docs[:max_docs]

    def parse(self, raw: RawDocument) -> ParsedDocument:
        data = raw.raw_data
        return ParsedDocument(
            source_id=raw.source_id,
            source_name="eric",
            title=data.get("title", "") or "",
            abstract=data.get("description", "") or "",
            subject_terms=data.get("subject", []) or [],
            peer_reviewed=bool(data.get("peerreviewed", False)),
            url=data.get("url", "") or "",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_sources.py::TestERICConnector -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest/sources/eric.py tests/test_ingest_sources.py
git commit -m "Add ERIC education research connector"
```

---

### Task 10: Government HTML crawler

**Files:**
- Create: `scripts/ingest/sources/government.py`
- Modify: `tests/test_ingest_sources.py`

- [ ] **Step 1: Add failing tests for Government connector**

Append to `tests/test_ingest_sources.py`:

```python
from scripts.ingest.sources.government import GovernmentConnector


class TestGovernmentConnector:
    def test_implements_protocol(self):
        connector = GovernmentConnector(rps=2.0)
        assert isinstance(connector, SourceConnector)

    def test_parse_html_page(self):
        connector = GovernmentConnector(rps=2.0)
        from scripts.ingest.models import RawDocument
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
        from scripts.ingest.models import RawDocument
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_sources.py::TestGovernmentConnector -v`
Expected: FAIL

- [ ] **Step 3: Write `scripts/ingest/sources/government.py`**

```python
import re
from urllib.parse import urlparse
import aiohttp
from bs4 import BeautifulSoup
from scripts.ingest.models import RawDocument, ParsedDocument
from scripts.ingest.rate_limiter import RateLimiter

# Scoped crawl targets
CRAWL_ROOTS = [
    "https://www.cdc.gov/adhd/",
    "https://www.nimh.nih.gov/health/topics/attention-deficit-hyperactivity-disorder-adhd",
]


class GovernmentConnector:
    source_name = "government"

    def __init__(self, rps: float = 2.0):
        self._limiter = RateLimiter(rps)

    async def fetch(self, query: str = "", max_docs: int = 1000) -> list[RawDocument]:
        raw_docs = []
        visited: set[str] = set()

        async with aiohttp.ClientSession() as session:
            for root_url in CRAWL_ROOTS:
                await self._crawl(session, root_url, raw_docs, visited, max_docs, depth=0, max_depth=3)
                if len(raw_docs) >= max_docs:
                    break

        return raw_docs[:max_docs]

    async def _crawl(
        self,
        session: aiohttp.ClientSession,
        url: str,
        raw_docs: list[RawDocument],
        visited: set[str],
        max_docs: int,
        depth: int,
        max_depth: int,
    ) -> None:
        if url in visited or len(raw_docs) >= max_docs or depth > max_depth:
            return
        visited.add(url)

        try:
            await self._limiter.acquire()
            resp = await session.get(url, timeout=aiohttp.ClientTimeout(total=30))
            if resp.status != 200:
                return
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                return
            html = await resp.text()
        except Exception:
            return

        # Create slug from URL path
        parsed_url = urlparse(url)
        slug = re.sub(r"[^\w]", "-", parsed_url.path.strip("/"))
        slug = re.sub(r"-+", "-", slug).strip("-")
        if not slug:
            slug = parsed_url.netloc.replace(".", "-")

        raw_docs.append(RawDocument(
            source_id=slug,
            source_name="government",
            raw_data={"url": url, "html": html},
            format="html",
        ))

        # Find links within the same domain scope
        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(url).netloc
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("/"):
                href = f"{parsed_url.scheme}://{base_domain}{href}"
            link_domain = urlparse(href).netloc
            if link_domain == base_domain and href not in visited:
                await self._crawl(session, href, raw_docs, visited, max_docs, depth + 1, max_depth)

    def parse(self, raw: RawDocument) -> ParsedDocument:
        html = raw.raw_data.get("html", "")
        url = raw.raw_data.get("url", "")
        soup = BeautifulSoup(html, "html.parser")

        # Remove nav, header, footer, script, style
        for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()

        # Title
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)

        # Lists
        list_items = []
        for li in soup.find_all("li"):
            text = li.get_text(strip=True)
            if text and len(text) > 10:
                list_items.append(text)
        has_lists = len(list_items) > 0

        # Body text
        body_text = soup.get_text(separator="\n", strip=True)

        return ParsedDocument(
            source_id=raw.source_id,
            source_name="government",
            title=title,
            abstract="",
            html_content=body_text,
            has_lists=has_lists,
            list_items=list_items,
            url=url,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_sources.py::TestGovernmentConnector -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest/sources/government.py tests/test_ingest_sources.py
git commit -m "Add government HTML crawler for CDC/NIH/NIMH"
```

---

### Task 11: CLI entry point and end-to-end pipeline

**Files:**
- Create: `scripts/ingest/main.py`
- Test: `tests/test_ingest_integration.py`

- [ ] **Step 1: Write `scripts/ingest/main.py`**

```python
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from scripts.ingest.config import IngestConfig
from scripts.ingest.gemini_client import GeminiClient
from scripts.ingest.models import RawDocument
from scripts.ingest.output.cache import RawCache
from scripts.ingest.output.writer import BatchWriter
from scripts.ingest.pipeline.deduplicator import Deduplicator
from scripts.ingest.pipeline.filter import RelevanceFilter
from scripts.ingest.pipeline.transformer import SchemaTransformer
from scripts.ingest.sources.pubmed import PubMedConnector
from scripts.ingest.sources.openalex import OpenAlexConnector
from scripts.ingest.sources.semantic_scholar import SemanticScholarConnector
from scripts.ingest.sources.eric import ERICConnector
from scripts.ingest.sources.government import GovernmentConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Source processing order (priority for dedup)
SOURCE_ORDER = ["pubmed", "semantic_scholar", "openalex", "eric", "government"]


def build_connectors(config: IngestConfig) -> dict:
    return {
        "pubmed": PubMedConnector(api_key=config.ncbi_api_key, rps=config.pubmed_rps),
        "openalex": OpenAlexConnector(email=config.openalex_email, rps=config.openalex_rps),
        "semantic_scholar": SemanticScholarConnector(api_key=config.s2_api_key, rps=config.s2_rps),
        "eric": ERICConnector(rps=config.eric_rps),
        "government": GovernmentConnector(rps=config.gov_rps),
    }


async def ingest_source(
    source_name: str,
    connector,
    queries: list[str],
    max_docs: int,
    cache: RawCache,
    transformer: SchemaTransformer,
    relevance_filter: RelevanceFilter | None,
    writer: BatchWriter,
    resume: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Fetch, parse, transform, filter, and write docs for one source."""
    logger.info(f"[{source_name}] Starting ingestion...")

    # Fetch
    all_raw: list[RawDocument] = []
    for query in queries:
        logger.info(f"[{source_name}] Fetching query: {query}")
        raw_docs = await connector.fetch(query, max_docs=max_docs)
        for raw in raw_docs:
            if resume and cache.has(source_name, raw.source_id):
                continue
            cache.save(source_name, raw.source_id, raw.raw_data)
            all_raw.append(raw)
        logger.info(f"[{source_name}] Fetched {len(raw_docs)} docs for '{query}'")

    if resume:
        # Load all cached docs
        cached_ids = cache.list_ids(source_name)
        for cid in cached_ids:
            data = cache.load(source_name, cid)
            if data:
                all_raw.append(RawDocument(
                    source_id=cid,
                    source_name=source_name,
                    raw_data=data,
                    format="json",
                ))

    logger.info(f"[{source_name}] Total raw documents: {len(all_raw)}")

    # Parse + Transform
    transformed = []
    for raw in all_raw:
        try:
            parsed = connector.parse(raw)
            doc = transformer.transform(parsed)
            transformed.append(doc)
        except Exception as e:
            logger.warning(f"[{source_name}] Failed to parse {raw.source_id}: {e}")

    logger.info(f"[{source_name}] Transformed: {len(transformed)} docs")

    # Filter
    if relevance_filter:
        transformed = await relevance_filter.filter(transformed)
        logger.info(f"[{source_name}] After filtering: {len(transformed)} docs")

    if dry_run:
        logger.info(f"[{source_name}] DRY RUN - would write {len(transformed)} docs")
        return transformed

    # Write
    files = writer.write(source_name, transformed)
    logger.info(f"[{source_name}] Wrote {len(files)} batch files")

    return transformed


async def run(args: argparse.Namespace) -> None:
    config = IngestConfig()
    connectors = build_connectors(config)
    cache = RawCache(config.raw_cache_dir)
    transformer = SchemaTransformer()
    writer = BatchWriter(config.output_dir, config.batch_size)
    deduplicator = Deduplicator(config.dedup_title_threshold)

    relevance_filter = None
    if config.gemini_api_key:
        client = GeminiClient(api_key=config.gemini_api_key)
        relevance_filter = RelevanceFilter(
            client=client,
            threshold=args.threshold or config.relevance_threshold,
            batch_size=config.filter_batch_size,
        )

    queries = [args.query] if args.query else config.default_queries
    sources = SOURCE_ORDER if args.source == "all" else [args.source]

    all_docs: list[dict] = []
    for source_name in sources:
        if source_name not in connectors:
            logger.error(f"Unknown source: {source_name}")
            continue

        if args.filter_only:
            # Re-filter cached docs
            cached_ids = cache.list_ids(source_name)
            connector = connectors[source_name]
            raw_docs = []
            for cid in cached_ids:
                data = cache.load(source_name, cid)
                if data:
                    raw = RawDocument(source_id=cid, source_name=source_name, raw_data=data, format="json")
                    try:
                        parsed = connector.parse(raw)
                        doc = transformer.transform(parsed)
                        raw_docs.append(doc)
                    except Exception:
                        pass
            if relevance_filter:
                raw_docs = await relevance_filter.filter(raw_docs)
            if not args.dry_run:
                writer.write(source_name, raw_docs)
            all_docs.extend(raw_docs)
            logger.info(f"[{source_name}] Re-filtered: {len(raw_docs)} docs")
        else:
            docs = await ingest_source(
                source_name=source_name,
                connector=connectors[source_name],
                queries=queries,
                max_docs=args.max_docs,
                cache=cache,
                transformer=transformer,
                relevance_filter=relevance_filter,
                writer=writer if not args.source == "all" else BatchWriter(config.output_dir, config.batch_size),
                resume=args.resume,
                dry_run=args.dry_run,
            )
            all_docs.extend(docs)

    # Cross-source dedup (only when ingesting all sources)
    if args.source == "all" and not args.dry_run:
        deduped = deduplicator.deduplicate(all_docs)
        logger.info(f"After dedup: {len(deduped)} docs (removed {len(all_docs) - len(deduped)} duplicates)")
        # Re-write with deduped docs, grouped by source prefix
        source_groups: dict[str, list[dict]] = {}
        for doc in deduped:
            prefix = doc["id"].split("_")[0]
            source_map = {"pmc": "pubmed", "oalex": "openalex", "s2": "semantic_scholar",
                          "eric": "eric", "gov": "government"}
            src = source_map.get(prefix, prefix)
            source_groups.setdefault(src, []).append(doc)
        for src, docs in source_groups.items():
            writer.write(src, docs)

    logger.info(f"Ingestion complete. Total documents: {len(all_docs)}")


def main():
    parser = argparse.ArgumentParser(description="Ingest ADHD knowledge documents")
    parser.add_argument("--source", required=True,
                        choices=["pubmed", "openalex", "semantic_scholar", "eric", "government", "all"],
                        help="Source to ingest from")
    parser.add_argument("--query", type=str, default=None,
                        help="Search query (overrides defaults)")
    parser.add_argument("--max-docs", type=int, default=10000,
                        help="Maximum documents to fetch per query")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and filter but don't write output")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted ingestion")
    parser.add_argument("--filter-only", action="store_true",
                        help="Re-filter cached docs without re-fetching")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Override relevance threshold")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write integration test with mocked APIs**

```python
# tests/test_ingest_integration.py
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
```

- [ ] **Step 3: Run integration tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_integration.py -v`
Expected: All PASSED

- [ ] **Step 4: Commit**

```bash
git add scripts/ingest/main.py tests/test_ingest_integration.py
git commit -m "Add CLI entry point and end-to-end integration tests"
```

---

### Task 12: App-side prerequisite -- update glob pattern

**Files:**
- Modify: `app/rag/knowledge_store.py:77`

- [ ] **Step 1: Update glob pattern**

Change line 77 in `app/rag/knowledge_store.py` from:
```python
for json_file in sorted(self.knowledge_dir.glob("*.json")):
```
to:
```python
for json_file in sorted(self.knowledge_dir.glob("**/*.json")):
```

- [ ] **Step 2: Run existing RAG tests to verify no regression**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py -v`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add app/rag/knowledge_store.py
git commit -m "Update knowledge store to load JSON from subdirectories"
```

---

### Task 13: Update .env.example and gitignore

**Files:**
- Modify: `.env.example`, `.gitignore`

- [ ] **Step 1: Add ingest-specific env vars to `.env.example`**

Append to `.env.example`:
```
# Ingestion pipeline (scripts/ingest)
INGEST_NCBI_API_KEY=           # Free from https://www.ncbi.nlm.nih.gov/account/settings/
INGEST_S2_API_KEY=             # Free from https://www.semanticscholar.org/product/api
INGEST_OPENALEX_EMAIL=         # Your email for polite pool
INGEST_GEMINI_API_KEY=         # For relevance scoring (can reuse GEMINI_API_KEY)
```

- [ ] **Step 2: Add cache dir to `.gitignore`**

Append to `.gitignore`:
```
scripts/ingest/.cache/
```

- [ ] **Step 3: Commit**

```bash
git add .env.example .gitignore
git commit -m "Add ingestion pipeline env vars and cache gitignore"
```

---

### Task 14: Run all tests

- [ ] **Step 1: Run the full test suite**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_ingest_transformer.py tests/test_ingest_filter.py tests/test_ingest_dedup.py tests/test_ingest_sources.py tests/test_ingest_writer.py tests/test_ingest_integration.py -v`
Expected: All PASSED

- [ ] **Step 2: Run existing app tests to check no regression**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py tests/test_prompts.py -v`
Expected: All PASSED

- [ ] **Step 3: Commit any fixes if needed**
