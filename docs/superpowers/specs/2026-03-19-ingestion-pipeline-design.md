# Data Ingestion Pipeline Design

## Purpose

A standalone CLI tool that downloads documents from authoritative sources (PubMed, OpenAlex, Semantic Scholar, ERIC, CDC/NIH/NIMH), filters them for relevance to ADHD parenting coaching, maps them to the existing knowledge base JSON schema, and writes them to `app/knowledge/<source>/`. The existing app pipeline handles chunking, embedding, and indexing.

## Goals

- Scale the knowledge base from ~100 curated documents to 100k+ documents
- Only ingest peer-reviewed research and official government sources
- Output documents in the exact same JSON schema used by existing curated content
- Run on-demand via CLI, with resumability for large ingestions
- Keep dependencies isolated from the main app

## Non-Goals

- Chunking, embedding, or Qdrant indexing (handled by existing app pipeline)
- Real-time or scheduled ingestion (run manually)
- Community content (Reddit, StackExchange)

---

## Architecture

```
scripts/ingest/
  requirements.txt          # LlamaIndex + source-specific deps (isolated from app)
  config.py                 # API keys, output paths, filter thresholds
  main.py                   # CLI entry point

  sources/                  # One module per data source
    base.py                 # SourceConnector protocol
    pubmed.py               # PMC E-Utilities fetcher + JATS XML parser
    openalex.py             # OpenAlex REST API with cursor pagination
    semantic_scholar.py     # S2 Graph API with TLDR extraction
    eric.py                 # ERIC REST API fetcher
    government.py           # CDC/NIH/NIMH HTML crawler

  pipeline/
    parser.py               # LlamaIndex-based parsing (XML, HTML)
    transformer.py          # Maps parsed documents to knowledge base JSON schema
    filter.py               # Gemini Flash relevance scoring (0-10)
    deduplicator.py         # Cross-source dedup by DOI and title similarity

  output/
    writer.py               # Writes JSON files to app/knowledge/<source>/
```

### Pipeline Flow

```
Source API -> Raw documents (XML/JSON/HTML)
  -> LlamaIndex Parser (extract clean text + metadata)
  -> Transformer (map to JSON schema)
  -> Gemini Flash Filter (score relevance 0-10, discard below threshold)
  -> Deduplicator (DOI exact match + title similarity)
  -> JSON Writer (app/knowledge/<source>/<batch>.json)
```

---

## Source Connectors

Each connector implements a common protocol:

```python
class SourceConnector(Protocol):
    source_name: str

    async def fetch(self, query: str, max_docs: int) -> list[RawDocument]
    def parse(self, raw: RawDocument) -> ParsedDocument
```

### PubMed/PMC

- E-Utilities API: `esearch` for PMC IDs, `efetch` for JATS XML
- Free API key from NCBI (10 req/sec)
- Extracts: title, abstract, MeSH terms, authors, DOI, publication date, publication type
- For full-text papers, extracts the abstract only (the app pipeline handles any further splitting)
- MeSH terms map to `tags`

### OpenAlex

- REST API with cursor pagination, `mailto` param for polite pool (faster rate)
- Reconstructs abstracts from `abstract_inverted_index` field
- Extracts: title, abstract, concepts (top 5 -> tags), cited_by_count, DOI, publication year
- Highest volume source (~200k+ ADHD-related works)

### Semantic Scholar

- Graph API v1, free API key for 10 req/sec
- Key advantage: `tldr` field provides pre-summarized content, maps well to `key_points`
- Extracts: title, abstract, tldr, fields_of_study, citation_count, DOI, open_access_pdf link

### ERIC

- Free REST API, no key needed
- Returns: title, description (abstract), subject terms (-> tags), peer_reviewed flag, PDF links
- Education-focused content that other sources miss (classroom strategies, IEP research)

### CDC/NIH/NIMH

- Scoped HTML crawl of ADHD sections using LlamaIndex web readers
- CDC: `cdc.gov/adhd/` tree
- NIH/NIMH: `nimh.nih.gov/health/topics/attention-deficit-hyperactivity-disorder-adhd`
- Smallest volume (~500-1k pages) but highest parent-readability
- All public domain (US Government works)

---

## Schema Mapping

All source documents map to the existing knowledge base schema:

| Schema Field | PubMed | OpenAlex | Semantic Scholar | ERIC | CDC/NIH |
|---|---|---|---|---|---|
| `id` | `pmc_{pmcid}` | `oalex_{id}` | `s2_{corpusid}` | `eric_{id}` | `gov_{url_slug}` |
| `name` | Paper title | Paper title | Paper title | Document title | Page title |
| `description` | Abstract (first 200 chars) | Abstract (first 200 chars) | TLDR (preferred) or abstract snippet | Description (first 200 chars) | First paragraph |
| `document_type` | `"fact"` or `"guidance"` based on publication type | `"fact"` | `"fact"` | `"guidance"` or `"strategy"` based on subject terms | `"guidance"` |
| `tags` | MeSH terms (normalized) | Concepts (top 5) | Fields of study | Subject terms | Derived from URL path |
| `age_range` | Inferred from MeSH age groups + keyword scan | Keyword scan on title/abstract | Keyword scan | Keyword scan | Keyword scan |
| `evidence_level` | By publication type: RCT/review = `"strong"`, observational = `"moderate"`, case study = `"emerging"` | Citation count heuristic: >50 = `"strong"`, >10 = `"moderate"`, else `"emerging"` | Same as OpenAlex | `peer_reviewed` = `"moderate"`, else `"emerging"` | `"expert_consensus"` |
| `source` | `"{first_author} et al. ({year})"` | Same | Same | Same | `"CDC"` / `"NIH"` / `"NIMH"` |
| `citations` | `[{"source_file": "", "source_name": "{source}", "detail": "{DOI}"}]` | Same pattern | Same pattern | Same with ERIC ID | Same with URL |
| `steps` | `[]` | `[]` | `[]` | `[]` | Extracted if page has ordered/unordered lists |
| `key_points` | Abstract split into sentences | Abstract split into sentences | TLDR as single point + abstract sentences | Description split into sentences | Page content split into bullets by paragraph |
| `contraindications` | `[]` | `[]` | `[]` | `[]` | `[]` |
| `related_ids` | `[]` | `[]` | `[]` | `[]` | `[]` |

### Age Range Inference

Keyword scan on title + abstract:
- `"preschool"`, `"early childhood"`, `"ages 3-5"`, `"pre-k"`, `"toddler"` -> `["preschool"]`
- `"school-age"`, `"elementary"`, `"ages 6-12"`, `"child"`, `"children"` -> `["school_age"]`
- `"adolescent"`, `"teen"`, `"ages 13-17"`, `"high school"` -> `["adolescent"]`
- Multiple matches -> include all matched ranges
- No match -> `["all"]`

### Document Type Inference

- PubMed: Randomized controlled trials, systematic reviews, meta-analyses -> `"fact"`. Clinical guidelines, practice recommendations -> `"guidance"`.
- ERIC: Documents with subject terms containing "strategies", "interventions", "programs" -> `"strategy"`. Others -> `"guidance"`.
- Government pages with actionable lists (steps, how-to) -> `"strategy"`. Informational pages -> `"guidance"`.

---

## Filtering

### Gemini Flash Relevance Scoring

Each parsed document is scored by Gemini Flash:

```
Rate this document's relevance to an ADHD parenting coaching chatbot (0-10):
- 9-10: Directly about ADHD parenting strategies, child behavior management, or family support
- 7-8: About ADHD in children/adolescents with practical implications for parents
- 5-6: About ADHD generally (neuroscience, adult ADHD, pharmacology) or general parenting
- 3-4: Tangentially related (general child psychology, education theory)
- 0-2: Not relevant

Title: {title}
Abstract: {abstract}
Tags: {tags}

Return JSON: {"score": int, "reason": str}
```

- Default threshold: discard score < 6
- Batch processing: 20 documents per Flash call to reduce API overhead
- Uses the app's existing `app/llm/client.py` Gemini wrapper
- Estimated cost: ~$1-2 per 100k abstracts at Flash pricing

### Cross-Source Deduplication

The same paper often appears in PubMed, OpenAlex, and Semantic Scholar:

1. **DOI exact match**: If two documents share a DOI, keep the richest version
2. **Title similarity fallback**: Normalized title comparison (lowercase, strip punctuation). Jaccard token similarity > 0.85 = duplicate.
3. **Source priority** (when merging): PubMed > Semantic Scholar > OpenAlex > ERIC > CDC/NIH. Higher-priority source's version is kept because it tends to have richer metadata.

---

## Output

### File Organization

```
app/knowledge/
  adhd_strategies.json          # Existing curated (never modified)
  adhd_facts.json               # Existing curated (never modified)
  executive_function.json       # Existing curated (never modified)
  parenting_guidance.json       # Existing curated (never modified)
  social_emotional.json         # Existing curated (never modified)
  pubmed/
    pubmed_batch_001.json       # 500 docs per file
    pubmed_batch_002.json
    ...
  openalex/
    openalex_batch_001.json
    ...
  semantic_scholar/
    s2_batch_001.json
    ...
  eric/
    eric_batch_001.json
    ...
  government/
    cdc_batch_001.json
    nih_batch_001.json
```

500 documents per JSON file. Existing curated files are never modified by the pipeline.

### Loader Update

The existing knowledge store loader (`app/rag/knowledge_store.py`) globs `*.json` in `app/knowledge/`. This needs a one-line update to glob `**/*.json` to pick up subdirectory files.

---

## CLI Interface

```bash
# Ingest from a specific source
python -m scripts.ingest --source pubmed --query "ADHD parenting" --max-docs 10000

# Ingest from all sources with default queries
python -m scripts.ingest --source all --max-docs 50000

# Dry run (fetch + parse + filter, print stats but don't write)
python -m scripts.ingest --source pubmed --query "ADHD" --dry-run

# Resume interrupted ingestion (skips already-fetched IDs)
python -m scripts.ingest --source pubmed --resume

# Re-filter existing cached docs with a different threshold
python -m scripts.ingest --filter-only --threshold 7
```

### Default Queries Per Source

Each source has a default set of ADHD-relevant queries:
- `"ADHD parenting strategies"`
- `"attention deficit hyperactivity disorder child behavior"`
- `"behavioral intervention pediatric ADHD"`
- `"executive function child intervention"`
- `"parent training ADHD"`
- `"ADHD school accommodations"`
- `"ADHD family support"`

---

## Configuration

```python
# scripts/ingest/config.py

# API keys (loaded from .env)
NCBI_API_KEY: str              # Free from https://www.ncbi.nlm.nih.gov/account/settings/
S2_API_KEY: str                # Free from https://www.semanticscholar.org/product/api
OPENALEX_EMAIL: str            # For polite pool (faster rate limits)

# Pipeline settings
RELEVANCE_THRESHOLD: int = 6   # Gemini Flash score 0-10, discard below
BATCH_SIZE: int = 500           # Documents per output JSON file
DEDUP_TITLE_THRESHOLD: float = 0.85  # Jaccard similarity for title dedup

# Rate limiting
PUBMED_RPS: float = 10.0
OPENALEX_RPS: float = 10.0
S2_RPS: float = 10.0
ERIC_RPS: float = 5.0
GOV_RPS: float = 2.0           # Polite crawl rate for government sites

# Paths
OUTPUT_DIR: str = "app/knowledge"
RAW_CACHE_DIR: str = "scripts/ingest/.cache"  # Gitignored, for resumability
```

### Raw Cache for Resumability

The pipeline caches raw API responses in `scripts/ingest/.cache/<source>/` so interrupted runs can resume without re-fetching. This directory is gitignored. Only the final filtered JSON files in `app/knowledge/` get committed.

---

## Dependencies

```
# scripts/ingest/requirements.txt (isolated from app)
llama-index-core>=0.12.0
llama-index-readers-web>=0.3.0
llama-index-readers-papers>=0.2.0
beautifulsoup4>=4.12.0
aiohttp>=3.9.0
tenacity>=9.0.0
```

No changes to the app's `requirements.txt`. The ingestion scripts use the app's `app/llm/client.py` for Gemini Flash calls (relevance scoring) but otherwise have no dependency on the app runtime.

---

## Estimated Volume

| Source | Raw Fetch | After Filtering (score >= 6) | After Dedup |
|---|---|---|---|
| PubMed/PMC | 50-80k | ~30-50k | ~30-50k |
| OpenAlex | 200k+ | ~60-80k | ~20-30k (heavy overlap with PubMed) |
| Semantic Scholar | 200k+ | ~60-80k | ~10-15k (after PubMed + OAlex dedup) |
| ERIC | 15-50k | ~8-20k | ~8-20k (low overlap) |
| CDC/NIH/NIMM | 500-1k | ~400-800 | ~400-800 |
| **Total** | | | **~70-120k unique documents** |
