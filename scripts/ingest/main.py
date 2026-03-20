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
from scripts.ingest.sources.pubmed_abstract import PubMedAbstractConnector
from scripts.ingest.sources.openalex import OpenAlexConnector
from scripts.ingest.sources.semantic_scholar import SemanticScholarConnector
from scripts.ingest.sources.eric import ERICConnector
from scripts.ingest.sources.government import GovernmentConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Source processing order (priority for dedup)
SOURCE_ORDER = ["pubmed", "pubmed_abstract", "semantic_scholar", "openalex", "eric", "government"]


def build_connectors(config: IngestConfig) -> dict:
    return {
        "pubmed": PubMedConnector(api_key=config.ncbi_api_key, rps=config.pubmed_rps),
        "pubmed_abstract": PubMedAbstractConnector(api_key=config.ncbi_api_key, rps=config.pubmed_rps),
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

    # Fetch (skip already-cached IDs when resuming)
    seen_ids: set[str] = set()
    all_raw: list[RawDocument] = []

    if resume:
        # Load previously cached docs first
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
                seen_ids.add(cid)
        logger.info(f"[{source_name}] Loaded {len(all_raw)} cached docs")

    for query in queries:
        logger.info(f"[{source_name}] Fetching query: {query}")
        raw_docs = await connector.fetch(query, max_docs=max_docs)
        new_count = 0
        for raw in raw_docs:
            if raw.source_id in seen_ids:
                continue
            seen_ids.add(raw.source_id)
            cache.save(source_name, raw.source_id, raw.raw_data)
            all_raw.append(raw)
            new_count += 1
        logger.info(f"[{source_name}] Fetched {len(raw_docs)} docs for '{query}' ({new_count} new)")

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
                writer=writer,
                resume=args.resume,
                dry_run=args.dry_run or args.source == "all",  # Skip per-source writes when doing all (dedup writes later)
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
            source_map = {"pmc": "pubmed", "pm": "pubmed_abstract", "oalex": "openalex",
                          "s2": "semantic_scholar", "eric": "eric", "gov": "government"}
            src = source_map.get(prefix, prefix)
            source_groups.setdefault(src, []).append(doc)
        for src, docs in source_groups.items():
            writer.write(src, docs)

    logger.info(f"Ingestion complete. Total documents: {len(all_docs)}")


def main():
    parser = argparse.ArgumentParser(description="Ingest ADHD knowledge documents")
    parser.add_argument("--source", required=True,
                        choices=["pubmed", "pubmed_abstract", "openalex", "semantic_scholar", "eric", "government", "all"],
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
