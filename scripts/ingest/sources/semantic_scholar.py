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
