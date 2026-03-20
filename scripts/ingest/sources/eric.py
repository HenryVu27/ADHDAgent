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
                data = await resp.json(content_type=None)

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
