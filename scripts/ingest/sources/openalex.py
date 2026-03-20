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
