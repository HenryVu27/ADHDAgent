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
            ids = await self._search(session, query, max_docs)
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
            title_el = meta.find(".//article-title")
            if title_el is not None:
                title = "".join(title_el.itertext()).strip()

            for aid in meta.findall("article-id"):
                if aid.get("pub-id-type") == "doi":
                    doi = aid.text or ""

            for contrib in meta.findall(".//contrib[@contrib-type='author']"):
                name_el = contrib.find("name")
                if name_el is not None:
                    surname = name_el.findtext("surname", "")
                    given = name_el.findtext("given-names", "")
                    authors.append(f"{surname} {given}".strip())

            for pub_date in meta.findall("pub-date"):
                year_el = pub_date.find("year")
                if year_el is not None and year_el.text:
                    year = int(year_el.text)
                    break

            abs_el = meta.find("abstract")
            if abs_el is not None:
                abstract = " ".join(abs_el.itertext()).strip()

            for kwd_group in meta.findall(".//kwd-group"):
                if "MeSH" in (kwd_group.get("kwd-group-type", "")):
                    for kwd in kwd_group.findall("kwd"):
                        if kwd.text:
                            mesh_terms.append(kwd.text.strip())

            for at in meta.findall(".//article-categories//subj-group/subject"):
                if at.text:
                    pub_type = at.text.strip()
                    break

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
