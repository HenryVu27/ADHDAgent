import json
import xml.etree.ElementTree as ET
import aiohttp
from scripts.ingest.models import RawDocument, ParsedDocument
from scripts.ingest.rate_limiter import RateLimiter

E_UTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedAbstractConnector:
    """Searches the full PubMed/MEDLINE index (not just PMC full-text).
    Returns abstracts + metadata for a much larger pool of papers."""

    source_name = "pubmed_abstract"

    def __init__(self, api_key: str = "", rps: float = 3.0):
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
            "db": "pubmed",
            "term": query,
            "retmax": min(max_docs, 100000),
            "retmode": "json",
        }
        if self._api_key:
            params["api_key"] = self._api_key
        resp = await session.get(f"{E_UTILS_BASE}/esearch.fcgi", params=params)
        text = await resp.text()
        data = json.loads(text)
        return data.get("esearchresult", {}).get("idlist", [])

    async def _fetch_batch(self, session: aiohttp.ClientSession, ids: list[str]) -> str:
        await self._limiter.acquire()
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if self._api_key:
            params["api_key"] = self._api_key
        resp = await session.get(f"{E_UTILS_BASE}/efetch.fcgi", params=params)
        return await resp.text()

    def _split_articles(self, xml_text: str) -> list[RawDocument]:
        docs = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return docs

        for article in root.iter("PubmedArticle"):
            pmid = ""
            medline = article.find(".//MedlineCitation")
            if medline is not None:
                pmid_el = medline.find("PMID")
                if pmid_el is not None and pmid_el.text:
                    pmid = pmid_el.text

            if not pmid:
                continue

            docs.append(RawDocument(
                source_id=pmid,
                source_name="pubmed_abstract",
                raw_data={"xml": ET.tostring(article, encoding="unicode")},
                format="xml",
            ))
        return docs

    def parse(self, raw: RawDocument) -> ParsedDocument:
        xml_str = raw.raw_data.get("xml", "")
        root = ET.fromstring(xml_str)

        title = ""
        abstract = ""
        authors = []
        doi = ""
        year = 0
        mesh_terms = []
        pub_type = ""

        medline = root.find(".//MedlineCitation")
        if medline is not None:
            # Title
            title_el = medline.find(".//ArticleTitle")
            if title_el is not None:
                title = "".join(title_el.itertext()).strip()

            # Abstract
            abs_el = medline.find(".//Abstract")
            if abs_el is not None:
                parts = []
                for abs_text in abs_el.findall("AbstractText"):
                    label = abs_text.get("Label", "")
                    text = "".join(abs_text.itertext()).strip()
                    if label:
                        parts.append(f"{label}: {text}")
                    else:
                        parts.append(text)
                abstract = " ".join(parts)

            # Authors
            for author in medline.findall(".//Author"):
                last = author.findtext("LastName", "")
                fore = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {fore}".strip())

            # Year
            pub_date = medline.find(".//PubDate")
            if pub_date is not None:
                year_el = pub_date.find("Year")
                if year_el is not None and year_el.text:
                    year = int(year_el.text)

            # MeSH terms
            for mesh in medline.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text.strip())

            # Publication type
            for pt in medline.findall(".//PublicationType"):
                if pt.text:
                    pub_type = pt.text.strip()
                    break

        # DOI
        article_data = root.find(".//PubmedData")
        if article_data is not None:
            for aid in article_data.findall(".//ArticleId"):
                if aid.get("IdType") == "doi" and aid.text:
                    doi = aid.text

        return ParsedDocument(
            source_id=raw.source_id,
            source_name="pubmed_abstract",
            title=title,
            abstract=abstract,
            authors=authors,
            doi=doi,
            publication_year=year,
            publication_type=pub_type,
            mesh_terms=mesh_terms,
        )
