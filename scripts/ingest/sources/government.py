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

        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(url).netloc
        root_path = ""
        for root in CRAWL_ROOTS:
            if url.startswith(root):
                root_path = urlparse(root).path
                break

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("/"):
                href = f"{parsed_url.scheme}://{base_domain}{href}"
            link_parsed = urlparse(href)
            if (link_parsed.netloc == base_domain
                    and link_parsed.path.startswith(root_path)
                    and href not in visited):
                await self._crawl(session, href, raw_docs, visited, max_docs, depth + 1, max_depth)

    def parse(self, raw: RawDocument) -> ParsedDocument:
        html = raw.raw_data.get("html", "")
        url = raw.raw_data.get("url", "")
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()

        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)

        list_items = []
        for li in soup.find_all("li"):
            text = li.get_text(strip=True)
            if text and len(text) > 10:
                list_items.append(text)
        has_lists = len(list_items) > 0

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
