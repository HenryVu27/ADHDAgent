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

    def _extract_year(self, doc: dict) -> int:
        source = doc.get("source", "")
        import re
        match = re.search(r"\((\d{4})\)", source)
        return int(match.group(1)) if match else 0

    def deduplicate(self, docs: list[dict]) -> list[dict]:
        """Remove duplicates. Input should be ordered by source priority
        (PubMed first, then Semantic Scholar, OpenAlex, ERIC, CDC/NIH)
        so first-seen wins."""
        if not docs:
            return []

        seen_dois: set[str] = set()
        seen_titles: list[tuple[str, int, str]] = []  # (title, year, id)
        kept: list[dict] = []

        for doc in docs:
            doi = self._extract_doi(doc)
            if doi:
                if doi in seen_dois:
                    continue
                seen_dois.add(doi)

            # Title similarity check with year as secondary filter
            title = doc.get("name", "")
            year = self._extract_year(doc)
            is_dup = False
            for seen_title, seen_year, _ in seen_titles:
                if self._jaccard_similarity(title, seen_title) >= self._title_threshold:
                    if seen_year == 0 or year == 0 or seen_year == year:
                        is_dup = True
                        break

            if not is_dup:
                seen_titles.append((title, year, doc["id"]))
                kept.append(doc)

        return kept
