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
        is_practitioner: bool = False,
    ) -> str:
        if is_government or is_practitioner:
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
        if parsed.source_name in ("government", "practitioner"):
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
        if parsed.source_name == "practitioner":
            if "chadd" in parsed.url.lower():
                return "CHADD"
            if "additudemag" in parsed.url.lower():
                return "ADDitude"
            if "understood" in parsed.url.lower():
                return "Understood"
            return parsed.source_name
        if parsed.authors:
            first = parsed.authors[0]
            suffix = " et al." if len(parsed.authors) > 1 else ""
            year = f" ({parsed.publication_year})" if parsed.publication_year else ""
            return f"{first}{suffix}{year}"
        return parsed.source_name

    def _build_key_points(self, parsed: ParsedDocument) -> list[str]:
        if parsed.source_name in ("government", "practitioner") and not parsed.has_lists:
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
            "pubmed_abstract": "pm_",
            "openalex": "oalex_",
            "semantic_scholar": "s2_",
            "eric": "eric_",
            "government": "gov_",
            "practitioner": "pract_",
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
                is_practitioner=parsed.source_name == "practitioner",
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
