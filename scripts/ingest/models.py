from dataclasses import dataclass, field


@dataclass
class RawDocument:
    """Raw document as fetched from a source API."""
    source_id: str
    source_name: str
    raw_data: dict
    format: str  # "xml", "json", "html"


@dataclass
class ParsedDocument:
    """Document after parsing, before schema transformation."""
    source_id: str
    source_name: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    publication_year: int = 0
    publication_type: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    subject_terms: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)
    tldr: str = ""
    work_type: str = ""  # OpenAlex/S2 type field (review, journal-article, etc.)
    peer_reviewed: bool = False
    url: str = ""
    html_content: str = ""  # For government pages
    has_lists: bool = False  # For government pages with step-like content
    list_items: list[str] = field(default_factory=list)
