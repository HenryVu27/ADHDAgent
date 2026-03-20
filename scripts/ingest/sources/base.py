from typing import Protocol, runtime_checkable
from scripts.ingest.models import RawDocument, ParsedDocument


@runtime_checkable
class SourceConnector(Protocol):
    source_name: str

    async def fetch(self, query: str, max_docs: int) -> list[RawDocument]:
        ...

    def parse(self, raw: RawDocument) -> ParsedDocument:
        ...
