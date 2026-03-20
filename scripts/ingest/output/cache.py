import json
from pathlib import Path


class RawCache:
    """Caches raw API responses for resumability."""

    def __init__(self, cache_dir: str):
        self._cache_dir = Path(cache_dir)

    def _path(self, source: str, doc_id: str) -> Path:
        return self._cache_dir / source / f"{doc_id}.json"

    def save(self, source: str, doc_id: str, data: dict) -> None:
        path = self._path(source, doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, source: str, doc_id: str) -> dict | None:
        path = self._path(source, doc_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def has(self, source: str, doc_id: str) -> bool:
        return self._path(source, doc_id).exists()

    def list_ids(self, source: str) -> list[str]:
        source_dir = self._cache_dir / source
        if not source_dir.exists():
            return []
        return [p.stem for p in source_dir.glob("*.json")]
