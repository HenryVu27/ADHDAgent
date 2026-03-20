import json
import pytest
from pathlib import Path
from scripts.ingest.output.writer import BatchWriter
from scripts.ingest.output.cache import RawCache


class TestBatchWriter:
    def test_writes_batch_files(self, tmp_path):
        writer = BatchWriter(output_dir=str(tmp_path), batch_size=2)
        docs = [
            {"id": "doc1", "name": "Doc 1"},
            {"id": "doc2", "name": "Doc 2"},
            {"id": "doc3", "name": "Doc 3"},
        ]
        writer.write("pubmed", docs)

        pubmed_dir = tmp_path / "pubmed"
        assert pubmed_dir.exists()
        files = sorted(pubmed_dir.glob("*.json"))
        assert len(files) == 2  # 2 docs + 1 doc = 2 files

        with open(files[0]) as f:
            batch1 = json.load(f)
        assert len(batch1) == 2

        with open(files[1]) as f:
            batch2 = json.load(f)
        assert len(batch2) == 1

    def test_overwrites_existing_directory(self, tmp_path):
        writer = BatchWriter(output_dir=str(tmp_path), batch_size=500)
        writer.write("pubmed", [{"id": "old"}])
        writer.write("pubmed", [{"id": "new"}])

        files = list((tmp_path / "pubmed").glob("*.json"))
        assert len(files) == 1
        with open(files[0]) as f:
            data = json.load(f)
        assert data[0]["id"] == "new"

    def test_empty_docs_creates_empty_dir(self, tmp_path):
        writer = BatchWriter(output_dir=str(tmp_path), batch_size=500)
        writer.write("pubmed", [])
        assert (tmp_path / "pubmed").exists()
        assert list((tmp_path / "pubmed").glob("*.json")) == []


class TestRawCache:
    def test_save_and_load(self, tmp_path):
        cache = RawCache(cache_dir=str(tmp_path))
        cache.save("pubmed", "doc1", {"title": "Test"})
        result = cache.load("pubmed", "doc1")
        assert result == {"title": "Test"}

    def test_load_missing_returns_none(self, tmp_path):
        cache = RawCache(cache_dir=str(tmp_path))
        assert cache.load("pubmed", "missing") is None

    def test_has(self, tmp_path):
        cache = RawCache(cache_dir=str(tmp_path))
        cache.save("pubmed", "doc1", {"title": "Test"})
        assert cache.has("pubmed", "doc1")
        assert not cache.has("pubmed", "doc2")

    def test_list_cached_ids(self, tmp_path):
        cache = RawCache(cache_dir=str(tmp_path))
        cache.save("pubmed", "doc1", {})
        cache.save("pubmed", "doc2", {})
        ids = cache.list_ids("pubmed")
        assert sorted(ids) == ["doc1", "doc2"]
