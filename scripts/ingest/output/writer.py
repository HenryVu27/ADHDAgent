import json
import shutil
from pathlib import Path


class BatchWriter:
    """Writes documents as batched JSON files to app/knowledge/<source>/."""

    def __init__(self, output_dir: str, batch_size: int = 500):
        self._output_dir = Path(output_dir)
        self._batch_size = batch_size

    def write(self, source_name: str, docs: list[dict]) -> list[Path]:
        """Write docs to source subdirectory. Overwrites existing files for this source."""
        source_dir = self._output_dir / source_name

        # Overwrite: remove existing source directory
        if source_dir.exists():
            shutil.rmtree(source_dir)
        source_dir.mkdir(parents=True, exist_ok=True)

        if not docs:
            return []

        written_files = []
        for i in range(0, len(docs), self._batch_size):
            batch = docs[i:i + self._batch_size]
            batch_num = (i // self._batch_size) + 1
            file_path = source_dir / f"{source_name}_batch_{batch_num:03d}.json"
            with open(file_path, "w") as f:
                json.dump(batch, f, indent=2)
            written_files.append(file_path)

        return written_files
