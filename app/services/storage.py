"""Keep the original uploaded file on disk, named by content hash."""
from pathlib import Path


def save_upload(upload_dir: str, document_id: str, filename: str, data: bytes) -> Path:
    folder = Path(upload_dir)
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    path = folder / f"{document_id}{suffix}"
    path.write_bytes(data)
    return path
