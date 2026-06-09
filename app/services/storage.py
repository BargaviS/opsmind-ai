import uuid
from pathlib import Path
from fastapi import UploadFile

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("opsmind.storage")


class FileStorageService:
    """
    Persists uploaded files to disk with a UUID-based filename.
    Returns metadata needed for downstream processing.
    """

    def __init__(self):
        settings = get_settings()
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, file: UploadFile) -> dict:
        """
        Save an uploaded file and return its metadata.
        """
        original_name = file.filename or "unknown"
        extension = Path(original_name).suffix.lstrip(".").lower()

        if not extension:
            raise ValueError(f"Cannot determine file type from: {original_name}")

        file_id = str(uuid.uuid4())
        stored_name = f"{file_id}.{extension}"
        dest_path = self.upload_dir / stored_name

        content = await file.read()

        with open(dest_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved file: {original_name} → {dest_path} ({len(content)} bytes)")

        return {
            "file_id": file_id,
            "file_path": str(dest_path),
            "filename": original_name,
            "file_type": extension,
            "size_bytes": len(content),
        }
