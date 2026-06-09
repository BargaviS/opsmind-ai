from pathlib import Path
from app.core.logger import get_logger

logger = get_logger("opsmind.extractor")


class TextExtractor:
    """
    Extracts raw text from uploaded files.
    Supports: .txt, .pdf
    """

    SUPPORTED_TYPES = {"txt", "pdf"}

    def extract(self, file_path: str, file_type: str) -> str:
        file_type = file_type.lower().strip(".")

        if file_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported file type: '{file_type}'. "
                f"Supported: {self.SUPPORTED_TYPES}"
            )

        logger.info(f"Extracting text from {file_path} (type={file_type})")

        if file_type == "pdf":
            return self._extract_pdf(file_path)
        elif file_type == "txt":
            return self._extract_txt(file_path)

    def _extract_pdf(self, file_path: str) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")

        reader = PdfReader(file_path)
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(text.strip())
            else:
                logger.warning(f"Page {i+1} returned no text (possibly scanned image)")

        full_text = "\n\n".join(pages)
        logger.info(f"Extracted {len(full_text)} characters from PDF ({len(reader.pages)} pages)")
        return full_text

    def _extract_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        logger.info(f"Extracted {len(text)} characters from TXT")
        return text
