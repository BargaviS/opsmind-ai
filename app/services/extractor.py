"""Turn an uploaded file into clean text. Raises ValueError with a safe message."""
import io
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(filename: str, data: bytes) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")

    if ext == ".txt":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
    else:
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except (PdfReadError, ValueError, KeyError, OSError):
            raise ValueError("This PDF is corrupt or unreadable.") from None

    text = clean_text(text)
    if not text:
        raise ValueError("No readable text found (scanned image PDFs are not supported).")
    return text
