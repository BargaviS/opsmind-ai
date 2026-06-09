from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("opsmind.chunker")


class DocumentChunker:
    """
    Splits raw text into overlapping chunks for embedding.
    Uses RecursiveCharacterTextSplitter for semantically aware splits.
    """

    def __init__(self):
        settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            raise ValueError("Cannot chunk empty text")

        chunks = self.splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
