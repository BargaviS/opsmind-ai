from functools import lru_cache
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("opsmind.embeddings")


class EmbeddingService:
    """
    Generates dense vector embeddings using a local sentence-transformers model.
    Model is loaded once and reused — never reload on each request.
    """

    def __init__(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model ready")

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Singleton — model loads exactly once per process."""
    settings = get_settings()
    return EmbeddingService(model_name=settings.EMBEDDING_MODEL)
