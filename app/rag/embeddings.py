"""Local embeddings (no API cost). The model loads once, on first use."""
from functools import lru_cache

# bge models retrieve better when the QUERY (not the documents) has this prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: keeps tests fast

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, normalize_embeddings=True, batch_size=32)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._load().encode(BGE_QUERY_PREFIX + text, normalize_embeddings=True)
        return vector.tolist()


@lru_cache
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)
