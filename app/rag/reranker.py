from sentence_transformers import CrossEncoder
from app.core.logger import get_logger
from functools import lru_cache

logger = get_logger("opsmind.reranker")


class Reranker:
    def __init__(self):
        logger.info("Loading CrossEncoder reranker model...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker ready")

    def rerank(self, query: str, chunks: list, top_k: int = 3) -> list:
        if not chunks:
            return []
        pairs = [[query, c["chunk"]] for c in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        logger.info(f"Reranked {len(chunks)} chunks, top {top_k} selected")
        return reranked[:top_k]


@lru_cache()
def get_reranker() -> Reranker:
    return Reranker()
