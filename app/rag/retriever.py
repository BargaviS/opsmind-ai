from app.rag.vector_store import VectorStore
from app.rag.reranker import get_reranker
from app.core.logger import get_logger

logger = get_logger("opsmind.retriever")


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.reranker = get_reranker()

    def retrieve(self, query: str, top_k: int = 5) -> list:
        logger.info(f"Stage 1 — semantic search: top {top_k} candidates")
        candidates = self.vector_store.search(query, top_k=top_k)
        if not candidates:
            return []
        logger.info(f"Stage 2 — reranking {len(candidates)} candidates")
        return self.reranker.rerank(query, candidates, top_k=3)

    def build_context(self, chunks: list) -> str:
        if not chunks:
            return "No relevant documents found."
        parts = []
        for i, item in enumerate(chunks, 1):
            source = item.get("filename", item.get("document_id", "unknown"))
            score = item.get("rerank_score", item.get("score", 0))
            parts.append(f"[Source {i}: {source} | relevance: {score:.2f}]\n{item['chunk']}")
        return "\n\n---\n\n".join(parts)
