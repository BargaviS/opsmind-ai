from app.rag.vector_store import VectorStore
from app.core.logger import get_logger

logger = get_logger("opsmind.retriever")


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        logger.info(f"Retrieving top {top_k} chunks for query: '{query}'")
        results = self.vector_store.search(query, top_k=top_k)
        logger.info(f"Retrieved {len(results)} chunks")
        return results

    def build_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return "No relevant documents found."
        parts = []
        for i, item in enumerate(chunks, 1):
            source = item.get("filename", item.get("document_id", "unknown"))
            parts.append(f"[Source {i}: {source}]\n{item['chunk']}")
        return "\n\n---\n\n".join(parts)
