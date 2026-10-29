"""Two-stage retrieval + relevance guardrail.

Stage 1: vector search -> top_k candidates (fast, approximate)
Stage 2: CrossEncoder rerank -> top_n (slower, precise)
Guardrail: if even the best reranked chunk is below min_rerank_score,
we ABSTAIN instead of sending weak context to the LLM.
"""
from app.rag.embeddings import Embedder
from app.rag.reranker import Reranker
from app.rag.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        top_k: int,
        top_n: int,
        min_rerank_score: float,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self.top_n = top_n
        self.min_rerank_score = min_rerank_score

    def retrieve(self, query: str) -> tuple[list[dict], bool]:
        """Returns (chunks, abstained)."""
        candidates = self.store.search(self.embedder.embed_query(query), self.top_k)
        if not candidates:
            return [], True

        scores = self.reranker.score(query, [c["text"] for c in candidates])
        for cand, s in zip(candidates, scores):
            cand["rerank_score"] = round(s, 4)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[: self.top_n]

        # Keep only chunks that individually pass the threshold. Weak chunks are
        # not sent to the LLM and not shown as sources.
        relevant = [c for c in ranked if c["rerank_score"] >= self.min_rerank_score]
        if not relevant:
            return [], True
        return relevant, False

    @staticmethod
    def build_context(chunks: list[dict]) -> str:
        return "\n\n".join(
            f"[Source: {c['filename']}, chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
        )
