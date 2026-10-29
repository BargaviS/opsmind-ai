"""ChromaDB wrapper. We compute embeddings ourselves and pass them in, so
Chroma never downloads its own default model.

Chunk IDs are deterministic (<document_id>-<index>), which makes re-ingestion
idempotent.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    def __init__(self, path: str, collection: str) -> None:
        self._client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def has_document(self, document_id: str) -> bool:
        found = self._col.get(where={"document_id": document_id}, limit=1)
        return len(found["ids"]) > 0

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        self._col.add(
            ids=[f"{document_id}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"document_id": document_id, "filename": filename, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        total = self._col.count()
        if total == 0:
            return []
        res = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append(
                {
                    "text": text,
                    "document_id": meta["document_id"],
                    "filename": meta["filename"],
                    "chunk_index": int(meta["chunk_index"]),
                    "vector_score": round(1.0 - float(dist), 4),  # cosine distance -> similarity
                }
            )
        return results

    def count_documents(self) -> int:
        metas = self._col.get(include=["metadatas"])["metadatas"] or []
        return len({m["document_id"] for m in metas})
