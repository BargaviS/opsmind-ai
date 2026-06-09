from functools import lru_cache
from typing import Optional
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("opsmind.vectorstore")


class VectorStore:
    """
    ChromaDB wrapper.

    Key design decisions:
    - Single PersistentClient shared across all requests (via get_vector_store singleton)
    - Embedding function registered at collection level — ChromaDB handles encoding
    - Metadata stored per chunk for source attribution in search results
    """

    def __init__(self):
        settings = get_settings()

        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)

        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"VectorStore ready — collection='{settings.CHROMA_COLLECTION}' "
            f"docs={self.collection.count()}"
        )

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[str],
    ) -> None:
        """
        Store chunks with metadata for source attribution.
        """
        if not chunks:
            raise ValueError("Cannot store empty chunk list")

        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"document_id": document_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )

        logger.info(f"Stored {len(chunks)} chunks for document '{filename}' (id={document_id})")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Semantic search. Returns list of dicts with chunk text, metadata, distance.
        """
        settings = get_settings()
        n = top_k or settings.SEARCH_TOP_K

        total = self.collection.count()
        if total == 0:
            logger.warning("Vector store is empty — no documents indexed yet")
            return []

        n = min(n, total)  # ChromaDB errors if n_results > total docs

        results = self.collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            output.append({
                "chunk": doc,
                "document_id": meta.get("document_id", ""),
                "filename": meta.get("filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(1 - dist, 4),  # cosine similarity from distance
            })

        return output


@lru_cache()
def get_vector_store() -> VectorStore:
    """Singleton — ChromaDB client created once per process."""
    return VectorStore()
