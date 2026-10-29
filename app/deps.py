"""Dependency providers. Each heavy object is created ONCE (lru_cache) and
reused. Tests replace them with fakes via app.dependency_overrides, which is
why no test needs a real model or a paid API.
"""
from functools import lru_cache

from fastapi import Depends

from app.core.config import get_settings
from app.rag.agent import Agent
from app.rag.chunker import DocumentChunker
from app.rag.embeddings import Embedder, get_embedder
from app.rag.reranker import Reranker, get_reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


@lru_cache
def get_store() -> VectorStore:
    s = get_settings()
    return VectorStore(s.chroma_dir, s.chroma_collection)


def get_embedder_dep() -> Embedder:
    return get_embedder(get_settings().embedding_model)


def get_reranker_dep() -> Reranker:
    return get_reranker(get_settings().reranker_model)


def get_chunker() -> DocumentChunker:
    s = get_settings()
    return DocumentChunker(s.chunk_size, s.chunk_overlap)


def get_retriever(
    store: VectorStore = Depends(get_store),
    embedder: Embedder = Depends(get_embedder_dep),
    reranker: Reranker = Depends(get_reranker_dep),
) -> Retriever:
    s = get_settings()
    return Retriever(store, embedder, reranker, s.search_top_k, s.rerank_top_n, s.min_rerank_score)


@lru_cache
def get_agent() -> Agent:
    s = get_settings()
    return Agent(s.groq_api_key, s.llm_model, s.llm_timeout_seconds, s.max_history_turns)
