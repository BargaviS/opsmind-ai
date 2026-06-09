from fastapi import APIRouter, HTTPException, Depends, Query

from app.schemas.models import SearchResponse, SearchResult
from app.rag.vector_store import VectorStore, get_vector_store
from app.rag.retriever import Retriever
from app.rag.llm import LLMService
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger("opsmind.route.search")

router = APIRouter(tags=["Search"])


@router.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1, description="Your question"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of chunks to retrieve"),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    RAG search pipeline:
    1. Embed query and retrieve top-k relevant chunks
    2. Build context from chunks
    3. Generate grounded answer via LLM
    4. Return answer + source attribution
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Retrieve relevant chunks
    retriever = Retriever(vector_store)
    chunks = retriever.retrieve(query, top_k=top_k)

    if not chunks:
        return SearchResponse(
            query=query,
            answer="No documents have been uploaded yet. Please upload documents first.",
            sources=[],
        )

    # 2. Build context
    context = retriever.build_context(chunks)

    # 3. Generate answer via LLM
    try:
        llm = LLMService()
        answer = llm.answer(query=query, context=context)
    except RuntimeError as e:
        # API key not set — fall back to returning raw chunks
        logger.warning(f"LLM unavailable: {e}. Returning raw chunks.")
        answer = (
            "LLM is not configured (ANTHROPIC_API_KEY missing). "
            "Showing raw retrieved chunks below."
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    # 4. Build response with source attribution
    sources = [
        SearchResult(
            chunk=c["chunk"],
            document_id=c["document_id"],
            score=c.get("score"),
        )
        for c in chunks
    ]

    return SearchResponse(query=query, answer=answer, sources=sources)
