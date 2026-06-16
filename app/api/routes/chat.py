from fastapi import APIRouter, HTTPException, Depends

from app.schemas.models import ChatRequest, ChatResponse, ChatMessage, SearchResult
from app.rag.vector_store import VectorStore, get_vector_store
from app.rag.retriever import Retriever
from app.rag.agent import AgentService
from app.core.logger import get_logger

logger = get_logger("opsmind.route.chat")

router = APIRouter(tags=["Agent"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Conversational RAG agent endpoint.
    Maintains multi-turn history and answers from uploaded documents.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Retrieve relevant chunks
    retriever = Retriever(vector_store)
    chunks = retriever.retrieve(request.query, top_k=request.top_k)
    context = retriever.build_context(chunks) if chunks else ""

    history = [
        {"role": m.role, "content": m.content}
        for m in (request.history or [])
    ]

    try:
        agent = AgentService()
        answer = agent.chat(
            query=request.query,
            context=context,
            history=history,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=502, detail=f"Agent error: {str(e)}")

    updated_history = list(request.history or []) + [
        ChatMessage(role="user", content=request.query),
        ChatMessage(role="model", content=answer),
    ]

    sources = [
        SearchResult(
            chunk=c["chunk"],
            document_id=c["document_id"],
            score=c.get("score"),
        )
        for c in chunks
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        history=updated_history,
    )
