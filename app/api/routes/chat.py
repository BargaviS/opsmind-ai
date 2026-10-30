from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.core.rate_limit import rate_limit
from app.deps import get_agent, get_retriever
from app.rag.agent import ABSTAIN_MESSAGE, Agent
from app.rag.retriever import Retriever
from app.schemas.models import ChatRequest, ChatResponse, Source

router = APIRouter(tags=["chat"])
log = get_logger("chat")


def _clean_query(req: ChatRequest, agent: Agent, settings: Settings, rid: str) -> str | None:
    """Ask the LLM to fix spelling / resolve 'it' in follow-ups. Falls back to the original."""
    if not settings.rewrite_queries:
        return None
    try:
        rewritten = agent.rewrite(req.query, [t.model_dump() for t in req.history])
    except Exception as exc:
        log.warning("rewrite_failed rid=%s type=%s", rid, type(exc).__name__)
        return None
    if not rewritten or len(rewritten) > 300:
        return None
    return rewritten


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(
    req: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    retriever: Retriever = Depends(get_retriever),
    agent: Agent = Depends(get_agent),
) -> ChatResponse:
    rid = request.state.request_id
    rewritten = _clean_query(req, agent, settings, rid)
    search_query = rewritten or req.query
    shown_rewrite = rewritten if rewritten and rewritten != req.query else None

    chunks, abstained = retriever.retrieve(search_query)

    # Guardrail: no relevant context => do NOT call the answering LLM (saves cost, prevents guessing).
    if abstained:
        log.info("abstained rid=%s", rid)
        return ChatResponse(
            answer=ABSTAIN_MESSAGE, sources=[], abstained=True, request_id=rid,
            rewritten_query=shown_rewrite,
        )

    try:
        answer = agent.answer(
            search_query, retriever.build_context(chunks), [t.model_dump() for t in req.history]
        )
    except Exception as exc:  # never leak provider errors or keys to the client
        log.error("llm_failure rid=%s type=%s", rid, type(exc).__name__)
        raise HTTPException(503, "The AI service is temporarily unavailable. Try again.") from None

    if answer.strip() == ABSTAIN_MESSAGE:  # model itself said "I don't know": show no sources
        return ChatResponse(
            answer=answer, sources=[], abstained=True, request_id=rid, rewritten_query=shown_rewrite
        )

    log.info("answered rid=%s chunks=%d", rid, len(chunks))
    return ChatResponse(
        answer=answer,
        abstained=False,
        request_id=rid,
        rewritten_query=shown_rewrite,
        sources=[
            Source(filename=c["filename"], chunk_index=c["chunk_index"], rerank_score=c["rerank_score"])
            for c in chunks
        ],
    )
