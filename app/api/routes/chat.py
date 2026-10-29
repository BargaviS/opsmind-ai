from fastapi import APIRouter, Depends, HTTPException

from app.core.logger import get_logger
from app.core.rate_limit import rate_limit
from app.deps import get_agent, get_retriever
from app.rag.agent import ABSTAIN_MESSAGE, Agent
from app.rag.retriever import Retriever
from app.schemas.models import ChatRequest, ChatResponse, Source

router = APIRouter(tags=["chat"])
log = get_logger("chat")


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(
    req: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    agent: Agent = Depends(get_agent),
) -> ChatResponse:
    chunks, abstained = retriever.retrieve(req.query)

    # Guardrail: no relevant context => do NOT call the LLM (saves cost, prevents guessing).
    if abstained:
        log.info("abstained")
        return ChatResponse(answer=ABSTAIN_MESSAGE, sources=[], abstained=True)

    try:
        answer = agent.answer(
            req.query, retriever.build_context(chunks), [t.model_dump() for t in req.history]
        )
    except Exception as exc:  # never leak provider errors or keys to the client
        log.error("llm_failure type=%s", type(exc).__name__)
        raise HTTPException(503, "The AI service is temporarily unavailable. Try again.") from None

    # If the model itself said "I don't know", show no sources.
    if answer.strip() == ABSTAIN_MESSAGE:
        return ChatResponse(answer=answer, sources=[], abstained=True)

    return ChatResponse(
        answer=answer,
        abstained=False,
        sources=[
            Source(
                filename=c["filename"],
                chunk_index=c["chunk_index"],
                rerank_score=c["rerank_score"],
            )
            for c in chunks
        ],
    )
