from fastapi import APIRouter, Depends, Query

from app.core.rate_limit import rate_limit
from app.deps import get_retriever
from app.rag.retriever import Retriever
from app.schemas.models import ChunkResult, SearchResponse

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse, dependencies=[Depends(rate_limit)])
def search(
    query: str = Query(min_length=2, max_length=1000),
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    chunks, abstained = retriever.retrieve(query)
    return SearchResponse(
        query=query, abstained=abstained, results=[ChunkResult(**c) for c in chunks]
    )
