from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.deps import get_store
from app.rag.vector_store import VectorStore
from app.schemas.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings), store: VectorStore = Depends(get_store)
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        env=settings.env,
        documents=store.count_documents(),
    )
