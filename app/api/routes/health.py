from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.deps import get_store
from app.rag.vector_store import VectorStore
from app.schemas.models import HealthResponse
from app.services.feedback_store import count_feedback

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings), store: VectorStore = Depends(get_store)
) -> HealthResponse:
    """Liveness: the process is up."""
    return HealthResponse(
        status="ok", app=settings.app_name, env=settings.env, documents=store.count_documents()
    )


@router.get("/ready")
def ready(settings: Settings = Depends(get_settings), store: VectorStore = Depends(get_store)) -> dict:
    """Readiness: the vector store and the database are actually reachable."""
    try:
        store.count_documents()
        count_feedback(settings.db_path)
    except Exception:
        raise HTTPException(503, "Not ready.") from None
    return {"ready": True}
