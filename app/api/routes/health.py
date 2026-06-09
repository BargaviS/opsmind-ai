from fastapi import APIRouter
from app.schemas.models import HealthResponse
from app.core.config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health():
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        env=settings.ENV,
    )
