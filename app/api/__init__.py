from fastapi import APIRouter
from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router
from app.api.routes.search import router as search_router
from app.api.routes.chat import router as chat_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(upload_router)
api_router.include_router(search_router)
api_router.include_router(chat_router)
