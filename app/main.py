from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.api import api_router
from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.vector_store import get_vector_store

logger = get_logger("opsmind.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} [{settings.ENV}]")
    get_vector_store()
    logger.info("Vector store warmed up")
    yield
    logger.info("Shutting down OpsMind AI")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="RAG-powered document intelligence agent",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    static_dir = os.path.join(os.path.dirname(__file__), "static")

    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def root():
        index_path = os.path.join(static_dir, "index.html")
        return FileResponse(index_path)

    return app


app = create_app()