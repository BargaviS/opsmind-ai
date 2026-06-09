from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import api_router
from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.vector_store import get_vector_store

logger = get_logger("opsmind.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm up singletons so the first request isn't slow.
    Shutdown: clean up resources.
    """
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} [{settings.ENV}]")

    # Pre-load vector store (and its embedding model) at startup
    get_vector_store()
    logger.info("Vector store warmed up")

    yield

    logger.info("Shutting down OpsMind AI")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="RAG-powered document intelligence API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten this in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/", tags=["Root"])
    def root():
        return {
            "app": settings.APP_NAME,
            "env": settings.ENV,
            "docs": "/docs",
        }

    return app


app = create_app()
