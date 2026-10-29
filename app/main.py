"""FastAPI entry point: validates config at startup, wires routes, serves the UI."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, health, search, upload
from app.core.config import get_settings
from app.core.logger import get_logger

log = get_logger("main")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.validate_runtime()  # fail fast, with a clear message
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    log.info("started app=%s env=%s", settings.app_name, settings.env)
    yield


app = FastAPI(title="OpsMind AI — HR Policy Bot", version="2.0.0", lifespan=lifespan)

for r in (health.router, upload.router, search.router, chat.router):
    app.include_router(r)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled type=%s", type(exc).__name__)  # no stack trace to the client
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
