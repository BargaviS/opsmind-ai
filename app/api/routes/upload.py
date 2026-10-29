import hashlib

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.core.rate_limit import rate_limit
from app.deps import get_chunker, get_embedder_dep, get_store
from app.rag.chunker import DocumentChunker
from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorStore
from app.schemas.models import UploadResponse
from app.services.extractor import extract_text
from app.services.storage import save_upload

router = APIRouter(tags=["documents"])
log = get_logger("upload")


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(rate_limit)])
def upload(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    store: VectorStore = Depends(get_store),
    embedder: Embedder = Depends(get_embedder_dep),
    chunker: DocumentChunker = Depends(get_chunker),
) -> UploadResponse:
    filename = (file.filename or "").strip() or "unnamed"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)  # read one extra byte to detect "too big"
    if len(data) > max_bytes:
        raise HTTPException(413, f"File too large. Maximum is {settings.max_upload_mb} MB.")

    # Same content => same ID, so re-uploading the same file is detected.
    document_id = hashlib.sha256(data).hexdigest()[:16]
    if store.has_document(document_id):
        return UploadResponse(
            document_id=document_id, filename=filename, chunks=0, duplicate=True,
            message="This document is already uploaded. Nothing was added.",
        )

    try:
        text = extract_text(filename, data)
    except ValueError as exc:  # safe, user-facing messages only
        raise HTTPException(422, str(exc)) from None

    chunks = chunker.split(text)
    store.add_chunks(document_id, filename, chunks, embedder.embed_documents(chunks))
    save_upload(settings.upload_dir, document_id, filename, data)
    log.info("ingested document_id=%s chunks=%d", document_id, len(chunks))  # no text logged
    return UploadResponse(
        document_id=document_id, filename=filename, chunks=len(chunks),
        message="Document uploaded and indexed.",
    )
