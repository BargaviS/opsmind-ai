from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from app.schemas.models import UploadResponse
from app.services.storage import FileStorageService
from app.services.extractor import TextExtractor
from app.rag.chunker import DocumentChunker
from app.rag.vector_store import VectorStore, get_vector_store
from app.core.logger import get_logger

logger = get_logger("opsmind.route.upload")

router = APIRouter(tags=["Documents"])

ALLOWED_TYPES = {"txt", "pdf"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Full ingestion pipeline:
    1. Validate file type
    2. Save to disk
    3. Extract text
    4. Chunk text
    5. Store chunks in vector DB
    """
    filename = file.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{extension}'. Allowed: {ALLOWED_TYPES}",
        )

    # 1. Save to disk
    try:
        storage = FileStorageService()
        file_meta = await storage.save(file)
    except Exception as e:
        logger.error(f"File save failed: {e}")
        raise HTTPException(status_code=500, detail=f"File storage error: {str(e)}")

    # 2. Extract text
    try:
        extractor = TextExtractor()
        text = extractor.extract(file_meta["file_path"], file_meta["file_type"])
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise HTTPException(status_code=422, detail=f"Text extraction error: {str(e)}")

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Extracted text is empty. The file may be blank or image-only.",
        )

    # 3. Chunk
    try:
        chunker = DocumentChunker()
        chunks = chunker.split(text)
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunking error: {str(e)}")

    # 4. Store in vector DB
    try:
        vector_store.add_chunks(
            document_id=file_meta["file_id"],
            filename=file_meta["filename"],
            chunks=chunks,
        )
    except Exception as e:
        logger.error(f"Vector store write failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector store error: {str(e)}")

    logger.info(
        f"Ingestion complete: '{filename}' → {len(chunks)} chunks stored "
        f"(id={file_meta['file_id']})"
    )

    return UploadResponse(
        document_id=file_meta["file_id"],
        filename=file_meta["filename"],
        file_type=file_meta["file_type"],
        chunks_stored=len(chunks),
        message="Document ingested successfully",
    )
