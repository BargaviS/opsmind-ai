from pydantic import BaseModel
from typing import List, Optional


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunks_stored: int
    message: str


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SearchResult(BaseModel):
    chunk: str
    document_id: str
    score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult]


class HealthResponse(BaseModel):
    status: str
    app: str
    env: str
