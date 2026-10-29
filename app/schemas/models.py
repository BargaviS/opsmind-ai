"""Pydantic request/response models: validation + automatic API docs."""
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    env: str
    documents: int


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    duplicate: bool = False
    message: str


class ChunkResult(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    vector_score: float
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    abstained: bool


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    filename: str
    chunk_index: int
    rerank_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    abstained: bool
