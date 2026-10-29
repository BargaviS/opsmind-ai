"""All settings come from environment variables (or a local .env file).

Why pydantic-settings: values are type-checked once, in one place, and a
missing/invalid value fails at startup instead of in the middle of a request.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "OpsMind AI"
    env: str = "development"

    # Storage
    upload_dir: str = "data/uploads"
    chroma_dir: str = "data/chroma"
    chroma_collection: str = "documents"

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: float = 30.0

    # RAG tuning
    chunk_size: int = Field(500, ge=100, le=4000)
    chunk_overlap: int = Field(100, ge=0, le=1000)
    search_top_k: int = Field(5, ge=1, le=20)      # Stage 1 candidates
    rerank_top_n: int = Field(3, ge=1, le=10)      # Stage 2 survivors
    min_rerank_score: float = Field(0.10, ge=0.0, le=1.0)  # abstain below this

    # Safety limits
    max_upload_mb: int = Field(10, ge=1, le=50)
    rate_limit_per_minute: int = Field(30, ge=1)
    max_history_turns: int = 6

    def validate_runtime(self) -> None:
        """Called at startup. Fail fast with a clear message."""
        if self.env != "test" and not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Get a free key at https://console.groq.com "
                "and set it in .env (local) or as a Secret (Hugging Face)."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
