from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "OpsMind AI"
    ENV: str = "development"

    UPLOAD_DIR: str = "data/uploads"
    CHROMA_DIR: str = "data/chroma"
    CHROMA_COLLECTION: str = "documents"

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    GEMINI_API_KEY: str = ""

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    SEARCH_TOP_K: int = 5

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
