# OpsMind AI

> Production-grade RAG (Retrieval Augmented Generation) system — upload documents and ask AI-powered questions.

## What it does

Upload any PDF or TXT document. Ask questions in plain English. Get accurate AI answers with source attribution — powered by Google Gemini and semantic search.

## Architecture
Document Upload
│
▼
Text Extraction (PDF/TXT)
│
▼
Chunking (LangChain RecursiveCharacterTextSplitter)
│
▼
Embedding (BAAI/bge-small-en-v1.5 via Sentence Transformers)
│
▼
Vector Store (ChromaDB — persisted locally)
│
▼
Semantic Search → Context Builder → Gemini LLM → Answer + Sources
## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers (BAAI/bge-small-en-v1.5) |
| LLM | Google Gemini 2.0 Flash |
| Text Splitting | LangChain Text Splitters |
| PDF Parsing | PyPDF |
| Validation | Pydantic v2 |
| Config | Pydantic Settings |

## Project Structure
app/
├── main.py                  # FastAPI app — startup, CORS, routing
├── core/
│   ├── config.py            # All settings via environment variables
│   └── logger.py            # Structured logging
├── api/routes/
│   ├── health.py            # Health check
│   ├── upload.py            # Document ingestion pipeline
│   └── search.py            # RAG search + LLM answer
├── rag/
│   ├── chunker.py           # Document chunking
│   ├── embeddings.py        # Vector embeddings (singleton)
│   ├── vector_store.py      # ChromaDB wrapper (singleton)
│   ├── retriever.py         # Semantic retrieval + context builder
│   └── llm.py               # Gemini LLM integration
├── services/
│   ├── storage.py           # File persistence
│   └── extractor.py         # PDF/TXT text extraction
└── schemas/
└── models.py            # Request/response Pydantic models
## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /upload | Upload PDF or TXT document |
| GET | /search?query= | Ask a question, get AI answer |

## Setup

```bash
git clone https://github.com/BargaviS/opsmind-ai.git
cd opsmind-ai

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Add your GEMINI_API_KEY in .env
```

## Run

```bash
PYTHONPATH=. uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive API.

## Environment Variables

| Variable | Description |
|----------|-------------|
| GEMINI_API_KEY | Your Google Gemini API key |
| EMBEDDING_MODEL | Sentence transformer model name |
| CHUNK_SIZE | Token size per chunk (default 500) |
| CHUNK_OVERLAP | Overlap between chunks (default 100) |
| SEARCH_TOP_K | Number of chunks to retrieve (default 5) |

## Key Design Decisions

- **Singleton pattern** for VectorStore and EmbeddingService — model loads once at startup, not per request
- **Dependency injection** via FastAPI `Depends()` for clean testability
- **Metadata stored per chunk** — every search result includes source filename and similarity score
- **Graceful fallback** — if LLM is unavailable, raw chunks are returned instead of crashing
- **Full error handling** at every pipeline stage with proper HTTP status codes
