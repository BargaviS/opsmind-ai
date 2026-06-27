---
title: HR Policy Bot
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 🤖 HR Policy Bot

> **AI-powered HR document assistant** — Upload your HR policies, employee handbooks, or company guidelines and ask questions in plain English. Get accurate, grounded answers instantly.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square&logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-purple?style=flat-square)
![RAG](https://img.shields.io/badge/Architecture-RAG_Pipeline-red?style=flat-square)

---

## 🎯 What It Does

HR teams deal with hundreds of policy questions daily. HR Policy Bot lets employees **upload any HR document and ask questions naturally** — the AI answers from the document only, never hallucinating.

**Example questions it answers:**
- *"How many days of annual leave do I get?"*
- *"What is the maternity leave policy?"*
- *"What is the notice period for senior staff?"*
- *"When is salary credited every month?"*

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────┐
                    │           HR Policy Bot              │
                    └─────────────────────────────────────┘

UPLOAD PIPELINE:
Document (PDF/TXT)
      │
      ▼
Text Extraction (PyPDF / plain text)
      │
      ▼
Chunking (LangChain RecursiveCharacterTextSplitter)
chunk_size=500, overlap=100
      │
      ▼
Embedding (BAAI/bge-small-en-v1.5 — local, no API cost)
      │
      ▼
Vector Store (ChromaDB — persisted, cosine similarity)

QUERY PIPELINE:
User Question
      │
      ▼
Stage 1: Semantic Search → Top 5 candidates (ChromaDB)
      │
      ▼
Stage 2: CrossEncoder Reranking → Top 3 precise results
         (cross-encoder/ms-marco-MiniLM-L-6-v2)
      │
      ▼
Context Builder → Formats chunks with source attribution
      │
      ▼
Groq LLaMA 3.3 70B → Grounded answer generation
      │
      ▼
Response + Source Attribution + Similarity Scores
```

---

## ⚡ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API Framework | FastAPI | Async, auto OpenAPI docs, Pydantic validation |
| Vector Database | ChromaDB | Persistent, cosine similarity, metadata support |
| Embeddings | Sentence Transformers (BAAI/bge-small-en-v1.5) | Local, no API cost, 384-dim vectors |
| Reranker | CrossEncoder (ms-marco-MiniLM-L-6-v2) | 2-stage retrieval for precise results |
| LLM | Groq — LLaMA 3.3 70B | Fast, free tier, reliable, no quota issues |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | Semantically aware chunking with overlap |
| PDF Parsing | PyPDF | Extract text from PDF documents |
| Validation | Pydantic v2 | Type-safe request/response schemas |
| Config | Pydantic Settings | Environment-based configuration |

---

## 🔑 Key Design Decisions

| Decision | Reason |
|----------|--------|
| **2-Stage Retrieval** | Stage 1 (semantic search) is fast but approximate. Stage 2 (CrossEncoder reranking) is precise. Together they give both speed and accuracy. |
| **Singleton Pattern** | VectorStore and EmbeddingService load once at startup — not per request. Critical for performance. |
| **Dependency Injection** | FastAPI `Depends()` for VectorStore — makes code testable and loosely coupled. |
| **Local Embeddings** | BAAI/bge-small-en-v1.5 runs on CPU — no API cost, no latency, no rate limits. |
| **Metadata per Chunk** | Every chunk stores document_id, filename, chunk_index — enables full source attribution. |
| **Layered Architecture** | Clean separation: api / rag / services / core — each layer has one responsibility. |

---

## 📁 Project Structure

```
hr-policy-bot/
├── app/
│   ├── main.py                    # FastAPI app — startup, CORS, static UI
│   ├── core/
│   │   ├── config.py              # All settings via environment variables
│   │   └── logger.py              # Structured logging (timestamped)
│   ├── api/
│   │   └── routes/
│   │       ├── health.py          # GET /health
│   │       ├── upload.py          # POST /upload — full ingestion pipeline
│   │       ├── search.py          # GET /search — semantic search
│   │       └── chat.py            # POST /chat — conversational agent
│   ├── rag/
│   │   ├── chunker.py             # Document chunking with overlap
│   │   ├── embeddings.py          # Local vector embeddings (singleton)
│   │   ├── vector_store.py        # ChromaDB wrapper (singleton)
│   │   ├── retriever.py           # 2-stage retrieval pipeline
│   │   ├── reranker.py            # CrossEncoder reranking
│   │   ├── llm.py                 # Groq LLM integration
│   │   └── agent.py               # Conversational agent with memory
│   ├── services/
│   │   ├── storage.py             # File persistence (UUID-based)
│   │   └── extractor.py           # PDF and TXT text extraction
│   ├── schemas/
│   │   └── models.py              # Pydantic request/response models
│   └── static/
│       └── index.html             # Chat UI (dark theme, no framework)
├── requirements.txt
├── .env.example
└── test_pipeline.py               # Smoke tests
```

---

## 🚀 Setup & Run

```bash
# 1. Clone
git clone https://github.com/BargaviS/opsmind-ai.git
cd opsmind-ai

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Add your GROQ_API_KEY in .env
# Get free key at: https://console.groq.com

# 5. Run
PYTHONPATH=. uvicorn app.main:app --reload
```

Open **http://localhost:8000** for the chat UI
Open **http://localhost:8000/docs** for API documentation

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with app info |
| `POST` | `/upload` | Upload PDF or TXT — full ingestion pipeline |
| `GET` | `/search?query=` | Semantic search with reranking |
| `POST` | `/chat` | Conversational agent with memory |

### Example: Upload
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@hr_policy.pdf"
```

### Example: Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How many days annual leave?", "history": [], "top_k": 5}'
```

---

## 🌍 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (free at console.groq.com) | required |
| `EMBEDDING_MODEL` | Sentence transformer model | BAAI/bge-small-en-v1.5 |
| `CHUNK_SIZE` | Characters per chunk | 500 |
| `CHUNK_OVERLAP` | Overlap between chunks | 100 |
| `SEARCH_TOP_K` | Candidates for retrieval | 5 |

---

## 💡 What I Would Add Next

- **Streaming responses** — word-by-word output like ChatGPT
- **Authentication** — JWT-based user sessions
- **Docker** — single command deployment
- **Pinecone** — swap ChromaDB for production-scale vector DB
- **Evaluation metrics** — RAGAS framework for RAG quality scoring

---

## 👩‍💻 Built By

**Bargavi S** — Aspiring GenAI Engineer

> *"I built this to understand production RAG systems from scratch — every layer from document ingestion, chunking, vector embeddings, 2-stage retrieval, to LLM answer generation."*

