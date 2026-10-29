---
title: HR Policy Bot
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# HR Policy Bot (OpsMind AI)

Upload an HR policy (PDF/TXT), ask questions in plain English, get answers grounded **only** in your document, with sources. If the document does not contain the answer, the bot says so instead of guessing.

**Live demo:** `<your Hugging Face Space URL>`  |  **Code:** https://github.com/BargaviS/opsmind-ai

## How it works

```
UPLOAD  file -> validate (type, size) -> extract text -> SHA-256 duplicate check
             -> chunk (500/100) -> embed (bge-small) -> ChromaDB

CHAT    question -> embed -> Stage 1: vector search (top 5)
                          -> Stage 2: CrossEncoder rerank (top 3)
                          -> Guardrail: best score < MIN_RERANK_SCORE ? ABSTAIN (no LLM call)
                          -> Groq LLaMA 3.3 70B answers from context only
                          -> answer + sources
```

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env            # then put your free GROQ_API_KEY inside
PYTHONPATH=. uvicorn app.main:app --reload
```

Open http://localhost:8000 (chat UI) and http://localhost:8000/docs (API docs).
The first start downloads the two models (about 150 MB), so wait a minute.

## Test (no internet, no API key, no GPU needed)

```bash
pip install -r requirements-test.txt
ENV=test pytest -q
ruff check .
```

## Measure retrieval quality (real numbers for your resume)

```bash
PYTHONPATH=. ENV=test python scripts/evaluate.py data/sample_hr_policy.txt
```

Prints Hit@k, abstention accuracy at several thresholds, and latency, and writes `evaluation/results.json`.
Use it to choose `MIN_RERANK_SCORE`.

## Deploy to Hugging Face Spaces (free, public link)

1. Create a Space at https://huggingface.co/new-space, choose **Docker** and a blank template.
2. In the Space: **Settings > Variables and secrets > New secret** named `GROQ_API_KEY`.
3. Push the code:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
4. Wait for the build (about 5 to 10 minutes). Your link is `https://<user>-<space-name>.hf.space`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status + number of documents |
| POST | `/upload` | PDF/TXT, max 10 MB, duplicate-safe |
| GET | `/search?query=` | retrieval only (debugging) |
| POST | `/chat` | `{query, history}` returns `{answer, sources, abstained}` |

## Configuration

See `.env.example`. Everything (chunk size, top-k, threshold, upload limit, rate limit, models) is an environment variable, validated at startup.

## Security controls

File type and size limits, per-IP rate limit, Pydantic input validation, retrieved text is wrapped in `<context>` and treated as untrusted (prompt-injection rule in the system prompt), no stack traces or provider errors in responses, no document text or keys in logs, non-root container, secrets only via environment variables.
This reduces risk. It does not make the app fully secure.

## Known limitations

- Hugging Face free tier has an ephemeral disk: uploaded documents disappear on restart.
- Rate limiter is in-memory (single process). Use Redis for many servers.
- Retrieval ignores chat history (follow-ups like "what about paternity?" are searched alone).
- No authentication; all users share one document pool.
- Scanned (image-only) PDFs are not supported (no OCR).
- The abstain threshold is calibrated on a small, synthetic question set.
