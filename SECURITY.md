# Security notes

This project reduces common risks. It is **not** claimed to be fully secure.

## Threat model

| Threat | Control in this project |
|---|---|
| Prompt injection inside an uploaded document | Retrieved text is wrapped in `<context>` tags; the system prompt says to treat it as untrusted and never follow instructions in it. Tested in `tests/test_units.py`. |
| Direct prompt injection ("reveal your rules") | System prompt forbids revealing rules; two injection questions are in the evaluation set. |
| Malicious or huge uploads | Only `.pdf` / `.txt`, size cap (`MAX_UPLOAD_MB`), safe error messages for corrupt files. |
| API abuse / expensive requests | Per-IP rate limit, query length limits, history length limit, LLM timeout. |
| Secret leakage | Keys only via environment variables; `.env` is git-ignored and docker-ignored; errors never include provider messages; logs never contain document text. |
| SQL injection | Feedback uses parameterised SQLite queries; tested with a malicious string. |
| Feedback data exposure | `/feedback` is write-only; there is no endpoint that reads it. |
| Stack traces to users | Global handler returns a generic 500. |
| Running as root in the container | Dockerfile runs as a non-root user. |

## Known limitations

- No authentication: all users share one document pool and anyone with the link can upload.
- In-memory rate limiter works for a single process only (use Redis for several servers).
- Prompt-injection defence is a prompt rule plus structure, not a guarantee.
- `request_id` in feedback is not checked against past requests.
- Scanned PDFs, tables and non-English documents are not handled.

## Future improvements

User login with per-user documents, antivirus scanning for uploads, output filtering, Redis rate limiting, audit logs.
