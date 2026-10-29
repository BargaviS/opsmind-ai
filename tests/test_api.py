from tests.conftest import SAMPLE


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_creates_chunks_and_metadata(client):
    r = client.post("/upload", files={"file": ("hr.txt", SAMPLE.encode(), "text/plain")})
    body = r.json()
    assert r.status_code == 200 and body["chunks"] >= 1 and not body["duplicate"]
    assert len(body["document_id"]) == 16


def test_duplicate_upload_is_detected(client, uploaded):
    r = client.post("/upload", files={"file": ("copy.txt", SAMPLE.encode(), "text/plain")})
    assert r.json()["duplicate"] is True and r.json()["chunks"] == 0
    assert client.get("/health").json()["documents"] == 1


def test_unsupported_file_rejected(client):
    r = client.post("/upload", files={"file": ("x.exe", b"abc", "application/octet-stream")})
    assert r.status_code == 422


def test_empty_file_rejected(client):
    r = client.post("/upload", files={"file": ("e.txt", b"  ", "text/plain")})
    assert r.status_code == 422


def test_corrupt_pdf_rejected(client):
    r = client.post("/upload", files={"file": ("c.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 422 and "corrupt" in r.json()["detail"]


def test_oversized_file_rejected(make_client):
    client = make_client(max_upload_mb=1)
    r = client.post("/upload", files={"file": ("big.txt", b"a" * (1024 * 1024 + 1), "text/plain")})
    assert r.status_code == 413


def test_chat_answers_with_sources(client, uploaded, agent):
    r = client.post("/chat", json={"query": "How many days of annual leave do I get?"})
    body = r.json()
    assert r.status_code == 200 and not body["abstained"]
    assert body["sources"] and body["sources"][0]["filename"] == "hr.txt"
    assert agent.calls == 1


def test_chat_abstains_without_calling_llm(client, uploaded, agent):
    r = client.post("/chat", json={"query": "Who won the football championship?"})
    body = r.json()
    assert body["abstained"] is True and body["sources"] == []
    assert agent.calls == 0  # guardrail saved an LLM call


def test_chat_with_no_documents_abstains(client, agent):
    r = client.post("/chat", json={"query": "How many sick leaves are there?"})
    assert r.json()["abstained"] is True and agent.calls == 0


def test_llm_failure_returns_safe_503(client, uploaded, agent):
    agent.fail = True
    r = client.post("/chat", json={"query": "How many days of annual leave?"})
    assert r.status_code == 503
    assert "secret-key-123" not in r.text


def test_chat_validates_input(client):
    assert client.post("/chat", json={"query": "x"}).status_code == 422
    bad = {"query": "hello there", "history": [{"role": "hacker", "content": "hi"}]}
    assert client.post("/chat", json=bad).status_code == 422


def test_search_endpoint(client, uploaded):
    r = client.get("/search", params={"query": "notice period for senior staff"})
    body = r.json()
    assert r.status_code == 200 and not body["abstained"]
    assert body["results"][0]["rerank_score"] >= 0.3


def test_rate_limit_blocks_abusive_client(make_client):
    client = make_client(rate_limit_per_minute=2)
    codes = [client.get("/search", params={"query": "leave days"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
