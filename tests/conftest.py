"""Shared fixtures. Everything external (models, LLM) is replaced by fast,
deterministic fakes, so tests need no internet, no GPU and no paid API."""
import os

os.environ.setdefault("ENV", "test")

import math
import re

import pytest
from fastapi.testclient import TestClient

from app import deps
from app.core.config import Settings, get_settings
from app.core.rate_limit import reset_rate_limiter
from app.main import app
from app.rag.vector_store import VectorStore

SAMPLE = (
    "Leave Policy:\n- Annual Leave: Every employee gets 18 days of paid annual leave per year.\n"
    "- Sick Leave: Employees get 12 days of paid sick leave per year.\n\n"
    "Resignation Policy:\n- Notice period is 30 days for junior staff and 60 days for senior staff.\n"
    "- Salaries are credited on the last working day of every month.\n"
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class FakeEmbedder:
    """Hashed bag-of-words vectors: similar words => similar vectors."""

    DIM = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for w in _words(text):
            v[sum(map(ord, w)) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    """Score = share of meaningful query words found in the chunk."""

    def score(self, query, texts):
        q = {w for w in _words(query) if len(w) > 3}
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(_words(t))) / len(q) for t in texts]


class FakeAgent:
    def __init__(self):
        self.calls = 0
        self.fail = False
        self.rewrite_calls = 0
        self.rewrite_to = None      # what rewrite() should return (None = same as input)
        self.rewrite_fail = False

    def rewrite(self, query, history):
        self.rewrite_calls += 1
        if self.rewrite_fail:
            raise RuntimeError("rewrite boom")
        return self.rewrite_to or query

    def answer(self, query, context, history):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom (contains secret-key-123)")
        return "Based on the policy: " + context.splitlines()[1]


@pytest.fixture
def agent():
    return FakeAgent()


@pytest.fixture
def make_client(tmp_path, agent):
    store = VectorStore(str(tmp_path / "chroma"), "test_docs")

    def _make(**overrides) -> TestClient:
        settings = Settings(
            env="test", upload_dir=str(tmp_path / "uploads"), min_rerank_score=0.3,
            db_path=str(tmp_path / "app.db"), **overrides
        )
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[deps.get_store] = lambda: store
        app.dependency_overrides[deps.get_embedder_dep] = lambda: FakeEmbedder()
        app.dependency_overrides[deps.get_reranker_dep] = lambda: FakeReranker()
        app.dependency_overrides[deps.get_agent] = lambda: agent
        reset_rate_limiter()
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def uploaded(client):
    r = client.post("/upload", files={"file": ("hr.txt", SAMPLE.encode(), "text/plain")})
    assert r.status_code == 200
    return r.json()
