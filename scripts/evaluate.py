"""Measure retrieval quality. No answering-LLM calls, no cost.

Run (from the project folder):
    PYTHONPATH=. ENV=test python scripts/evaluate.py
    PYTHONPATH=. python scripts/evaluate.py --rewrite     # also tests query rewriting (needs GROQ_API_KEY)

It ingests every data/sample_*.txt into a TEMPORARY database, runs all questions in
evaluation/questions.jsonl through the real embedder + reranker, and tries several chunk
sizes and thresholds. It writes evaluation/results.json, results.csv and report.md.

Three strategies are compared:
  A. Vector search only        -> Hit@3 (vector order)
  B. Vector search + reranking -> Hit@3 and MRR (reranked order)
  C. B + relevance threshold   -> answered correctly / refused out-of-scope
"""
import csv
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

from app.core.config import get_settings
from app.rag.agent import Agent
from app.rag.chunker import DocumentChunker
from app.rag.embeddings import Embedder
from app.rag.reranker import Reranker
from app.rag.vector_store import VectorStore
from app.services.extractor import extract_text

CHUNK_SIZES = [120, 200, 300, 500]
THRESHOLDS = [0.02, 0.05, 0.10, 0.20]


def load_questions() -> list[dict]:
    lines = Path("evaluation/questions.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_store(tmp: str, chunk_size: int, embedder) -> tuple[VectorStore, int]:
    store = VectorStore(tmp, f"eval_{chunk_size}")
    chunker = DocumentChunker(chunk_size, chunk_size // 5)
    total = 0
    for doc in sorted(Path("data").glob("sample_*.txt")):
        chunks = chunker.split(extract_text(doc.name, doc.read_bytes()))
        store.add_chunks(doc.stem, doc.name, chunks, embedder.embed_documents(chunks))
        total += len(chunks)
    return store, total


def run_question(item: dict, store, embedder, reranker, top_k: int, query: str | None = None) -> dict:
    q = query or item["q"]
    t0 = time.perf_counter()
    cands = store.search(embedder.embed_query(q), top_k)
    scores = reranker.score(q, [c["text"] for c in cands]) if cands else []
    ms = (time.perf_counter() - t0) * 1000
    expect = item["expect"].lower()

    def first_hit(order: list[dict]) -> int | None:  # 1-based rank of the first chunk holding the fact
        for i, c in enumerate(order, 1):
            if expect and expect in c["text"].lower():
                return i
        return None

    reranked = [cands[i] for i in sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)]
    return {
        "category": item["category"], "answerable": item["answerable"],
        "vec_rank": first_hit(cands), "rr_rank": first_hit(reranked),
        "best": max(scores, default=0.0), "ms": ms,
    }


def summarize(results: list[dict], threshold: float) -> dict:
    ans = [r for r in results if r["answerable"]]
    out = [r for r in results if not r["answerable"]]

    def within(r: dict, key: str, k: int = 3) -> bool:
        return r[key] is not None and r[key] <= k

    def pct(values) -> float:
        values = list(values)
        return round(sum(values) / len(values), 3) if values else 0.0

    return {
        "hit3_vector": pct(within(r, "vec_rank") for r in ans),
        "hit3_rerank": pct(within(r, "rr_rank") for r in ans),
        "mrr_rerank": round(statistics.mean((1 / r["rr_rank"]) if r["rr_rank"] else 0 for r in ans), 3),
        "answered_correctly": pct(r["best"] >= threshold and within(r, "rr_rank") for r in ans),
        "refused_out_of_scope": pct(r["best"] < threshold for r in out),
    }


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1)


def main(use_rewrite: bool) -> None:
    s = get_settings()
    questions = load_questions()
    embedder, reranker = Embedder(s.embedding_model), Reranker(s.reranker_model)
    embedder.embed_query("warm up")  # the first call loads the model; keep it out of the timings
    reranker.score("warm up", ["warm up"])

    sizes = sorted(set(CHUNK_SIZES + [s.chunk_size]))
    thresholds = sorted(set(THRESHOLDS + [s.min_rerank_score]))
    rows, per_size, latencies = [], {}, {}
    for cs in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            store, n_chunks = build_store(tmp, cs, embedder)
            results = [run_question(q, store, embedder, reranker, s.search_top_k) for q in questions]
            per_size[cs] = (results, n_chunks)
            latencies[cs] = [r["ms"] for r in results]
            if use_rewrite and cs == s.chunk_size:
                agent = Agent(s.groq_api_key, s.llm_model, s.llm_timeout_seconds, s.max_history_turns)
                typo = [q for q in questions if q["category"] == "typo"]
                rewritten = [
                    run_question(q, store, embedder, reranker, s.search_top_k,
                                 query=agent.rewrite(q["q"], []) or q["q"])
                    for q in typo
                ]
        for t in thresholds:
            rows.append({"chunk_size": cs, "chunks": n_chunks, "threshold": t, **summarize(results, t)})

    default = next(r for r in rows if r["chunk_size"] == s.chunk_size and r["threshold"] == s.min_rerank_score)
    results, _ = per_size[s.chunk_size]
    by_category = {}
    for cat in sorted({r["category"] for r in results}):
        group = [r for r in results if r["category"] == cat]
        ok = [(r["best"] >= s.min_rerank_score) == r["answerable"] and (not r["answerable"] or r["rr_rank"] is not None
              and r["rr_rank"] <= 3) for r in group]
        by_category[cat] = {"questions": len(group), "handled_correctly": round(sum(ok) / len(group), 3)}

    report = {
        "questions": len(questions), "default_chunk_size": s.chunk_size,
        "default_threshold": s.min_rerank_score, "default": default, "by_category_at_default": by_category,
        "latency_ms_avg": round(statistics.mean(latencies[s.chunk_size]), 1),
        "latency_ms_p95": p95(latencies[s.chunk_size]), "grid": rows,
    }
    if use_rewrite:
        before = summarize([r for r in results if r["category"] == "typo"], s.min_rerank_score)
        after = summarize(rewritten, s.min_rerank_score)
        report["typo_without_rewrite"] = before["answered_correctly"]
        report["typo_with_rewrite"] = after["answered_correctly"]

    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/results.json").write_text(json.dumps(report, indent=2))
    with open("evaluation/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path("evaluation/report.md").write_text(render_report(report))
    print(Path("evaluation/report.md").read_text())


def render_report(r: dict) -> str:
    pc = lambda v: f"{v * 100:.0f}%"  # noqa: E731
    d = r["default"]
    lines = [
        "# Evaluation report", "",
        f"{r['questions']} labelled questions over 2 synthetic documents "
        f"(answerable, typo/weak-English, out-of-scope, prompt-injection). Real embedder and reranker; "
        "the answering LLM is not called. These numbers were produced by `scripts/evaluate.py`.", "",
        f"## Default configuration: chunk size {r['default_chunk_size']}, threshold {r['default_threshold']}", "",
        "| Strategy | Metric | Result |", "|---|---|---|",
        f"| A. Vector search only | Hit@3 | {pc(d['hit3_vector'])} |",
        f"| B. Vector + reranking | Hit@3 | {pc(d['hit3_rerank'])} |",
        f"| B. Vector + reranking | MRR | {d['mrr_rerank']} |",
        f"| C. B + threshold | In-scope questions answered with the right chunk | {pc(d['answered_correctly'])} |",
        f"| C. B + threshold | Out-of-scope / injection questions refused | {pc(d['refused_out_of_scope'])} |",
        f"| Latency (embed + search + rerank, CPU) | average / p95 | {r['latency_ms_avg']} ms / {r['latency_ms_p95']} ms |",
        "", "### By question type (default configuration)", "", "| Type | Questions | Handled correctly |", "|---|---|---|",
    ]
    lines += [f"| {k} | {v['questions']} | {pc(v['handled_correctly'])} |" for k, v in r["by_category_at_default"].items()]
    if "typo_with_rewrite" in r:
        lines += ["", "### Effect of query rewriting on typo questions", "",
                  f"Without rewrite: {pc(r['typo_without_rewrite'])}. With rewrite: {pc(r['typo_with_rewrite'])}."]
    lines += ["", "## All settings tried", "",
              "| Chunk size | Chunks | Threshold | Hit@3 vector | Hit@3 rerank | MRR | Answered correctly | Refused out-of-scope |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [
        f"| {g['chunk_size']} | {g['chunks']} | {g['threshold']} | {pc(g['hit3_vector'])} | {pc(g['hit3_rerank'])} | "
        f"{g['mrr_rerank']} | {pc(g['answered_correctly'])} | {pc(g['refused_out_of_scope'])} |"
        for g in r["grid"]
    ]
    lines += ["", "## Limitations", "",
              "- Small, synthetic benchmark (2 short documents). Results will differ on real, longer documents.",
              "- The expected fact is matched by text, so this measures retrieval, not answer quality or groundedness.",
              "- The threshold was tuned on this same set, so it is optimistic. Re-tune on your own documents.",
              "- Latency excludes the LLM calls and was measured on one machine."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main("--rewrite" in sys.argv)
