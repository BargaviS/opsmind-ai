"""Measure retrieval quality on evaluation/questions.jsonl. No LLM, no API cost.

Run:  PYTHONPATH=. ENV=test python scripts/evaluate.py data/sample_hr_policy.txt

It ingests the document into a TEMPORARY database, runs every question through
the real embedder + reranker, then reports for several thresholds:
  - Hit@k (answerable): did a retrieved chunk contain the expected fact?
  - Answered rate (answerable): how many were NOT wrongly abstained?
  - Abstain rate (unanswerable): how many were correctly refused?
and latency (average and P95). It also prints the score ranges so you can
CALIBRATE MIN_RERANK_SCORE with evidence instead of guessing.
"""
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

from app.core.config import get_settings
from app.rag.chunker import DocumentChunker
from app.rag.embeddings import Embedder
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.services.extractor import extract_text

THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]


def main(doc_path: str) -> None:
    s = get_settings()
    text = extract_text(Path(doc_path).name, Path(doc_path).read_bytes())
    chunks = DocumentChunker(s.chunk_size, s.chunk_overlap).split(text)
    embedder, reranker = Embedder(s.embedding_model), Reranker(s.reranker_model)

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(tmp, "eval")
        store.add_chunks("evaldoc", Path(doc_path).name, chunks, embedder.embed_documents(chunks))
        retr = Retriever(store, embedder, reranker, s.search_top_k, s.rerank_top_n, 0.0)

        rows, latencies = [], []
        for line in Path("evaluation/questions.jsonl").read_text().splitlines():
            item = json.loads(line)
            t0 = time.perf_counter()
            ranked, _ = retr.retrieve(item["q"])  # threshold 0 => we see raw scores
            latencies.append((time.perf_counter() - t0) * 1000)
            best = ranked[0]["rerank_score"] if ranked else 0.0
            hit = any(item["expect"].lower() in c["text"].lower() for c in ranked)
            best_vec = max((c["vector_score"] for c in ranked), default=0.0)
            rows.append({**item, "best_score": best, "best_vec": best_vec, "hit": hit})

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    report = {
        "chunks": len(chunks),
        "questions": len(rows),
        "hit_at_n": round(sum(r["hit"] for r in ans) / len(ans), 3),
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
        "score_range_answerable": [round(min(r["best_score"] for r in ans), 3),
                                   round(max(r["best_score"] for r in ans), 3)],
        "score_range_unanswerable": [round(min(r["best_score"] for r in unans), 3),
                                     round(max(r["best_score"] for r in unans), 3)],
        "by_threshold": {
            str(t): {
                "answerable_answered": round(sum(r["best_score"] >= t for r in ans) / len(ans), 3),
                "unanswerable_abstained": round(sum(r["best_score"] < t for r in unans) / len(unans), 3),
            }
            for t in THRESHOLDS
        },
    }
    report["wrongly_abstained"] = [r["q"] for r in ans if r["best_score"] < 0.05]
    report["vector_range_answerable"] = [
        round(min(r["best_vec"] for r in ans), 3), round(max(r["best_vec"] for r in ans), 3)]
    report["vector_range_unanswerable"] = [
        round(min(r["best_vec"] for r in unans), 3), round(max(r["best_vec"] for r in unans), 3)]
    report["hybrid_rerank_or_vector"] = {}
    for v in (0.5, 0.6, 0.7, 0.8):
        ok = lambda r: r["best_score"] >= 0.05 or r["best_vec"] >= v  # noqa: E731
        report["hybrid_rerank_or_vector"][str(v)] = {
            "answerable_answered": round(sum(ok(r) for r in ans) / len(ans), 3),
            "unanswerable_abstained": round(sum(not ok(r) for r in unans) / len(unans), 3),
        }
    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nPick the threshold where BOTH numbers are high. Set it as MIN_RERANK_SCORE.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_hr_policy.txt")
