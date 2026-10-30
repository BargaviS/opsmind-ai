# Evaluation report

50 labelled questions over 2 synthetic documents (answerable, typo/weak-English, out-of-scope, prompt-injection). Real embedder and reranker; the answering LLM is not called. These numbers were produced by `scripts/evaluate.py`.

## Default configuration: chunk size 200, threshold 0.05

| Strategy | Metric | Result |
|---|---|---|
| A. Vector search only | Hit@3 | 94% |
| B. Vector + reranking | Hit@3 | 100% |
| B. Vector + reranking | MRR | 0.986 |
| C. B + threshold | In-scope questions answered with the right chunk | 69% |
| C. B + threshold | Out-of-scope / injection questions refused | 100% |
| Latency (embed + search + rerank, CPU) | average / p95 | 49.0 ms / 53.8 ms |

### By question type (default configuration)

| Type | Questions | Handled correctly |
|---|---|---|
| injection | 2 | 100% |
| normal | 25 | 96% |
| out_of_scope | 12 | 100% |
| typo | 11 | 9% |

### Effect of query rewriting on typo questions

Without rewrite: 9%. With rewrite: 91%.

## All settings tried

| Chunk size | Chunks | Threshold | Hit@3 vector | Hit@3 rerank | MRR | Answered correctly | Refused out-of-scope |
|---|---|---|---|---|---|---|---|
| 120 | 32 | 0.02 | 97% | 97% | 0.94 | 78% | 100% |
| 120 | 32 | 0.05 | 97% | 97% | 0.94 | 75% | 100% |
| 120 | 32 | 0.1 | 97% | 97% | 0.94 | 75% | 100% |
| 120 | 32 | 0.2 | 97% | 97% | 0.94 | 72% | 100% |
| 200 | 20 | 0.02 | 94% | 100% | 0.986 | 72% | 100% |
| 200 | 20 | 0.05 | 94% | 100% | 0.986 | 69% | 100% |
| 200 | 20 | 0.1 | 94% | 100% | 0.986 | 67% | 100% |
| 200 | 20 | 0.2 | 94% | 100% | 0.986 | 67% | 100% |
| 300 | 13 | 0.02 | 97% | 97% | 0.965 | 69% | 100% |
| 300 | 13 | 0.05 | 97% | 97% | 0.965 | 64% | 100% |
| 300 | 13 | 0.1 | 97% | 97% | 0.965 | 64% | 100% |
| 300 | 13 | 0.2 | 97% | 97% | 0.965 | 64% | 100% |
| 500 | 6 | 0.02 | 94% | 100% | 0.986 | 69% | 100% |
| 500 | 6 | 0.05 | 94% | 100% | 0.986 | 64% | 100% |
| 500 | 6 | 0.1 | 94% | 100% | 0.986 | 64% | 100% |
| 500 | 6 | 0.2 | 94% | 100% | 0.986 | 64% | 100% |

## Limitations

- Small, synthetic benchmark (2 short documents). Results will differ on real, longer documents.
- The expected fact is matched by text, so this measures retrieval, not answer quality or groundedness.
- The threshold was tuned on this same set, so it is optimistic. Re-tune on your own documents.
- Latency excludes the LLM calls and was measured on one machine.
