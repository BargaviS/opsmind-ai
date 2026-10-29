"""Stage 2: a CrossEncoder reads (question, chunk) TOGETHER and scores relevance.

Slower than vector search but much more precise, so we only run it on the
few candidates Stage 1 found. Scores are passed through a sigmoid, so they
are probabilities between 0 and 1.
"""
from functools import lru_cache


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy import

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        import torch

        scores = self._load().predict(
            [(query, t) for t in texts], activation_fct=torch.nn.Sigmoid()
        )
        return [float(s) for s in scores]


@lru_cache
def get_reranker(model_name: str) -> Reranker:
    return Reranker(model_name)
