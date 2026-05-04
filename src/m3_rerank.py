"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = FlashrankReranker()
            self._model_type = "flashrank"
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()

        # Create query-document pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Compute reranking scores
        if self._model_type == "flashrank":
            return self._model.rerank(query, documents, top_k)

        # Combine scores with documents
        scored_docs = list(zip(scores, documents))

        # Sort by rerank score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return top-k results
        results = []
        for rank, (score, doc) in enumerate(scored_docs[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=rank
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank using Flashrank (lightweight, fast)."""
        if not documents:
            return []

        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()

            passages = [{"text": d["text"]} for d in documents]
            request = RerankRequest(query=query, passages=passages)
            flash_results = self._model.rerank(request)

            results = []
            for rank, r in enumerate(flash_results[:top_k]):
                # Find matching document for metadata
                doc_meta = {}
                doc_score = 0.0
                for d in documents:
                    if d["text"] == r.get("text", r.get("passage", "")):
                        doc_meta = d.get("metadata", {})
                        doc_score = d.get("score", 0.0)
                        break

                results.append(RerankResult(
                    text=r.get("text", r.get("passage", "")),
                    original_score=doc_score,
                    rerank_score=float(r.get("score", 0.0)),
                    metadata=doc_meta,
                    rank=rank
                ))
            return results
        except ImportError:
            return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to milliseconds
        times.append(elapsed)

    return {
        "avg_ms": round(sum(times) / len(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
