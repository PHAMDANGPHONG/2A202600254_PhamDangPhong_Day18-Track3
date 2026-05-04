"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words using underthesea."""
    # Vietnamese word segmentation: "nghỉ phép" is 1 word, not 2
    # This is critical for BM25 accuracy on Vietnamese text
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        # Fallback if underthesea not installed
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks

        # Segment Vietnamese text then tokenize by whitespace
        self.corpus_tokens = []
        for chunk in chunks:
            segmented = segment_vietnamese(chunk["text"])
            tokens = segmented.split()
            self.corpus_tokens.append(tokens)

        # Build BM25 index
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        # Segment and tokenize the query
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                doc = self.documents[idx]
                results.append(SearchResult(
                    text=doc["text"],
                    score=float(scores[idx]),
                    metadata=doc.get("metadata", {}),
                    method="bm25"
                ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path="qdrant_db")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        # Recreate collection with proper vector config
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )

        # Encode all texts
        texts = [c["text"] for c in chunks]
        encoder = self._get_encoder()
        vectors = encoder.encode(texts, show_progress_bar=True)

        # Create points and upload
        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]}
            )
            for i, c in enumerate(chunks)
        ]

        # Upload in batches of 100
        batch_size = 100
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            self.client.upsert(collection_name=collection, points=batch)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        encoder = self._get_encoder()
        query_vector = encoder.encode(query).tolist()

        hits = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(SearchResult(
                text=payload.get("text", ""),
                score=hit.score,
                metadata={k: v for k, v in payload.items() if k != "text"},
                method="dense"
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    # Track RRF scores per unique document text
    rrf_scores: dict[str, dict] = {}  # text → {"score": float, "result": SearchResult}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            doc_key = result.text
            if doc_key not in rrf_scores:
                rrf_scores[doc_key] = {
                    "score": 0.0,
                    "result": result
                }
            rrf_scores[doc_key]["score"] += 1.0 / (k + rank + 1)

    # Sort by RRF score descending
    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    # Return top_k with method="hybrid"
    results = []
    for item in sorted_docs[:top_k]:
        original = item["result"]
        results.append(SearchResult(
            text=original.text,
            score=item["score"],
            metadata=original.metadata,
            method="hybrid"
        ))
    return results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
