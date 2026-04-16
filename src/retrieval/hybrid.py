from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    rrf_k: int = 60,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (rrf_k + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return fused[:top_k]


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        fetch_k: int = 100,
        rrf_k: int = 60,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        bm25_results = self.bm25_retriever.search(query, top_k=self.fetch_k)
        dense_results = self.dense_retriever.search(query, top_k=self.fetch_k)

        return reciprocal_rank_fusion(
            ranked_lists=[bm25_results, dense_results],
            rrf_k=self.rrf_k,
            top_k=top_k,
        )