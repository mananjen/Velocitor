from typing import Dict, Iterable, List, Sequence


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    hit = any(doc_id in relevant_set for doc_id in top_k)
    return 1.0 if hit else 0.0


def reciprocal_rank(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str]) -> float:
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def mean_score(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def qrels_to_relevant_doc_ids(qrels_for_query: Dict[str, int]) -> List[str]:
    return [doc_id for doc_id, relevance in qrels_for_query.items() if relevance > 0]