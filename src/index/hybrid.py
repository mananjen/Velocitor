from src.index.bm25 import build_fiqa_bm25_retriever
from src.index.dense import build_fiqa_dense_retriever
from src.retrieval.hybrid import HybridRetriever


def build_fiqa_hybrid_retriever(
    fetch_k: int = 100,
    rrf_k: int = 60,
) -> tuple[HybridRetriever, dict[str, str]]:
    bm25_retriever, doc_texts = build_fiqa_bm25_retriever()
    dense_retriever, _ = build_fiqa_dense_retriever()

    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        fetch_k=fetch_k,
        rrf_k=rrf_k,
    )
    return retriever, doc_texts