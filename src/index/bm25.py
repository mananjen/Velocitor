from src.data import get_doc_texts, load_fiqa_dev
from src.retrieval.bm25 import BM25Retriever


def build_fiqa_bm25_retriever() -> tuple[BM25Retriever, dict[str, str]]:
    corpus, _, _ = load_fiqa_dev()
    doc_texts = get_doc_texts(corpus)
    retriever = BM25Retriever(doc_texts)
    return retriever, doc_texts