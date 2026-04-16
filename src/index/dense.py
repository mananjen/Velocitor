from src.data import get_doc_texts, load_fiqa_dev
from src.retrieval.dense import DenseRetriever


def build_fiqa_dense_retriever(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[DenseRetriever, dict[str, str]]:
    corpus, _, _ = load_fiqa_dev()
    doc_texts = get_doc_texts(corpus)
    retriever = DenseRetriever(doc_texts=doc_texts, model_name=model_name)
    return retriever, doc_texts