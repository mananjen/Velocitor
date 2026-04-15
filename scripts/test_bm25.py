from src.data import get_doc_texts, load_fiqa_dev
from src.retrieval.bm25 import BM25Retriever


def main() -> None:
    corpus, _, _ = load_fiqa_dev()
    doc_texts = get_doc_texts(corpus)

    retriever = BM25Retriever(doc_texts)

    query = "what is short selling?"
    results = retriever.search(query, top_k=5)

    print(f"Query: {query}")
    print("-" * 60)
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"Rank {rank}: doc_id={doc_id}, score={score:.4f}")
        print(doc_texts[doc_id][:300].replace("\n", " "))
        print("-" * 60)


if __name__ == "__main__":
    main()