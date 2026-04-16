from src.index.dense import build_fiqa_dense_retriever


def main() -> None:
    retriever, doc_texts = build_fiqa_dense_retriever()

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