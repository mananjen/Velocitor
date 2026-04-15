import argparse

from src.index.bm25 import build_fiqa_bm25_retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search FiQA passages.")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    return parser


def snippet(text: str, max_chars: int = 300) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    retriever, doc_texts = build_fiqa_bm25_retriever()
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print("=" * 80)

    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"doc_id: {doc_id}")
        print(f"score: {score:.4f}")
        print(f"passage: {snippet(doc_texts[doc_id])}")
        print("-" * 80)


if __name__ == "__main__":
    main()