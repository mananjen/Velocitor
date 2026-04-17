import argparse
from pathlib import Path

from src.index.dense import build_fiqa_dense_retriever


def safe_model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-build retrieval artifacts under src/index."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Dense model to cache for retrieval.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts") / "dense",
        help="Root directory for dense index artifacts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device for embedding and indexing. Use cpu for the take-home.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recompute dense artifacts even if a valid cache already exists.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    retriever, doc_texts = build_fiqa_dense_retriever(
        model_name=args.model_name,
        cache_root=args.cache_root,
        force_rebuild=args.force_rebuild,
        device=args.device,
    )

    cache_dir = Path(args.cache_root) / safe_model_dir_name(args.model_name)

    print("Built retrieval artifacts.")
    print(f"Documents indexed: {len(doc_texts)}")
    print(f"Cache directory: {cache_dir}")
    print(f"Device: {retriever.device}")


if __name__ == "__main__":
    main()