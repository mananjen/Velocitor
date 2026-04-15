import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search FiQA passages.")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("CLI placeholder")
    print(f"query={args.query}")
    print(f"top_k={args.top_k}")


if __name__ == "__main__":
    main()