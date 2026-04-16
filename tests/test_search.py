from src.search import build_parser


def test_build_parser_requires_query() -> None:
    parser = build_parser()

    args = parser.parse_args(["--query", "what is short selling?"])

    assert args.query == "what is short selling?"
    assert args.top_k == 10


def test_build_parser_accepts_custom_top_k() -> None:
    parser = build_parser()

    args = parser.parse_args(["--query", "what is short selling?", "--top-k", "5"])

    assert args.query == "what is short selling?"
    assert args.top_k == 5

def test_build_parser_accepts_retriever_choice() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["--query", "what is short selling?", "--retriever", "dense", "--top-k", "5"]
    )

    assert args.query == "what is short selling?"
    assert args.retriever == "dense"
    assert args.top_k == 5

def test_build_parser_accepts_hybrid_retriever_choice() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["--query", "what is short selling?", "--retriever", "hybrid", "--top-k", "5"]
    )

    assert args.query == "what is short selling?"
    assert args.retriever == "hybrid"
    assert args.top_k == 5