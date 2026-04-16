import argparse
import json
import platform
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import psutil

from src.data import load_fiqa_dev
from src.index.bm25 import build_fiqa_bm25_retriever
from src.index.dense import build_fiqa_dense_retriever
from src.utils import query_length_bucket, token_count, top_10_percent_count


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0

    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_set)


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


def has_long_gold_doc(
    relevant_doc_ids: Iterable[str],
    doc_lengths: Dict[str, int],
    longest_doc_cutoff: int,
) -> bool:
    return any(doc_lengths.get(doc_id, 0) >= longest_doc_cutoff for doc_id in relevant_doc_ids)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]

    position = (p / 100.0) * (len(xs) - 1)
    lower = int(position)
    upper = min(lower + 1, len(xs) - 1)
    weight = position - lower
    return xs[lower] * (1.0 - weight) + xs[upper] * weight


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def get_machine_info() -> dict:
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "total_ram_gb": vm.total / (1024 ** 3),
    }


def measure_single_query_latency_ms(retriever, query_text: str, top_k: int) -> float:
    start_ns = time.perf_counter_ns()
    retriever.search(query_text, top_k=top_k)
    end_ns = time.perf_counter_ns()
    return (end_ns - start_ns) / 1_000_000.0


def measure_latency_and_peak_ram(
    retriever,
    query_texts: Sequence[str],
    top_k: int,
    warmup_count: int = 100,
    cold_count: int = 20,
) -> dict:
    process = psutil.Process()
    peak_rss_bytes = process.memory_info().rss

    cold_queries = list(query_texts[: min(cold_count, len(query_texts))])
    warmup_queries = list(query_texts[: min(warmup_count, len(query_texts))])
    warm_queries = list(query_texts[len(warmup_queries):])

    cold_latencies_ms: list[float] = []
    for query_text in cold_queries:
        latency_ms = measure_single_query_latency_ms(retriever, query_text, top_k=top_k)
        cold_latencies_ms.append(latency_ms)
        peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)

    for query_text in warmup_queries:
        retriever.search(query_text, top_k=top_k)
        peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)

    warm_latencies_ms: list[float] = []
    for query_text in warm_queries:
        latency_ms = measure_single_query_latency_ms(retriever, query_text, top_k=top_k)
        warm_latencies_ms.append(latency_ms)
        peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)

    return {
        "latency_ms": {
            "cold_count": len(cold_latencies_ms),
            "warm_count": len(warm_latencies_ms),
            "cold_p50": percentile(cold_latencies_ms, 50),
            "cold_p95": percentile(cold_latencies_ms, 95),
            "warm_p50": percentile(warm_latencies_ms, 50),
            "warm_p95": percentile(warm_latencies_ms, 95),
        },
        "peak_ram_mb": bytes_to_mb(peak_rss_bytes),
    }


def evaluate_with_builder(
    name: str,
    builder: Callable[[], tuple[object, dict[str, str]]],
    top_k: int = 10,
    max_queries: int | None = None,
) -> dict:
    corpus, queries, qrels = load_fiqa_dev()
    retriever, doc_texts = builder()

    doc_lengths = {doc_id: token_count(text) for doc_id, text in doc_texts.items()}
    sorted_doc_lengths = sorted(doc_lengths.values(), reverse=True)
    top_n = top_10_percent_count(len(doc_lengths))
    longest_doc_cutoff = sorted_doc_lengths[top_n - 1]

    query_items = list(queries.items())
    if max_queries is not None:
        query_items = query_items[:max_queries]

    query_texts = [query_text for _, query_text in query_items]
    serving_stats = measure_latency_and_peak_ram(retriever, query_texts, top_k=top_k)

    recall_scores: list[float] = []
    reciprocal_ranks: list[float] = []

    recall_by_query_length = {"short": [], "medium": [], "long": []}
    recall_by_gold_doc_length = {"long_gold": [], "rest": []}

    full_rank_depth = len(doc_texts)

    for query_id, query_text in query_items:
        relevant_doc_ids = qrels_to_relevant_doc_ids(qrels.get(query_id, {}))

        results = retriever.search(query_text, top_k=full_rank_depth)
        ranked_doc_ids = [doc_id for doc_id, _ in results]

        query_recall = recall_at_k(ranked_doc_ids, relevant_doc_ids, k=top_k)
        query_rr = reciprocal_rank(ranked_doc_ids, relevant_doc_ids)

        recall_scores.append(query_recall)
        reciprocal_ranks.append(query_rr)

        length_bucket = query_length_bucket(query_text)
        recall_by_query_length[length_bucket].append(query_recall)

        gold_bucket = (
            "long_gold"
            if has_long_gold_doc(relevant_doc_ids, doc_lengths, longest_doc_cutoff)
            else "rest"
        )
        recall_by_gold_doc_length[gold_bucket].append(query_recall)

    warm_p95 = serving_stats["latency_ms"]["warm_p95"]
    peak_ram_mb = serving_stats["peak_ram_mb"]

    return {
        "name": name,
        "dataset": "fiqa-dev",
        "query_count": len(query_items),
        "top_k": top_k,
        "mrr_rank_depth": full_rank_depth,
        "latency_search_top_k": top_k,
        "metrics": {
            "recall@10": mean_score(recall_scores),
            "mrr": mean_score(reciprocal_ranks),
            "latency_ms": serving_stats["latency_ms"],
            "peak_ram_mb": peak_ram_mb,
        },
        "stratified": {
            "recall@10_by_query_length": {
                "short": mean_score(recall_by_query_length["short"]),
                "medium": mean_score(recall_by_query_length["medium"]),
                "long": mean_score(recall_by_query_length["long"]),
            },
            "recall@10_by_gold_doc_length": {
                "long_gold": mean_score(recall_by_gold_doc_length["long_gold"]),
                "rest": mean_score(recall_by_gold_doc_length["rest"]),
            },
        },
        "dataset_stats": {
            "corpus_size": len(corpus),
            "dev_queries": len(queries),
            "dev_qrels": len(qrels),
            "top_10_percent_long_doc_count": top_n,
            "long_doc_token_cutoff": longest_doc_cutoff,
        },
        "machine": get_machine_info(),
        "constraints": {
            "cpu_only": True,
            "warm_p95_le_50ms": warm_p95 <= 50.0,
            "serve_time_ram_le_2gb": peak_ram_mb <= 2048.0,
        },
    }


def evaluate_bm25(top_k: int = 10, max_queries: int | None = None) -> dict:
    return evaluate_with_builder(
        name="bm25",
        builder=build_fiqa_bm25_retriever,
        top_k=top_k,
        max_queries=max_queries,
    )


def evaluate_dense(top_k: int = 10, max_queries: int | None = None) -> dict:
    return evaluate_with_builder(
        name="dense-minilm",
        builder=build_fiqa_dense_retriever,
        top_k=top_k,
        max_queries=max_queries,
    )


def write_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrievers on FiQA dev.")
    parser.add_argument("--top-k", type=int, default=10, help="Recall cutoff and serve-time search depth")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional query cap for faster debugging",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "bench.json",
        help="Path to write evaluation JSON",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    report = {
        "configs": [
            evaluate_bm25(top_k=args.top_k, max_queries=args.max_queries),
            evaluate_dense(top_k=args.top_k, max_queries=args.max_queries),
        ]
    }

    write_json(report, args.output)

    print(f"Wrote evaluation to: {args.output}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()