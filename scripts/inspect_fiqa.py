from collections import Counter
from statistics import mean, median

from src.data import build_passage_text, load_fiqa_dev
from src.utils import query_length_bucket, token_count, top_10_percent_count


def percentile(values: list[int], p: float) -> int:
    if not values:
        raise ValueError("Cannot compute percentile of empty list.")
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    values = sorted(values)
    idx = int(round((p / 100) * (len(values) - 1)))
    return values[idx]


def print_length_stats(name: str, lengths: list[int]) -> None:
    print(f"{name} length stats (tokens)")
    print(f"  min:    {min(lengths)}")
    print(f"  p25:    {percentile(lengths, 25)}")
    print(f"  median: {int(median(lengths))}")
    print(f"  p75:    {percentile(lengths, 75)}")
    print(f"  p90:    {percentile(lengths, 90)}")
    print(f"  p95:    {percentile(lengths, 95)}")
    print(f"  max:    {max(lengths)}")
    print(f"  mean:   {mean(lengths):.2f}")
    print()


def snippet(text: str, max_chars: int = 300) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def print_query_examples(
    corpus: dict,
    queries: dict,
    qrels: dict,
    max_examples: int = 3,
) -> None:
    print("Sample query -> gold passage examples")
    print("-" * 60)

    shown = 0
    for query_id, query_text in queries.items():
        if query_id not in qrels or not qrels[query_id]:
            continue

        gold_doc_ids = sorted(qrels[query_id].keys())
        gold_doc_id = gold_doc_ids[0]
        gold_doc = corpus[gold_doc_id]
        gold_text = build_passage_text(gold_doc)

        print(f"Query ID: {query_id}")
        print(f"Query: {query_text}")
        print(f"Gold doc IDs: {gold_doc_ids}")
        print(f"Displayed gold doc ID: {gold_doc_id}")
        print(f"Gold relevance labels: {qrels[query_id]}")
        print(f"Gold passage snippet: {snippet(gold_text)}")
        print("-" * 60)

        shown += 1
        if shown >= max_examples:
            break

    print()


def print_extreme_docs(corpus: dict, doc_lengths: dict[str, int]) -> None:
    shortest_doc_id = min(doc_lengths, key=doc_lengths.get)
    longest_doc_id = max(doc_lengths, key=doc_lengths.get)

    shortest_doc = build_passage_text(corpus[shortest_doc_id])
    longest_doc = build_passage_text(corpus[longest_doc_id])

    print("Shortest document")
    print(f"  doc_id: {shortest_doc_id}")
    print(f"  tokens: {doc_lengths[shortest_doc_id]}")
    print(f"  snippet: {snippet(shortest_doc)}")
    print()

    print("Longest document")
    print(f"  doc_id: {longest_doc_id}")
    print(f"  tokens: {doc_lengths[longest_doc_id]}")
    print(f"  snippet: {snippet(longest_doc)}")
    print()


def main() -> None:
    corpus, queries, qrels = load_fiqa_dev()

    doc_texts = {doc_id: build_passage_text(doc) for doc_id, doc in corpus.items()}
    doc_lengths = {doc_id: token_count(text) for doc_id, text in doc_texts.items()}
    query_lengths = {query_id: token_count(text) for query_id, text in queries.items()}

    titled_docs = sum(1 for doc in corpus.values() if (doc.get("title") or "").strip())
    untitled_docs = len(corpus) - titled_docs

    query_bucket_counts = Counter(query_length_bucket(q) for q in queries.values())
    relevant_per_query = [len(rels) for rels in qrels.values()]

    sorted_doc_lengths = sorted(doc_lengths.values(), reverse=True)
    top_n = top_10_percent_count(len(doc_lengths))
    longest_doc_cutoff = sorted_doc_lengths[top_n - 1]

    print("FiQA inspection")
    print("=" * 60)
    print(f"Corpus size: {len(corpus)}")
    print(f"Dev queries: {len(queries)}")
    print(f"Dev qrels: {len(qrels)}")
    print()

    print("Top 10% longest-doc stats")
    print(f"  count:  {top_n}")
    print(f"  cutoff: {longest_doc_cutoff} tokens")
    print()

    print("Title coverage")
    print(f"  with title:    {titled_docs}")
    print(f"  without title: {untitled_docs}")
    print(f"  title rate:    {titled_docs / len(corpus):.2%}")
    print()

    print_length_stats("Document", list(doc_lengths.values()))
    print_length_stats("Query", list(query_lengths.values()))

    print("Query length buckets")
    for bucket in ("short", "medium", "long"):
        print(f"  {bucket}: {query_bucket_counts[bucket]}")
    print()

    print("Relevant passages per query")
    print(f"  min:    {min(relevant_per_query)}")
    print(f"  median: {int(median(relevant_per_query))}")
    print(f"  max:    {max(relevant_per_query)}")
    print(f"  mean:   {mean(relevant_per_query):.2f}")
    print()

    print_extreme_docs(corpus, doc_lengths)
    print_query_examples(corpus, queries, qrels, max_examples=5)


if __name__ == "__main__":
    main()