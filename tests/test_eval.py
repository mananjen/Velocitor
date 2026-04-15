from src.eval import (
    has_long_gold_doc,
    mean_score,
    percentile,
    qrels_to_relevant_doc_ids,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_returns_fraction_of_relevant_docs_found_in_top_k() -> None:
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = ["d3", "d4", "d9"]

    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=3) == 1 / 3
    assert recall_at_k(retrieved, relevant, k=4) == 2 / 3


def test_reciprocal_rank_returns_inverse_of_first_relevant_rank() -> None:
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = ["d3", "d4"]

    assert reciprocal_rank(retrieved, relevant) == 1 / 3


def test_reciprocal_rank_returns_zero_when_no_relevant_doc_is_found() -> None:
    retrieved = ["d1", "d2", "d3"]
    relevant = ["d8"]

    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_mean_score_returns_average() -> None:
    assert mean_score([1.0, 0.0, 1.0]) == 2 / 3
    assert mean_score([]) == 0.0


def test_qrels_to_relevant_doc_ids_filters_non_positive_labels() -> None:
    qrels = {"d1": 1, "d2": 0, "d3": 2, "d4": -1}

    assert qrels_to_relevant_doc_ids(qrels) == ["d1", "d3"]


def test_has_long_gold_doc_checks_relevant_docs_against_cutoff() -> None:
    doc_lengths = {"d1": 100, "d2": 275, "d3": 80}

    assert has_long_gold_doc(["d1", "d2"], doc_lengths, longest_doc_cutoff=270) is True
    assert has_long_gold_doc(["d1", "d3"], doc_lengths, longest_doc_cutoff=270) is False


def test_percentile_returns_interpolated_values() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == 48.0