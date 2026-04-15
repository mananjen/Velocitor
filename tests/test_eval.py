from src.eval import mean_score, qrels_to_relevant_doc_ids, recall_at_k, reciprocal_rank


def test_recall_at_k_returns_hit_when_relevant_doc_is_in_top_k() -> None:
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = ["d3", "d9"]

    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    assert recall_at_k(retrieved, relevant, k=2) == 0.0


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