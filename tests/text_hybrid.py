from src.retrieval.hybrid import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_prefers_documents_supported_by_multiple_rankings() -> None:
    ranked_lists = [
        [("d1", 10.0), ("d2", 9.0), ("d3", 8.0)],
        [("d2", 0.9), ("d4", 0.8), ("d1", 0.7)],
    ]

    fused = reciprocal_rank_fusion(ranked_lists, rrf_k=60, top_k=4)
    fused_doc_ids = [doc_id for doc_id, _ in fused]

    assert fused_doc_ids[0] == "d2"
    assert "d1" in fused_doc_ids[:3]

def test_rrf_returns_requested_top_k() -> None:
    ranked_lists = [
        [("d1", 1.0), ("d2", 0.9), ("d3", 0.8)],
        [("d2", 1.0), ("d4", 0.9), ("d1", 0.8)],
    ]

    fused = reciprocal_rank_fusion(ranked_lists, rrf_k=60, top_k=2)

    assert len(fused) == 2