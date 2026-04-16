from src.retrieval.dense import compute_corpus_fingerprint, safe_model_name


def test_safe_model_name_replaces_path_separators() -> None:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    assert safe_model_name(model_name) == "sentence-transformers__all-MiniLM-L6-v2"


def test_corpus_fingerprint_is_stable_for_same_inputs() -> None:
    doc_ids = ["1", "2"]
    doc_texts = {"1": "hello world", "2": "short selling example"}

    fp1 = compute_corpus_fingerprint(doc_ids, doc_texts)
    fp2 = compute_corpus_fingerprint(doc_ids, doc_texts)

    assert fp1 == fp2


def test_corpus_fingerprint_changes_when_text_changes() -> None:
    doc_ids = ["1", "2"]
    doc_texts_a = {"1": "hello world", "2": "short selling example"}
    doc_texts_b = {"1": "hello world", "2": "different text"}

    fp_a = compute_corpus_fingerprint(doc_ids, doc_texts_a)
    fp_b = compute_corpus_fingerprint(doc_ids, doc_texts_b)

    assert fp_a != fp_b