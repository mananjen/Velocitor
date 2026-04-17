import sys
from pathlib import Path

import src.eval as eval_mod


def _fake_config(name: str) -> dict:
    config = {
        "name": name,
        "dataset": "fiqa-dev",
        "query_count": 500,
        "top_k": 10,
        "mrr_rank_depth": 57638,
        "latency_search_top_k": 10,
        "metrics": {
            "recall@10": 0.4,
            "mrr": 0.3,
            "latency_ms": {
                "cold_count": 20,
                "warm_count": 400,
                "cold_p50": 1.0,
                "cold_p95": 2.0,
                "warm_p50": 1.0,
                "warm_p95": 2.0,
            },
            "peak_ram_mb": 512.0,
        },
        "stratified": {
            "recall@10_by_query_length": {
                "short": 0.1,
                "medium": 0.2,
                "long": 0.3,
            },
            "recall@10_by_gold_doc_length": {
                "long_gold": 0.25,
                "rest": 0.45,
            },
        },
        "dataset_stats": {
            "corpus_size": 57638,
            "dev_queries": 500,
            "dev_qrels": 500,
            "top_10_percent_long_doc_count": 5764,
            "long_doc_token_cutoff": 270,
        },
        "machine": {
            "platform": "Windows-10",
            "python_version": "3.11.15",
            "cpu": "13th Gen Intel(R) Core(TM) i7-13700HX (2.10 GHz)",
            "logical_cores": 24,
            "physical_cores": 16,
            "total_ram_gb": 15.75,
        },
        "constraints": {
            "cpu_only": True,
            "warm_p95_le_50ms": True,
            "serve_time_ram_le_2gb": True,
        },
    }
    if name.startswith("hybrid"):
        config["hybrid_params"] = {"fetch_k": 100, "rrf_k": 60}
    return config


def test_eval_main_writes_expected_report_schema(tmp_path, monkeypatch):
    written = {}

    monkeypatch.setattr(eval_mod, "evaluate_bm25", lambda **kwargs: _fake_config("bm25"))
    monkeypatch.setattr(eval_mod, "evaluate_dense", lambda **kwargs: _fake_config("dense-minilm"))
    monkeypatch.setattr(
        eval_mod,
        "evaluate_hybrid",
        lambda **kwargs: _fake_config(kwargs.get("name", "hybrid-rrf")),
    )

    def fake_write_json(data, output_path):
        written["data"] = data
        written["output_path"] = output_path

    monkeypatch.setattr(eval_mod, "write_json", fake_write_json)

    output_path = tmp_path / "bench.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output",
            str(output_path),
            "--hybrid-ablation-fetch-k",
            "200",
            "--hybrid-ablation-rrf-k",
            "60",
        ],
    )

    eval_mod.main()

    assert "data" in written
    assert written["output_path"] == output_path

    report = written["data"]
    assert "configs" in report
    assert len(report["configs"]) == 4

    required_keys = {
        "name",
        "dataset",
        "query_count",
        "top_k",
        "mrr_rank_depth",
        "latency_search_top_k",
        "metrics",
        "stratified",
        "dataset_stats",
        "machine",
        "constraints",
    }

    for config in report["configs"]:
        assert required_keys.issubset(config.keys())

    hybrid_configs = [cfg for cfg in report["configs"] if cfg["name"].startswith("hybrid")]
    assert hybrid_configs
    for config in hybrid_configs:
        assert "hybrid_params" in config