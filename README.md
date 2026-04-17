# Velocitor

Constrained retrieval system for the BEIR / FiQA corpus.

Implemented retrieval strategies:
- BM25
- Dense (`sentence-transformers/all-MiniLM-L6-v2`)
- Hybrid (BM25 + Dense via reciprocal rank fusion)

## Repo layout

```text
.
├── README.md
├── DESIGN.md
├── Dockerfile
├── Makefile
├── requirements.txt
├── src/
│   ├── eval.py
│   ├── search.py
│   ├── index/
│   └── retrieval/
├── scripts/
│   ├── download_fiqa.py
│   └── inspect_fiqa.py
├── tests/
└── results/
```

## Chosen operating point

The selected operating point is **dense MiniLM on CPU**. `DESIGN.md` updated for the benchmark table, trade-offs, and failure analysis summary.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Download FiQA:

```bash
python -m scripts.download_fiqa
```

Inspect dataset stats:

```bash
python -m scripts.inspect_fiqa
```

## Search CLI

This repo supports:

```bash
python -m src.search --query "what is short selling?" --retriever bm25 --top-k 10
python -m src.search --query "what is short selling?" --retriever dense --top-k 10
python -m src.search --query "what is short selling?" --retriever hybrid --top-k 10
```


## Indexing

Pre-build retrieval artifacts:

```bash
python -m src.index
```

If dense artifacts already exist and match the corpus/model metadata, they are reused from cache. Building the cache takes 6-7 mins on my system.

## Evaluation

Run the full FiQA dev evaluation:

```bash
python -m src.eval
```

Run with explicit hybrid ablation settings:

```bash
python -m src.eval --hybrid-fetch-k 100 --hybrid-rrf-k 60 --hybrid-ablation-fetch-k 200 --hybrid-ablation-rrf-k 60
```

Benchmark output is written to:

```text
results/bench.json
```

Failure analysis is in:

```text
results/failures.md
```

Design notes are in:

```text
DESIGN.md
```

## Make targets

The repo includes Make targets for convenience:

```bash
make setup
make download-data
make inspect
make index
make eval
make bench
make test
```

## Docker

A `Dockerfile` is included for reproducibility.

Typical flow:

```bash
docker build -t velocitor .
docker run --rm \
  -v ${PWD}/datasets:/app/datasets \
  -v ${PWD}/artifacts:/app/artifacts \
  -v ${PWD}/results:/app/results \
  velocitor \
  python -m src.eval
```

## Important note on verification

The Makefile is included to satisfy the reproducibility requirements of the take-home, but I was  **not fully able to end-to-end validate it locally before submission**. The Python entrypoints and benchmark outputs in the repo are the authoritative paths that were used for development. The Dockerfile runs the eval properly.

## Tests

Run the test suite with:

```bash
pytest -q
```

## Evaluation note

There are two different depths used in evaluation by design:

- **Latency** is measured at the configured serve-time `top_k`
- **Quality metrics** are computed from a full ranking over the corpus

This is intentional so that:
- latency reflects the actual serving configuration
- MRR is measured against the full ranked list rather than being artificially truncated

The output JSON records both:
- `latency_search_top_k`
- `mrr_rank_depth`

