# Velocitor

## Implemented retrievers

- BM25
- Dense (`sentence-transformers/all-MiniLM-L6-v2`)
- Hybrid (BM25 + Dense via RRF)

## Setup

```bash
pip install -r requirements.txt
```

## Download dataset

```bash
python -m scripts.download_fiqa
```

## Inspect dataset

```bash
python -m scripts.inspect_fiqa
```

## Search

```bash
python -m src.search --query "what is short selling?" --retriever bm25 --top-k 10
python -m src.search --query "what is short selling?" --retriever dense --top-k 10
python -m src.search --query "what is short selling?" --retriever hybrid --top-k 10
```

## Evaluate

```bash
python -m src.eval
```

## Tests

```bash
pytest -q
```

## Benchmark output
Raw benchmark results are written to:
```bash
results/bench.json
```