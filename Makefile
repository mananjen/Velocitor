.PHONY: setup download-data inspect search eval test

setup:
	pip install -r requirements.txt

download-data:
	python -m scripts.download_fiqa

inspect:
	python -m scripts.inspect_fiqa

search:
	python -m src.search --query "what is short selling?" --retriever bm25 --top-k 10

eval:
	python -m src.eval

test:
	pytest -q