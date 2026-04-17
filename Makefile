.PHONY: setup download-data inspect index search eval bench test docker-build docker-eval docker-bench

PYTHON ?= python
IMAGE_NAME ?= velocitor

TOP_K ?= 10
HYBRID_FETCH_K ?= 100
HYBRID_RRF_K ?= 60
HYBRID_ABLATION_FETCH_K ?= 200
HYBRID_ABLATION_RRF_K ?= 60

setup:
	$(PYTHON) -m pip install -r requirements.txt

download-data:
	$(PYTHON) -m scripts.download_fiqa

inspect:
	$(PYTHON) -m scripts.inspect_fiqa

index:
	$(PYTHON) -m src.index --device cpu

search:
	$(PYTHON) -m src.search --query "what is short selling?" --retriever bm25 --top-k 10

eval:
	$(PYTHON) -m src.eval \
		--top-k $(TOP_K) \
		--hybrid-fetch-k $(HYBRID_FETCH_K) \
		--hybrid-rrf-k $(HYBRID_RRF_K)

bench:
	$(PYTHON) -m src.eval \
		--top-k $(TOP_K) \
		--hybrid-fetch-k $(HYBRID_FETCH_K) \
		--hybrid-rrf-k $(HYBRID_RRF_K) \
		--hybrid-ablation-fetch-k $(HYBRID_ABLATION_FETCH_K) \
		--hybrid-ablation-rrf-k $(HYBRID_ABLATION_RRF_K) \
		--output results/bench.json

test:
	$(PYTHON) -m pytest -q

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-eval:
	docker run --rm \
		-v "$(PWD)/datasets:/app/datasets" \
		-v "$(PWD)/artifacts:/app/artifacts" \
		-v "$(PWD)/results:/app/results" \
		$(IMAGE_NAME) \
		python -m src.eval \
			--top-k $(TOP_K) \
			--hybrid-fetch-k $(HYBRID_FETCH_K) \
			--hybrid-rrf-k $(HYBRID_RRF_K)

docker-bench:
	docker run --rm \
		-v "$(PWD)/datasets:/app/datasets" \
		-v "$(PWD)/artifacts:/app/artifacts" \
		-v "$(PWD)/results:/app/results" \
		$(IMAGE_NAME) \
		python -m src.eval \
			--top-k $(TOP_K) \
			--hybrid-fetch-k $(HYBRID_FETCH_K) \
			--hybrid-rrf-k $(HYBRID_RRF_K) \
			--hybrid-ablation-fetch-k $(HYBRID_ABLATION_FETCH_K) \
			--hybrid-ablation-rrf-k $(HYBRID_ABLATION_RRF_K) \
			--output results/bench.json