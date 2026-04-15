from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader


def main() -> None:
    dataset = "fiqa"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"

    out_dir = Path("datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = util.download_and_unzip(url, str(out_dir))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

    print(f"Downloaded to: {data_path}")
    print(f"Corpus size: {len(corpus)}")
    print(f"Dev queries: {len(queries)}")
    print(f"Dev qrels: {len(qrels)}")


if __name__ == "__main__":
    main()