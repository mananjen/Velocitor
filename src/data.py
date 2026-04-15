from pathlib import Path
from typing import Dict, Tuple

from beir.datasets.data_loader import GenericDataLoader


DEFAULT_DATA_DIR = Path("datasets") / "fiqa"


def load_fiqa_dev(data_dir: Path | str = DEFAULT_DATA_DIR) -> Tuple[Dict, Dict, Dict]:
    data_dir = Path(data_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_dir)).load(split="dev")
    return corpus, queries, qrels


def build_passage_text(doc: Dict) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def get_doc_texts(corpus: Dict) -> Dict[str, str]:
    return {doc_id: build_passage_text(doc) for doc_id, doc in corpus.items()}