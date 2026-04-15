from typing import Dict, List, Tuple


class BM25Retriever:
    def __init__(self, doc_texts: Dict[str, str]) -> None:
        self.doc_texts = doc_texts

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        raise NotImplementedError("BM25 search is not implemented yet.")