from typing import Dict, List, Tuple

import bm25s


class BM25Retriever:
    def __init__(self, doc_texts: Dict[str, str], stopwords: str | None = "en") -> None:
        self.doc_texts = doc_texts
        self.doc_ids = list(doc_texts.keys())
        self.corpus = [doc_texts[doc_id] for doc_id in self.doc_ids]
        self.stopwords = stopwords

        corpus_tokens = bm25s.tokenize(self.corpus, stopwords=self.stopwords)
        self.retriever = bm25s.BM25(corpus=self.doc_ids)
        self.retriever.index(corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query = query.strip()
        if not query:
            return []

        top_k = max(1, min(top_k, len(self.doc_ids)))

        query_tokens = bm25s.tokenize(query, stopwords=self.stopwords)
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)

        ranked = []
        for doc_id, score in zip(results[0], scores[0]):
            ranked.append((str(doc_id), float(score)))

        return ranked