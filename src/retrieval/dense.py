from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class DenseRetriever:
    def __init__(
        self,
        doc_texts: Dict[str, str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.doc_texts = doc_texts
        self.doc_ids = list(doc_texts.keys())
        self.corpus = [doc_texts[doc_id] for doc_id in self.doc_ids]
        self.model_name = model_name

        self.model = SentenceTransformer(model_name)

        doc_embeddings = self.model.encode(
            self.corpus,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")

        doc_embeddings = l2_normalize(doc_embeddings)

        self.index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        self.index.add(doc_embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query = query.strip()
        if not query:
            return []

        top_k = max(1, min(top_k, len(self.doc_ids)))

        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")

        query_embedding = l2_normalize(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        ranked: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            ranked.append((self.doc_ids[int(idx)], float(score)))

        return ranked