import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def compute_corpus_fingerprint(doc_ids: List[str], doc_texts: Dict[str, str]) -> str:
    hasher = hashlib.sha256()
    for doc_id in doc_ids:
        hasher.update(doc_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(doc_texts[doc_id].encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


class DenseRetriever:
    def __init__(
        self,
        doc_texts: Dict[str, str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_root: Path | str = Path("artifacts") / "dense",
        force_rebuild: bool = False,
        device: str = "cpu",
    ) -> None:
        self.doc_texts = doc_texts
        self.doc_ids = sorted(doc_texts.keys())
        self.corpus = [doc_texts[doc_id] for doc_id in self.doc_ids]
        self.model_name = model_name
        self.device = device

        self.model = SentenceTransformer(model_name, device=device)

        self.cache_dir = Path(cache_root) / safe_model_name(model_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.doc_ids_path = self.cache_dir / "doc_ids.json"
        self.embeddings_path = self.cache_dir / "doc_embeddings.npy"
        self.index_path = self.cache_dir / "index.faiss"
        self.meta_path = self.cache_dir / "meta.json"

        self.corpus_fingerprint = compute_corpus_fingerprint(self.doc_ids, self.doc_texts)

        if not force_rebuild and self._cache_is_valid():
            self.index = self._load_index_from_cache()
        else:
            self.index = self._build_index_from_scratch()

    def _cache_is_valid(self) -> bool:
        required_paths = [self.doc_ids_path, self.embeddings_path, self.meta_path]
        if not all(path.exists() for path in required_paths):
            return False

        cached_meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        cached_doc_ids = json.loads(self.doc_ids_path.read_text(encoding="utf-8"))

        return (
            cached_meta.get("model_name") == self.model_name
            and cached_meta.get("device") == self.device
            and cached_meta.get("corpus_size") == len(self.doc_ids)
            and cached_meta.get("corpus_fingerprint") == self.corpus_fingerprint
            and cached_doc_ids == self.doc_ids
        )

    def _load_index_from_cache(self):
        print(f"Loading dense cache from: {self.cache_dir}")
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))

        doc_embeddings = np.load(self.embeddings_path).astype("float32")
        index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        index.add(doc_embeddings)
        faiss.write_index(index, str(self.index_path))
        return index

    def _build_index_from_scratch(self):
        print(f"Building dense cache at: {self.cache_dir}")

        doc_embeddings = self.model.encode(
            self.corpus,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")
        doc_embeddings = l2_normalize(doc_embeddings)

        np.save(self.embeddings_path, doc_embeddings)
        self.doc_ids_path.write_text(json.dumps(self.doc_ids, indent=2), encoding="utf-8")
        self.meta_path.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "device": self.device,
                    "corpus_size": len(self.doc_ids),
                    "embedding_dim": int(doc_embeddings.shape[1]),
                    "corpus_fingerprint": self.corpus_fingerprint,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        index.add(doc_embeddings)
        faiss.write_index(index, str(self.index_path))
        return index

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