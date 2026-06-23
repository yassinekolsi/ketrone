from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field

import numpy as np

from src.config import settings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Sentence Transformers is used through PyTorch here. Explicitly disabling its
# optional TensorFlow/Flax imports avoids Keras-version conflicts on API hosts.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


@dataclass
class Embedder:
    model_name: str = settings.embedding_model
    fallback_dim: int = 384
    force_fallback: bool = False
    backend: str = field(init=False)

    def __post_init__(self) -> None:
        self._model = None
        self.backend = "deterministic_hash"
        if self.force_fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.backend = "sentence_transformer"
        except Exception:
            self._model = None

    def encode(self, texts: list[str], *, kind: str = "passage") -> list[list[float]]:
        if self._model is not None:
            prefix = "query: " if kind == "query" else "passage: "
            vectors = self._model.encode([prefix + text for text in texts], normalize_embeddings=True, show_progress_bar=False)
            return [vector.astype(float).tolist() for vector in vectors]
        return [self._hash_embedding(text).tolist() for text in texts]

    def _hash_embedding(self, text: str) -> np.ndarray:
        vector = np.zeros(self.fallback_dim, dtype=float)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.fallback_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return normalize(vector)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = math.sqrt(float(np.dot(va, va))) * math.sqrt(float(np.dot(vb, vb)))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
