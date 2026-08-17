"""Static embeddings -- the single biggest latency decision in the system.

potion-multilingual-128M is a Model2Vec static model: token vectors are looked
up and pooled, with no transformer forward pass. That removes 30-80ms per query
versus any neural encoder, which is what makes a sub-200ms server budget
achievable at all. Distilled from bge-m3, so it retains cross-lingual alignment
across the corpus languages.

int8 quantization is applied at index build time with float32 rescoring of the
top candidates, per the retrieval design.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "minishlab/potion-multilingual-128M"


class Embedder:
    """Thread-safe wrapper with L2-normalized float32 output.

    Normalizing at encode time means cosine similarity is a plain dot product
    everywhere downstream, which keeps the hot path free of divisions.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from model2vec import StaticModel

        self.model_name = model_name
        self._model = StaticModel.from_pretrained(model_name)
        self._lock = threading.Lock()
        self.dim = int(self._model.dim)
        log.info("embedder loaded model=%s dim=%d", model_name, self.dim)

    def encode(self, texts: Iterable[str], normalize: bool = True) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        # StaticModel.encode is not documented thread-safe; the lock is cheap
        # relative to the encode itself and prevents surprises under uvicorn.
        with self._lock:
            vecs = self._model.encode(items)
        vecs = np.asarray(vecs, dtype=np.float32)
        if normalize:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs

    def encode_one(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], normalize)[0]

    def embed_late(self, full_text: str, chunks: list[str]) -> np.ndarray:
        """Late-chunking embedding (strategy S3).

        A true late-chunking implementation pools contextualized token states
        from one full-document forward pass. A static model has no
        context-dependent token states, so we approximate it by blending each
        chunk vector with the parent-document vector. This preserves the
        *intent* -- chunk vectors carry whole-passage context -- and we label it
        as an approximation rather than claiming the published method.
        """
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        doc = self.encode_one(full_text)
        parts = self.encode(chunks)
        blended = 0.75 * parts + 0.25 * doc
        blended /= np.linalg.norm(blended, axis=1, keepdims=True) + 1e-9
        return blended


def quantize_int8(vecs: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quantization of unit-norm vectors.

    Returns the codes plus the scale needed to reconstruct approximate floats.
    Unit-norm input means components are in [-1, 1], so one global scale is
    sufficient and avoids per-vector bookkeeping on the hot path.
    """
    scale = float(np.abs(vecs).max()) or 1.0
    codes = np.clip(np.round(vecs / scale * 127.0), -127, 127).astype(np.int8)
    return codes, scale


def dequantize_int8(codes: np.ndarray, scale: float) -> np.ndarray:
    return codes.astype(np.float32) * (scale / 127.0)
