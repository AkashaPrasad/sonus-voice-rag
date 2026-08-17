"""Hybrid retrieval: dense int8 + BM25 sparse, fused with RRF.

Everything here runs in-process. A remote vector DB would add 10-40ms of
network on every query, which is 5-20% of the entire budget for a corpus that
fits comfortably in RAM. The index is a flat int8 matrix scored with a single
BLAS call: at demo-corpus scale (~10^5 chunks) exhaustive scoring is faster and
far more predictable than HNSW graph traversal, and it has no recall cliff.

Scoring is two-stage: int8 for the full sweep, then float32 rescoring of the
top candidates so quantization error cannot reorder the final ranking.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk_id: str
    parent_id: str
    text: str
    score: float
    lang: str = ""
    dense_rank: int | None = None
    sparse_rank: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"))


def tokenize(text: str) -> list[str]:
    """Unicode-aware tokenizer that keeps Indic words intact.

    Python's `\\w` excludes the combining marks (Unicode Mn/Mc) that carry
    vowels in Indic scripts, so a `\\w+` split shatters every word at each
    matra: "हिरलूम" becomes ['ह','रल','म'] and BM25 matches consonant debris
    instead of words. We therefore build tokens by character category, treating
    letters, marks, and digits as word-continuing.

    bm25s' default English stemmer/stopwords are also wrong for these scripts,
    so we let BM25's term statistics work over whole words.
    """
    out: list[str] = []
    cur: list[str] = []
    for ch in text.translate(_ZERO_WIDTH):
        cat = unicodedata.category(ch)
        # L* = letters, M* = combining marks (matras), Nd = digits
        if cat[0] in ("L", "M") or cat == "Nd":
            cur.append(ch)
        elif cur:
            out.append("".join(cur).lower())
            cur = []
    if cur:
        out.append("".join(cur).lower())
    return out


class HybridIndex:
    """In-memory hybrid index over a chunk collection."""

    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self.parent_ids: list[str] = []
        self.texts: list[str] = []
        self.langs: list[str] = []
        self.metas: list[dict] = []
        self.codes: np.ndarray | None = None      # int8 [N, D]
        self.scale: float = 1.0
        self.vectors: np.ndarray | None = None    # float32 [N, D] for rescoring
        self._bm25 = None
        self._corpus_tokens: list[list[str]] = []

    # ── build ────────────────────────────────────────────────────────────
    def build(self, chunks: list, embedder, batch_size: int = 512) -> dict:
        from .embedder import quantize_int8

        t0 = time.perf_counter()
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.parent_ids = [c.parent_id for c in chunks]
        self.texts = [c.text for c in chunks]
        self.langs = [c.lang for c in chunks]
        self.metas = [dict(c.meta) for c in chunks]

        vecs = np.zeros((len(chunks), embedder.dim), dtype=np.float32)
        for i in range(0, len(chunks), batch_size):
            vecs[i:i + batch_size] = embedder.encode(self.texts[i:i + batch_size])
        self.vectors = vecs
        self.codes, self.scale = quantize_int8(vecs)
        t_dense = time.perf_counter() - t0

        t1 = time.perf_counter()
        import bm25s

        self._corpus_tokens = [tokenize(t) for t in self.texts]
        self._bm25 = bm25s.BM25()
        self._bm25.index(self._corpus_tokens, show_progress=False)
        t_sparse = time.perf_counter() - t1

        stats = {
            "n_chunks": len(chunks),
            "dim": embedder.dim,
            "dense_build_s": round(t_dense, 2),
            "sparse_build_s": round(t_sparse, 2),
            "int8_bytes": int(self.codes.nbytes),
            "float32_bytes": int(vecs.nbytes),
        }
        log.info("index built: %s", stats)
        return stats

    # ── search ───────────────────────────────────────────────────────────
    def search_dense(self, qvec: np.ndarray, k: int = 50,
                     rescore: int = 200) -> list[tuple[int, float]]:
        """int8 sweep, then float32 rescore of the top `rescore` candidates."""
        if self.codes is None or not len(self.chunk_ids):
            return []
        # int8 matmul in int32 space; monotonic in the true cosine, so it is a
        # valid prefilter even though the magnitudes are unnormalized.
        approx = self.codes.astype(np.int16) @ np.round(
            qvec / self.scale * 127.0).astype(np.int16)
        n = len(approx)
        cand = np.argpartition(-approx, min(rescore, n - 1))[:rescore] if n > rescore \
            else np.arange(n)
        exact = self.vectors[cand] @ qvec           # float32, unit-norm -> cosine
        order = np.argsort(-exact)[:k]
        return [(int(cand[i]), float(exact[i])) for i in order]

    def search_sparse(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        k = min(k, len(self.chunk_ids))
        idx, scores = self._bm25.retrieve([toks], k=k, show_progress=False)
        return [(int(idx[0][j]), float(scores[0][j])) for j in range(idx.shape[1])]

    # ── fusion ───────────────────────────────────────────────────────────
    @staticmethod
    def rrf(dense: list[tuple[int, float]], sparse: list[tuple[int, float]],
            k: int = 60, alpha: float = 0.6) -> dict[int, tuple[float, int | None, int | None]]:
        """Reciprocal Rank Fusion.

        Rank-based rather than score-based, so BM25's unbounded scores and
        cosine's [-1,1] never need calibrating against each other.
        """
        fused: dict[int, list] = {}
        for r, (i, _) in enumerate(dense):
            fused.setdefault(i, [0.0, None, None])
            fused[i][0] += alpha / (k + r + 1)
            fused[i][1] = r
        for r, (i, _) in enumerate(sparse):
            fused.setdefault(i, [0.0, None, None])
            fused[i][0] += (1.0 - alpha) / (k + r + 1)
            fused[i][2] = r
        return {i: (v[0], v[1], v[2]) for i, v in fused.items()}

    def search(self, query: str, qvec: np.ndarray, k: int = 5, top_dense: int = 50,
               top_sparse: int = 50, rrf_k: int = 60, alpha: float = 0.6,
               lang: str | None = None, cross_lingual: bool = True,
               mmr_lambda: float = 0.7) -> tuple[list[Hit], dict[str, float]]:
        """Full hybrid search. Returns hits plus per-stage timings in ms."""
        timings: dict[str, float] = {}

        t = time.perf_counter()
        dense = self.search_dense(qvec, top_dense)
        timings["dense_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        sparse = self.search_sparse(query, top_sparse)
        timings["sparse_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        fused = self.rrf(dense, sparse, rrf_k, alpha)
        exact = dict(dense)

        scored: list[Hit] = []
        for i, (s, dr, sr) in fused.items():
            m = self.metas[i]
            score = s
            # Cheap feature reranking: no cross-encoder on the hot path.
            if self.langs[i] == lang:
                score *= 1.05                                  # same-language bonus
            elif not cross_lingual and lang and self.langs[i] != lang:
                continue                                       # filtered out
            if m.get("dup_count", 1) > 1:
                score *= 1.0 + min(0.03 * (m["dup_count"] - 1), 0.09)
            scored.append(Hit(
                chunk_id=self.chunk_ids[i], parent_id=self.parent_ids[i],
                text=self.texts[i], score=score, lang=self.langs[i],
                dense_rank=dr, sparse_rank=sr,
                meta={**m, "cosine": exact.get(i, 0.0), "index": i},
            ))
        scored.sort(key=lambda h: -h.score)
        timings["fuse_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        hits = self._mmr(scored[:max(k * 4, 12)], k, mmr_lambda)
        timings["rerank_ms"] = (time.perf_counter() - t) * 1000
        return hits, timings

    def _mmr(self, hits: list[Hit], k: int, lam: float) -> list[Hit]:
        """Maximal Marginal Relevance -- drops near-duplicate passages.

        MS MARCO reuses passages heavily, so without this the top-3 context is
        often three phrasings of one passage.
        """
        if not hits or self.vectors is None:
            return hits[:k]
        chosen: list[Hit] = []
        pool = list(hits)
        while pool and len(chosen) < k:
            if not chosen:
                chosen.append(pool.pop(0))
                continue
            cv = np.stack([self.vectors[h.meta["index"]] for h in chosen])
            best_i, best_v = 0, -1e9
            for idx, h in enumerate(pool):
                sim = float((cv @ self.vectors[h.meta["index"]]).max())
                val = lam * h.score - (1 - lam) * sim
                if val > best_v:
                    best_i, best_v = idx, val
            chosen.append(pool.pop(best_i))
        return chosen
