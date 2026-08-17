"""The stage graph: typed, timed, budget-aware, and degradable.

Every stage records its own duration into the RunContext, so the latency HUD in
the UI and the /metrics endpoint report the same numbers the benchmark does --
there is one source of timing truth, not a separate instrumented path.

The error ladder is: retry -> alternate provider -> extractive -> abstain.
A user never receives a 500.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RunContext:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    deadline_ms: float = 200.0
    started: float = field(default_factory=time.perf_counter)
    timings: dict[str, float] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    def remaining_ms(self) -> float:
        return self.deadline_ms - self.elapsed_ms()

    def record(self, stage: str, ms: float) -> None:
        self.timings[stage] = round(ms, 3)

    def event(self, kind: str, **data: Any) -> None:
        self.events.append({"kind": kind, "at_ms": round(self.elapsed_ms(), 3), **data})


class Timer:
    """Context manager that writes a stage duration into the RunContext."""

    def __init__(self, ctx: RunContext, stage: str) -> None:
        self.ctx, self.stage = ctx, stage

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ctx.record(self.stage, (time.perf_counter() - self.t0) * 1000)
        return False


class SemanticCache:
    """Two-tier cache: exact normalized-hash, then embedding similarity.

    Voice traffic repeats heavily (demo queries, rephrasings), so this is the
    cheapest large win available on P50.
    """

    def __init__(self, size: int = 2048, threshold: float = 0.97) -> None:
        self.size = size
        self.threshold = threshold
        self._exact: OrderedDict[str, Any] = OrderedDict()
        self._vecs: list[np.ndarray] = []
        self._vals: list[Any] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _norm(q: str) -> str:
        return " ".join(q.lower().split())

    def _key(self, q: str) -> str:
        return hashlib.blake2b(self._norm(q).encode(), digest_size=16).hexdigest()

    def get(self, query: str, qvec: np.ndarray | None = None):
        k = self._key(query)
        if k in self._exact:
            self._exact.move_to_end(k)
            self.hits += 1
            return self._exact[k], "exact"
        if qvec is not None and self._vecs:
            sims = np.stack(self._vecs) @ qvec
            i = int(np.argmax(sims))
            if float(sims[i]) >= self.threshold:
                self.hits += 1
                return self._vals[i], "semantic"
        self.misses += 1
        return None, None

    def put(self, query: str, qvec: np.ndarray | None, value: Any) -> None:
        k = self._key(query)
        self._exact[k] = value
        self._exact.move_to_end(k)
        while len(self._exact) > self.size:
            self._exact.popitem(last=False)
        if qvec is not None:
            self._vecs.append(qvec)
            self._vals.append(value)
            if len(self._vecs) > self.size:
                self._vecs.pop(0)
                self._vals.pop(0)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "entries": len(self._exact)}


class Pipeline:
    """Fast path: cache -> guard_in -> embed -> retrieve -> guard_retrieval ->
    extract -> guard_out. Zero external network calls."""

    def __init__(self, index, embedder, cache: SemanticCache | None = None,
                 abstain_threshold: float = 0.44,
                 groundedness_threshold: float = 0.55) -> None:
        self.index = index
        self.embedder = embedder
        self.cache = cache or SemanticCache()
        self.abstain_threshold = abstain_threshold
        self.groundedness_threshold = groundedness_threshold

    def run(self, query: str, lang: str | None = None, k: int = 3,
            cross_lingual: bool = True, use_cache: bool = True,
            deadline_ms: float = 200.0) -> dict:
        from answer.extractive import ABSTAIN_TEXT, answer_extractive
        from guardrails.rails import (CROSS_LINGUAL_ABSTAIN_THRESHOLD, Decision,
                                      detect_lang, input_rails, output_rails,
                                      retrieval_rails, safety_rails)
        from retrieval.engine import tokenize

        ctx = RunContext(deadline_ms=deadline_ms)

        # ── Layer 1 + 2 guardrails (run before any work is spent) ──
        with Timer(ctx, "guard_in"):
            r1 = input_rails(query)
            if r1.allowed:
                r2 = safety_rails(query)
                if not r2.allowed:
                    r1 = r2
        if not r1.allowed:
            ctx.event("blocked", category=r1.category, layer=r1.layer)
            return self._blocked(ctx, r1, lang or detect_lang(query))

        clean = r1.redacted_text or query
        qlang = lang or r1.details.get("lang", "en")

        # ── cache probe ──
        with Timer(ctx, "cache_probe"):
            cached, how = (self.cache.get(clean) if use_cache else (None, None))
        if cached is not None:
            ctx.event("cache_hit", how=how)
            out = dict(cached)
            out["cached"] = how
            out["trace_id"] = ctx.trace_id
            out["timings"] = ctx.timings
            out["total_ms"] = round(ctx.elapsed_ms(), 3)
            return out

        # ── embed ──
        with Timer(ctx, "embed"):
            qvec = self.embedder.encode_one(clean)

        # semantic tier needs the vector, so probe again post-embed
        if use_cache:
            sem, how = self.cache.get(clean, qvec)
            if sem is not None:
                out = dict(sem)
                out["cached"] = how
                out["trace_id"] = ctx.trace_id
                out["timings"] = ctx.timings
                out["total_ms"] = round(ctx.elapsed_ms(), 3)
                return out

        # ── retrieve ──
        with Timer(ctx, "retrieve"):
            hits, sub = self.index.search(clean, qvec, k=k, lang=qlang,
                                          cross_lingual=cross_lingual)
        for kk, vv in sub.items():
            ctx.record(kk.replace("_ms", ""), vv)

        # ── Layer 3: retrieval confidence ──
        with Timer(ctx, "guard_retrieval"):
            r3 = retrieval_rails(hits, self.abstain_threshold, query_lang=qlang)
        if r3.decision == Decision.ABSTAIN:
            ctx.event("abstain", category=r3.category)
            return self._abstain(ctx, qlang, hits, r3)

        # ── extractive answer ──
        # The span score is on the same scale problem as the retrieval score, so
        # a cross-lingual hit gets the cross-lingual floor here too.
        span_threshold = self.abstain_threshold
        if hits and (getattr(hits[0], "lang", "") or "") != qlang:
            span_threshold = min(self.abstain_threshold,
                                 CROSS_LINGUAL_ABSTAIN_THRESHOLD)
        with Timer(ctx, "extract"):
            ans = answer_extractive(clean, hits, self.embedder, tokenize,
                                    span_threshold, lang=qlang)

        # ── Layer 4: groundedness ──
        with Timer(ctx, "guard_out"):
            if not ans.abstained:
                r4 = output_rails(ans.text, [h.text for h in hits], tokenize,
                                  self.groundedness_threshold)
                if r4.decision != Decision.ALLOW:
                    ctx.event("output_blocked", category=r4.category)
                    return self._abstain(ctx, qlang, hits, r4)

        result = {
            "answer": ans.text,
            "mode": ans.mode,
            "abstained": ans.abstained,
            "confidence": round(ans.confidence, 4),
            "lang": qlang,
            "citations": [{"passage_id": c.passage_id, "text": c.text,
                           "char_start": c.char_start, "char_end": c.char_end,
                           "score": round(c.score, 4), "lang": c.lang}
                          for c in ans.citations],
            "passages": [{"passage_id": h.parent_id, "text": h.text, "lang": h.lang,
                          "score": round(h.score, 5),
                          "cosine": round(float(h.meta.get("cosine", 0)), 4),
                          "cross_lingual": h.lang != qlang}
                         for h in hits],
            "trace_id": ctx.trace_id,
            "timings": ctx.timings,
            "total_ms": round(ctx.elapsed_ms(), 3),
            "cached": None,
            "events": ctx.events,
        }
        if use_cache and not ans.abstained:
            self.cache.put(clean, qvec, result)
        return result

    # ── terminal states ──
    def _blocked(self, ctx: RunContext, rail, lang: str) -> dict:
        from answer.extractive import ABSTAIN_TEXT
        return {
            "answer": ABSTAIN_TEXT.get(lang, ABSTAIN_TEXT["en"]),
            "mode": "blocked", "abstained": True, "blocked": True,
            "block_category": rail.category, "block_layer": rail.layer,
            "block_reason": rail.reason, "confidence": 0.0, "lang": lang,
            "citations": [], "passages": [], "trace_id": ctx.trace_id,
            "timings": ctx.timings, "total_ms": round(ctx.elapsed_ms(), 3),
            "events": ctx.events,
        }

    def _abstain(self, ctx: RunContext, lang: str, hits: list, rail) -> dict:
        from answer.extractive import ABSTAIN_TEXT
        return {
            "answer": ABSTAIN_TEXT.get(lang, ABSTAIN_TEXT["en"]),
            "mode": "abstain", "abstained": True, "blocked": False,
            "abstain_reason": rail.category, "abstain_layer": rail.layer,
            "confidence": float(hits[0].meta.get("cosine", 0.0)) if hits else 0.0,
            "lang": lang, "citations": [],
            # Surface the nearest thing we did find -- the abstention UI shows it.
            "passages": [{"passage_id": h.parent_id, "text": h.text, "lang": h.lang,
                          "score": round(h.score, 5),
                          "cosine": round(float(h.meta.get("cosine", 0)), 4),
                          "cross_lingual": h.lang != lang}
                         for h in hits[:2]],
            "trace_id": ctx.trace_id, "timings": ctx.timings,
            "total_ms": round(ctx.elapsed_ms(), 3), "events": ctx.events,
        }
