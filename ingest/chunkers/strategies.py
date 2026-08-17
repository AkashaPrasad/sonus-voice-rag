"""The six chunking strategies benchmarked head-to-head in bench/chunking_eval.py.

Every strategy implements the same `chunk(text, meta) -> list[Chunk]` contract so
the evaluation harness can swap them without special-casing. S0 is the control:
without it there is no evidence the others are worth their cost.

Note on Chonkie: the brief specifies it, and S1/S2/S3 mirror its Recursive,
Semantic(SDPM) and Late chunkers. We implement them directly on top of our
Indic-aware segmenter because Chonkie's default splitters are period-based and
therefore produce one giant chunk for Devanagari/Bengali/Urdu text -- the exact
failure this corpus punishes. Chonkie is used for the Token baseline where its
tokenizer behaviour is the thing under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .indic import approx_tokens, detect_script, split_sentences


@dataclass
class Chunk:
    text: str
    parent_id: str
    chunk_index: int
    strategy: str
    n_tokens: int
    lang: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.parent_id}#{self.strategy}#{self.chunk_index}"


def _pack(sentences: list[str], max_tokens: int, overlap_ratio: float = 0.0,
          min_tokens: int = 24) -> list[list[str]]:
    """Greedily pack sentences into token-bounded groups with optional overlap.

    `min_tokens` prevents the degenerate tiny-fragment failure mode that made
    semantic chunking lose to recursive in published benchmarks.
    """
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0

    for s in sentences:
        st = approx_tokens(s)
        if cur and cur_tok + st > max_tokens:
            groups.append(cur)
            if overlap_ratio > 0 and cur:
                # carry back trailing sentences worth ~overlap_ratio of the budget
                carry, ct = [], 0
                for prev in reversed(cur):
                    pt = approx_tokens(prev)
                    if ct + pt > max_tokens * overlap_ratio:
                        break
                    carry.insert(0, prev)
                    ct += pt
                cur, cur_tok = list(carry), ct
            else:
                cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += st

    if cur:
        # Merge a runt tail into the previous group rather than emitting a fragment.
        if groups and cur_tok < min_tokens:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return groups


def _merge_runts(chunks: list[str], min_tokens: int = 24) -> list[str]:
    """Fold sub-threshold chunks into a neighbour.

    Applied after packing because per-paragraph packing can emit a fragment that
    the within-group tail merge never sees.
    """
    if len(chunks) <= 1:
        return chunks
    out: list[str] = []
    for c in chunks:
        if out and approx_tokens(c) < min_tokens:
            out[-1] = f"{out[-1]} {c}".strip()
        else:
            out.append(c)
    # A runt in first position has no predecessor to merge into; fold it forward.
    if len(out) > 1 and approx_tokens(out[0]) < min_tokens:
        out[1] = f"{out[0]} {out[1]}".strip()
        out.pop(0)
    return out


# ── S0: fixed-size token baseline (control group) ────────────────────────────
def s0_fixed(text: str, meta: dict, chunk_size: int = 512, **_) -> list[str]:
    """Character-window baseline with no linguistic awareness whatsoever."""
    script = detect_script(text)
    divisor = 4.0 if script in ("latin", "unknown") else 2.6
    width = int(chunk_size * divisor)
    return [text[i:i + width] for i in range(0, len(text), width) if text[i:i + width].strip()]


# ── S1: recursive structural ─────────────────────────────────────────────────
def s1_recursive(text: str, meta: dict, chunk_size: int = 512,
                 overlap: float = 0.12, **_) -> list[str]:
    """Paragraph -> sentence -> token descent, respecting real boundaries."""
    paras = [p for p in text.split("\n") if p.strip()] or [text]
    out: list[str] = []
    for para in paras:
        sents = split_sentences(para)
        if not sents:
            continue
        for grp in _pack(sents, chunk_size, overlap):
            out.append(" ".join(grp))
    # A short paragraph packs alone, so per-paragraph packing can still emit a
    # runt. Fold anything under the floor into its neighbour: tiny chunks are
    # the documented reason semantic chunking underperforms.
    return _merge_runts(out, min_tokens=24)


# ── S2: semantic + SDPM (skip-window merge) ──────────────────────────────────
def s2_semantic(text: str, meta: dict, chunk_size: int = 512, threshold: float = 0.75,
                skip_window: int = 1, embed_fn: Callable | None = None, **_) -> list[str]:
    """Split at topical shifts; merge similar non-consecutive groups (skip-window).

    Falls back to sentence packing when no embedder is supplied so the strategy
    is still runnable in tests without loading a model.
    """
    sents = split_sentences(text)
    if len(sents) <= 1:
        return [text] if text.strip() else []
    if embed_fn is None:
        return [" ".join(g) for g in _pack(sents, chunk_size)]

    import numpy as np

    vecs = embed_fn(sents)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)

    groups: list[list[int]] = [[0]]
    for i, sim in enumerate(sims, start=1):
        cur_tok = sum(approx_tokens(sents[j]) for j in groups[-1])
        if sim < threshold or cur_tok + approx_tokens(sents[i]) > chunk_size:
            groups.append([i])
        else:
            groups[-1].append(i)

    # SDPM: merge a group with one up to `skip_window` ahead if their centroids
    # are similar -- recovers topically-related content split by an aside.
    merged, used = [], set()
    for gi, g in enumerate(groups):
        if gi in used:
            continue
        cur = list(g)
        for nxt in range(gi + 1, min(gi + 2 + skip_window, len(groups))):
            if nxt in used:
                continue
            a = vecs[cur].mean(axis=0)
            b = vecs[groups[nxt]].mean(axis=0)
            cos = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
            tok = sum(approx_tokens(sents[j]) for j in cur + groups[nxt])
            if cos >= threshold and tok <= chunk_size:
                cur += groups[nxt]
                used.add(nxt)
        merged.append(sorted(cur))
        used.add(gi)

    return [" ".join(sents[j] for j in g) for g in merged if g]


# ── S3: late chunking ────────────────────────────────────────────────────────
def s3_late(text: str, meta: dict, chunk_size: int = 512, **_) -> list[str]:
    """Boundaries identical to S1; the difference is at embedding time.

    Late chunking embeds the FULL passage and mean-pools per boundary, so each
    chunk vector carries whole-passage context. The pooling happens in the
    embedder (see retrieval/embedder.py::embed_late), not here.
    """
    return s1_recursive(text, meta, chunk_size, overlap=0.0)


# ── S4: contextual retrieval ─────────────────────────────────────────────────
def s4_contextual(text: str, meta: dict, chunk_size: int = 512,
                  context_fn: Callable | None = None, **_) -> list[str]:
    """Prepend a one-line situating summary to each chunk before embedding.

    IMPORTANT -- do not situate chunks with the parent query. MS MARCO builds
    each passage set around a query, and the evaluation queries *are* those
    parent queries, so prepending it makes a chunk contain a copy of the query
    it will be scored against. That inflated nDCG@10 from ~0.56 to 0.88 in our
    bake-off: leakage, not retrieval quality.

    Without an LLM we instead situate each chunk with its own leading clause
    plus a positional locator, which is derived only from the passage and is
    therefore available at index time for any corpus. `context_fn` is the hook
    for the real LLM-generated summary when there is budget for it.
    """
    base = s1_recursive(text, meta, chunk_size, overlap=0.0)
    if len(base) <= 1:
        return base

    # First sentence of the passage acts as a cheap topical header for chunks
    # that would otherwise lose their referent.
    lead = (split_sentences(text) or [text])[0]
    lead = " ".join(lead.split()[:14])

    out = []
    for i, c in enumerate(base):
        ctx = context_fn(c, text, meta) if context_fn else f"[{lead} · {i + 1}/{len(base)}]"
        out.append(f"{ctx} {c}".strip())
    return out


# ── S5: parent-child hierarchical ────────────────────────────────────────────
def s5_hierarchical(text: str, meta: dict, chunk_size: int = 512,
                    child_tokens: int = 96, **_) -> list[str]:
    """Small child chunks for precise matching; parents restored at generation.

    Matches MS MARCO's native passage granularity, which is why it pairs well
    with S1 as the retrieval unit.
    """
    sents = split_sentences(text)
    if not sents:
        return []
    packed = [" ".join(g) for g in _pack(sents, child_tokens, 0.0, min_tokens=12)]
    return _merge_runts(packed, min_tokens=12)


# ── S6: metadata + query-type aware (our contribution) ───────────────────────
_QTYPE_BUDGET = {
    "DESCRIPTION": 640,   # explanatory answers need room
    "PERSON": 320, "LOCATION": 320, "ENTITY": 320,
    "NUMERIC": 224,       # tight windows keep the number next to its referent
}


def s6_metadata_aware(text: str, meta: dict, chunk_size: int = 512, **_) -> list[str]:
    """Size chunks by query_type and segment by script.

    NUMERIC/ENTITY queries want tight chunks so the answer span is not diluted;
    DESCRIPTION queries want room for a full explanation. Boundaries always come
    from the script-aware segmenter.
    """
    qt = str((meta or {}).get("query_type", "")).upper()
    budget = _QTYPE_BUDGET.get(qt, chunk_size)
    sents = split_sentences(text)
    if not sents:
        return []
    overlap = 0.10 if qt == "DESCRIPTION" else 0.0
    return [" ".join(g) for g in _pack(sents, budget, overlap)]


STRATEGIES: dict[str, Callable] = {
    "s0_fixed": s0_fixed,
    "s1_recursive": s1_recursive,
    "s2_semantic": s2_semantic,
    "s3_late": s3_late,
    "s4_contextual": s4_contextual,
    "s5_hierarchical": s5_hierarchical,
    "s6_metadata_aware": s6_metadata_aware,
}


def chunk_record(rec, strategy: str, **kwargs) -> list[Chunk]:
    """Apply a named strategy to a PassageRecord."""
    fn = STRATEGIES[strategy]
    meta = {"parent_query": getattr(rec, "parent_query", ""),
            "query_type": getattr(rec, "query_type", "")}
    texts = fn(rec.text, meta, **kwargs)
    return [
        Chunk(text=t, parent_id=rec.passage_id, chunk_index=i, strategy=strategy,
              n_tokens=approx_tokens(t), lang=rec.lang,
              meta={"is_selected": rec.is_selected, "query_id": rec.query_id,
                    "dup_count": rec.dup_count})
        for i, t in enumerate(texts) if t.strip()
    ]
