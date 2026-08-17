"""Extractive answering -- the fast path, with zero external calls.

MS MARCO is natively a span-answer dataset, so selecting the best supporting
span from the top passage is the task's own answer form, not a degraded
shortcut. This is what keeps T_pipeline inside budget: a cross-continent LLM
call cannot fit in 200ms from Asia, and pretending otherwise would be dishonest.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ingest.chunkers.indic import split_sentences  # noqa: E402

ABSTAIN_TEXT = {
    "hi": "इस संग्रह में मुझे इसका आधारभूत उत्तर नहीं मिला।",
    "bn": "এই সংগ্রহে এর ভিত্তিযুক্ত উত্তর পাইনি।",
    "ta": "இந்தத் தொகுப்பில் இதற்கான ஆதாரப் பதில் இல்லை.",
    "te": "ఈ సేకరణలో దీనికి ఆధారిత సమాధానం లేదు.",
    "en": "I don't have grounded information about that in this corpus.",
}


@dataclass
class Citation:
    passage_id: str
    char_start: int
    char_end: int
    score: float
    text: str = ""
    lang: str = ""


@dataclass
class Answer:
    text: str
    mode: str                       # "extractive" | "generative" | "abstain"
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def _lexical_overlap(q_tokens: set[str], sent: str, tokenize) -> float:
    st = set(tokenize(sent))
    if not st or not q_tokens:
        return 0.0
    return len(q_tokens & st) / len(q_tokens)


def select_span(query: str, hits: list, embedder, tokenize,
                max_sentences: int = 3) -> tuple[str, Citation | None, float]:
    """Pick the best contiguous 1-3 sentence span from the top passage.

    Scores each candidate window by embedding similarity to the query plus
    lexical overlap. Both signals are cheap; the embedding catches paraphrase
    and the lexical term catches names and numbers the static model may blur.
    """
    if not hits:
        return "", None, 0.0

    top = hits[0]
    sents = split_sentences(top.text) or [top.text]
    qv = embedder.encode_one(query)
    q_tokens = set(tokenize(query))

    # Build candidate windows of 1..max_sentences consecutive sentences.
    windows: list[tuple[int, int, str]] = []
    for i in range(len(sents)):
        for n in range(1, max_sentences + 1):
            if i + n <= len(sents):
                windows.append((i, i + n, " ".join(sents[i:i + n])))
    if not windows:
        return top.text, None, float(getattr(top, "score", 0.0))

    texts = [w[2] for w in windows]
    vecs = embedder.encode(texts)
    sims = vecs @ qv
    lex = np.array([_lexical_overlap(q_tokens, t, tokenize) for t in texts])
    # Slight preference for longer spans: a single clause often under-answers.
    length_bonus = np.array([min(0.06 * (w[1] - w[0] - 1), 0.12) for w in windows])
    combined = 0.62 * sims + 0.32 * lex + length_bonus

    best = int(np.argmax(combined))
    i0, i1, span = windows[best]
    start = top.text.find(sents[i0])
    end = start + len(span) if start >= 0 else len(top.text)

    cit = Citation(
        passage_id=top.parent_id,
        char_start=max(start, 0),
        char_end=end,
        score=float(combined[best]),
        text=span,
        lang=getattr(top, "lang", ""),
    )
    return span, cit, float(combined[best])


def answer_extractive(query: str, hits: list, embedder, tokenize,
                      abstain_threshold: float = 0.44,
                      min_score_gap: float = 0.0,
                      lang: str = "en") -> Answer:
    """Build a grounded extractive answer, or abstain.

    Abstention is driven by retrieval confidence: when the corpus itself has
    nothing close to the query, that is the cheapest and most reliable
    off-topic signal available.
    """
    if not hits:
        return Answer(text=ABSTAIN_TEXT.get(lang, ABSTAIN_TEXT["en"]),
                      mode="abstain", abstained=True, confidence=0.0)

    span, cit, score = select_span(query, hits, embedder, tokenize)
    top_cos = float(hits[0].meta.get("cosine", 0.0))

    # Flat score distributions mean "I don't know" even when the top score is ok.
    gap_ok = True
    if min_score_gap > 0 and len(hits) >= 2:
        gap_ok = (hits[0].score - hits[-1].score) >= min_score_gap

    if score < abstain_threshold or not gap_ok:
        return Answer(
            text=ABSTAIN_TEXT.get(lang, ABSTAIN_TEXT["en"]),
            mode="abstain", abstained=True, confidence=score,
            citations=[cit] if cit else [],
            meta={"reason": "low_confidence" if score < abstain_threshold else "flat_scores",
                  "span_score": score, "top_cosine": top_cos},
        )

    return Answer(
        text=span, mode="extractive", citations=[cit] if cit else [],
        confidence=score, meta={"span_score": score, "top_cosine": top_cos},
    )
