"""Four guardrail layers, ordered cheap -> expensive.

Layer 2 deliberately uses in-process pattern and lexicon checks rather than a
hosted safety model: Llama Guard benchmarks around 459ms P95, which is more than
twice our entire pipeline budget. The heavyweight model belongs in offline audit,
not on the hot path.

The highest-value rail here is Layer 3: retrieval confidence. The corpus itself
tells you when a question isn't about the corpus, and it costs nothing because
the scores already exist.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ABSTAIN = "abstain"


@dataclass
class RailResult:
    decision: Decision = Decision.ALLOW
    category: str = ""
    reason: str = ""
    layer: str = ""
    redacted_text: str | None = None
    details: dict = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == Decision.ALLOW


# ── Layer 1: input rails ─────────────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|the)\s+(instructions?|rules?|prompts?)",
    r"forget\s+(everything|all|your\s+instructions?)",
    r"^\s*system\s*[:>]",
    r"</?(system|assistant|user)>",
    r"\[\s*(system|inst|/inst)\s*\]",
    r"you\s+are\s+now\s+(a|an|in)\b",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"print\s+(your\s+)?(instructions?|system\s+prompt)",
    r"developer\s+mode",
    r"\bDAN\b\s+mode",
    r"pretend\s+(to\s+be|you\s+are)",
    r"act\s+as\s+(if\s+)?(you\s+are\s+)?(an?\s+)?(unrestricted|jailbroken|evil)",
    r"(bypass|override|disable)\s+(your\s+)?(safety|guardrails?|filters?|restrictions?)",
    # Indic-script variants: injection attempts are not English-only.
    r"पिछले\s+निर्देश",
    r"निर्देशों?\s+को\s+(अनदेखा|नज़रअंदाज)",
    r"முந்தைய\s+வழிமுறை",
    r"మునుపటి\s+సూచన",
    r"পূর্ববর্তী\s+নির্দেশ",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# Long base64-ish blobs are a common smuggling vector.
_B64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")

# Indian PII formats. Redact before logging, never before retrieval.
PII_PATTERNS = {
    "aadhaar": re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "phone_in": re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"),
    "upi": re.compile(r"\b[\w.\-]{3,}@(?:okhdfcbank|oksbi|okicici|okaxis|paytm|ybl|upi)\b"),
    "email": re.compile(r"\b[\w.\-]+@[\w\-]+\.[A-Za-z]{2,}\b"),
}

UNSAFE_LEXICON = {
    "weapons": [r"\b(build|make|construct|synthes\w+)\s+(a\s+)?(bomb|explosive|ied|nerve\s+agent)",
                r"\bpipe\s+bomb\b", r"\bhow\s+to\s+make\s+(meth|methamphetamine|ricin|sarin)\b"],
    "self_harm": [r"\bhow\s+to\s+(kill|end)\s+(myself|my\s+life)\b", r"\bcommit\s+suicide\b",
                  r"\bways?\s+to\s+(die|self[\s-]harm)\b"],
    "cyber": [r"\b(write|create)\s+(a\s+)?(ransomware|keylogger|botnet)\b",
              r"\bsteal\s+(credit\s+card|password|credential)s?\b",
              r"\bsql\s+injection\s+(attack|payload)\s+for\b"],
    "illegal": [r"\bhow\s+to\s+(launder\s+money|buy\s+(drugs|cocaine|heroin)\s+online)\b",
                r"\bhire\s+(a\s+)?hit\s?man\b"],
}
_UNSAFE_RE = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in UNSAFE_LEXICON.items()}

SUPPORTED_LANGS = {"hi", "bn", "ta", "te", "en", "mr", "gu", "kn", "ml", "pa", "or",
                   "as", "ne", "sa", "ur"}

_SCRIPT_LANG = [
    (0x0900, 0x097F, "hi"), (0x0980, 0x09FF, "bn"), (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"), (0x0B00, 0x0B7F, "or"), (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"), (0x0C80, 0x0CFF, "kn"), (0x0D00, 0x0D7F, "ml"),
    (0x0600, 0x06FF, "ur"),
]


def detect_lang(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, lang in _SCRIPT_LANG:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
        else:
            if ch.isalpha() and cp < 0x0250:
                counts["en"] = counts.get("en", 0) + 1
    return max(counts, key=counts.get) if counts else "en"


def _normalize_confusables(text: str) -> str:
    """Fold homoglyphs so unicode tricks cannot slip past the pattern pack."""
    return unicodedata.normalize("NFKC", text)


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    out = text
    for name, rx in PII_PATTERNS.items():
        if rx.search(out):
            found.append(name)
            out = rx.sub(f"[{name.upper()}_REDACTED]", out)
    return out, found


def input_rails(text: str, max_chars: int = 1000) -> RailResult:
    """Layer 1 -- cheap structural checks (~1ms)."""
    raw = (text or "").strip()
    if not raw:
        return RailResult(Decision.BLOCK, "empty", "empty transcript", "L1_input")
    if len(raw) > max_chars:
        return RailResult(Decision.BLOCK, "too_long",
                          f"query exceeds {max_chars} chars", "L1_input")

    norm = _normalize_confusables(raw)

    # Pure noise: STT on silence yields filler with almost no letters.
    letters = sum(1 for c in norm if c.isalpha())
    if letters < max(2, int(len(norm) * 0.25)):
        return RailResult(Decision.BLOCK, "noise", "no intelligible speech", "L1_input")

    for rx in _INJECTION_RE:
        if rx.search(norm):
            return RailResult(Decision.BLOCK, "prompt_injection",
                              f"injection pattern: {rx.pattern[:40]}", "L1_input")

    if _B64_RE.search(norm.replace(" ", "")):
        return RailResult(Decision.BLOCK, "prompt_injection",
                          "encoded blob in query", "L1_input")

    lang = detect_lang(norm)
    if lang not in SUPPORTED_LANGS:
        return RailResult(Decision.BLOCK, "unsupported_language",
                          f"language {lang} not supported", "L1_input")

    redacted, pii = redact_pii(norm)
    return RailResult(Decision.ALLOW, redacted_text=redacted,
                      details={"lang": lang, "pii": pii}, layer="L1_input")


def safety_rails(text: str) -> RailResult:
    """Layer 2 -- in-process safety classification (~1ms here, no model call)."""
    norm = _normalize_confusables(text or "")
    for category, patterns in _UNSAFE_RE.items():
        for rx in patterns:
            if rx.search(norm):
                return RailResult(Decision.BLOCK, category,
                                  f"unsafe request ({category})", "L2_safety")
    return RailResult(Decision.ALLOW, layer="L2_safety")


# Cross-lingual retrieval is measured, not assumed, and it does not work well
# enough here to answer on. The corpus is Indic; an English query must match
# through the static embedder's weak cross-lingual alignment (~0.25 cosine for a
# correct hi/en pair vs ~0.59 same-language). Measured over 8 in-domain and 12
# out-of-domain English queries, the two classes overlap almost entirely:
#
#   in-domain  cosine 0.117-0.490,  BM25 0.00-2.81
#   off-topic  cosine 0.126-0.471,  BM25 0.00-5.08   (BM25 is *higher* for OOD)
#
# No threshold on either signal separates them, so a permissive cross-lingual
# floor buys in-domain recall by letting off-topic English through -- it trades
# a false-positive problem for a hallucination problem, which is worse. We
# therefore hold cross-lingual answers to the same-language bar: English queries
# about Indic-only content abstain rather than answer ungrounded. Fixing this
# properly needs a true multilingual encoder on the retrieval path, which is
# documented as a known limitation rather than hidden behind a tuned constant.
CROSS_LINGUAL_ABSTAIN_THRESHOLD = 0.44


def retrieval_rails(hits: list, abstain_threshold: float = 0.44,
                    min_score_gap: float = 0.0,
                    query_lang: str | None = None) -> RailResult:
    """Layer 3 -- off-topic detection via retrieval confidence (free).

    Uses raw cosine rather than the fused RRF score: RRF is rank-based, so its
    magnitude says nothing about whether anything relevant was actually found.
    """
    if not hits:
        return RailResult(Decision.ABSTAIN, "no_results", "empty retrieval", "L3_retrieval")

    top_cos = float(hits[0].meta.get("cosine", 0.0))

    # When the best passage is in another language, score against the
    # cross-lingual scale instead of the same-language one.
    top_lang = getattr(hits[0], "lang", "") or ""
    cross = bool(query_lang and top_lang and top_lang != query_lang)
    threshold = min(abstain_threshold, CROSS_LINGUAL_ABSTAIN_THRESHOLD) if cross \
        else abstain_threshold

    if top_cos < threshold:
        return RailResult(Decision.ABSTAIN, "off_topic",
                          f"top cosine {top_cos:.3f} < {threshold}",
                          "L3_retrieval",
                          details={"top_cosine": top_cos, "cross_lingual": cross})

    if min_score_gap > 0 and len(hits) >= 5:
        cosines = [float(h.meta.get("cosine", 0.0)) for h in hits[:5]]
        if (cosines[0] - cosines[-1]) < min_score_gap:
            return RailResult(Decision.ABSTAIN, "flat_distribution",
                              "no clear best passage", "L3_retrieval",
                              details={"gap": cosines[0] - cosines[-1]})

    return RailResult(Decision.ALLOW, layer="L3_retrieval",
                      details={"top_cosine": top_cos})


def output_rails(answer_text: str, context: list[str], tokenize,
                 groundedness_threshold: float = 0.55) -> RailResult:
    """Layer 4 -- groundedness, numeric and entity consistency.

    Every claim sentence must overlap the retrieved context, and any number in
    the answer must appear in the context. Numeric hallucinations are the most
    damaging kind and the cheapest to catch.
    """
    if not answer_text or not answer_text.strip():
        return RailResult(Decision.ABSTAIN, "empty_answer", "nothing to ship", "L4_output")

    ctx_tokens: set[str] = set()
    for c in context:
        ctx_tokens |= set(tokenize(c))
    if not ctx_tokens:
        return RailResult(Decision.ABSTAIN, "no_context", "empty context", "L4_output")

    ans_tokens = set(tokenize(answer_text))
    if not ans_tokens:
        return RailResult(Decision.ABSTAIN, "empty_answer", "no tokens", "L4_output")

    overlap = len(ans_tokens & ctx_tokens) / len(ans_tokens)
    if overlap < groundedness_threshold:
        return RailResult(Decision.ABSTAIN, "ungrounded",
                          f"overlap {overlap:.2f} < {groundedness_threshold}",
                          "L4_output", details={"groundedness": overlap})

    # Numbers must be traceable to the context.
    ctx_all = " ".join(context)
    nums = set(re.findall(r"\d+(?:[.,]\d+)?", answer_text))
    ctx_nums = set(re.findall(r"\d+(?:[.,]\d+)?", ctx_all))
    unsupported = {n for n in nums if n not in ctx_nums}
    if unsupported:
        return RailResult(Decision.ABSTAIN, "numeric_hallucination",
                          f"unsupported numbers: {sorted(unsupported)[:3]}",
                          "L4_output", details={"unsupported": sorted(unsupported)})

    return RailResult(Decision.ALLOW, layer="L4_output",
                      details={"groundedness": overlap})
