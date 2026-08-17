"""Indic-script-aware sentence segmentation.

Naive pipelines split sentences on ".", which is simply wrong for most of the
languages in this corpus:

  * Devanagari (hi, mr, ne, sa), Bengali (bn, as), Gujarati, Punjabi, Odia all
    terminate sentences with the danda U+0964 and double-danda U+0965.
  * Urdu uses the Arabic full stop U+06D4 and Arabic question mark U+061F.
  * Tamil/Telugu/Kannada/Malayalam do use ".", but abbreviations and numeric
    decimals still need protecting.

Splitting Hindi on "." yields one giant chunk (there are no periods), which is
the exact failure mode that makes a naive chunker score badly on this dataset.
"""

from __future__ import annotations

import re

DANDA = "।"
DOUBLE_DANDA = "॥"
ARABIC_FULL_STOP = "۔"
ARABIC_QUESTION = "؟"

# Sentence terminators across every script in the corpus.
TERMINATORS = f"[.!?{DANDA}{DOUBLE_DANDA}{ARABIC_FULL_STOP}{ARABIC_QUESTION}]"

# A decimal number or a single-letter abbreviation must not end a sentence.
_PROTECT = re.compile(r"(\d)\.(\d)")
_SENT_SPLIT = re.compile(rf"(?<={TERMINATORS})\s+")

_SCRIPT_RANGES = [
    ("devanagari", 0x0900, 0x097F), ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F), ("gujarati", 0x0A80, 0x0AFF),
    ("odia", 0x0B00, 0x0B7F), ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F), ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F), ("arabic", 0x0600, 0x06FF),
]


def detect_script(text: str) -> str:
    """Majority script of a string, used to pick boundary rules and fonts."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            if ch.isalpha() and cp < 0x0250:
                counts["latin"] = counts.get("latin", 0) + 1
    return max(counts, key=counts.get) if counts else "unknown"


def split_sentences(text: str) -> list[str]:
    """Script-aware sentence split. Never returns empty strings."""
    if not text or not text.strip():
        return []
    protected = _PROTECT.sub(r"\1<DEC>\2", text.strip())
    parts = _SENT_SPLIT.split(protected)
    return [p.replace("<DEC>", ".").strip() for p in parts if p.strip()]


def approx_tokens(text: str) -> int:
    """Cheap token estimate.

    Indic scripts pack more characters per token than Latin under most
    tokenizers, so a flat chars/4 rule badly overestimates. Use a per-script
    divisor rather than pulling a tokenizer onto this path.
    """
    script = detect_script(text)
    divisor = 4.0 if script in ("latin", "unknown") else 2.6
    return max(1, int(len(text) / divisor))
