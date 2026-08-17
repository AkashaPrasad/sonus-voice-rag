"""Dataset loading with a documented fallback ladder.

The MSMARCO-XI dataset card documents `load_dataset("ai4bharat/MSMARCO-XI", "hi")`,
but the repo actually exposes a single `default` config with per-language files.
The HF viewer is also known-broken (JobManagerCrashedError), so the schema is
verified by loading a real shard rather than trusting the viewer.

Ladder, in order:
  1. load_dataset(repo, lang, streaming=True)        -- the documented path
  2. load_dataset(repo, streaming=True) + lang filter -- single default config
  3. hf_hub_download of the specific per-language file -> load_dataset("parquet"/"json")

Never downloads the full 55.6 GB: every path is streaming or single-file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Iterator

log = logging.getLogger(__name__)

XI_REPO = "ai4bharat/MSMARCO-XI"
DEV_REPO = "ai4bharat/IndicMSMARCO"

# 14 languages present in MSMARCO-XI
ALL_LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur"]

# MSMARCO-XI names its files with 3-letter codes (hinval.parquet), NOT the
# 2-letter codes the dataset card uses. Verified against list_repo_files, since
# the HF viewer is broken. A substring match on the 2-letter code is unsafe here
# ("as" matches "asmtrain" but also collides elsewhere), so map explicitly.
XI_FILE_CODE = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal",
    "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "ta": "tam",
    "te": "tel", "ur": "urd",
}

CORPUS_PROFILES = {
    "dev": {"repo": DEV_REPO, "langs": ALL_LANGS, "rows_per_lang": 1000, "split": "train"},
    "demo": {"repo": XI_REPO, "langs": ["hi", "ta", "te", "bn"], "rows_per_lang": 20000,
             "split": "validation"},
    "full": {"repo": XI_REPO, "langs": ALL_LANGS, "rows_per_lang": 100000, "split": "validation"},
}


@dataclass
class PassageRecord:
    """One passage exploded from a dataset row.

    English_passages[i] and Translated_passages[i] are a parallel pair sharing a
    passage_id, which is what makes cross-lingual retrieval possible later.
    """

    passage_id: str
    text: str
    lang: str
    source: str  # "translated" | "english"
    query_id: str
    parent_query: str
    query_type: str
    is_selected: int
    answer: str = ""
    dup_count: int = 1
    meta: dict[str, Any] = field(default_factory=dict)


def _first_present(row: dict, *names, default=None):
    """Schema drifts between the XI and IndicMSMARCO repos; tolerate both."""
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default


def explode_row(row: dict, lang: str) -> list[PassageRecord]:
    """Turn one dataset row into passage records.

    Handles BOTH real schemas, verified by loading actual shards:

    * MSMARCO-XI  -- nested: a `passages` struct holding parallel
      `English_passages[] / Translated_passages[] / is_selected[]` arrays,
      ~10 passages per row.
    * IndicMSMARCO -- flat: ONE passage per row (`passage`, `is_selected` bool,
      `language`), with no English/Translated pair at all.

    The dataset card only documents the nested form, so a nested-only
    implementation silently yields zero passages on the dev corpus.
    """
    out: list[PassageRecord] = []
    passages = _first_present(row, "passages", default=None)

    # --- Flat schema (IndicMSMARCO): one passage per row ---
    if not isinstance(passages, dict):
        text = _first_present(row, "passage", "text", default="") or ""
        if not text:
            return out
        qid = str(_first_present(row, "query_id", "qid", default=""))
        pid = _first_present(row, "passage_id", default="") or f"{lang}:{qid}:0"
        sel = _first_present(row, "is_selected", default=False)
        return [PassageRecord(
            passage_id=str(pid),
            text=str(text).strip(),
            lang=str(_first_present(row, "language", default=lang) or lang),
            source="translated",
            query_id=qid,
            parent_query=str(_first_present(row, "query", default="") or ""),
            query_type=str(_first_present(row, "query_type", default="UNKNOWN") or "UNKNOWN"),
            is_selected=int(bool(sel)),
            answer=str(_first_present(row, "answer", "Answer", default="") or ""),
        )]

    # --- Nested schema (MSMARCO-XI) ---

    translated = passages.get("Translated_passages") or []
    english = passages.get("English_passages") or []
    selected = passages.get("is_selected") or []

    qid = str(_first_present(row, "query_id", "qid", default=""))
    query = _first_present(row, "query", "Query", default="") or ""
    qtype = _first_present(row, "query_type", default="UNKNOWN") or "UNKNOWN"
    answer = _first_present(row, "Answer", "answer", default="") or ""
    eng_answer = _first_present(row, "Eng_Answer", default="") or ""

    n = max(len(translated), len(english))
    for i in range(n):
        sel = int(selected[i]) if i < len(selected) and selected[i] is not None else 0
        pid = f"{lang}:{qid}:{i}"

        if i < len(translated) and translated[i]:
            out.append(PassageRecord(
                passage_id=pid, text=str(translated[i]).strip(), lang=lang,
                source="translated", query_id=qid, parent_query=str(query),
                query_type=str(qtype), is_selected=sel, answer=str(answer),
            ))
        if i < len(english) and english[i]:
            out.append(PassageRecord(
                passage_id=f"{pid}:en", text=str(english[i]).strip(), lang="en",
                source="english", query_id=qid,
                parent_query=str(_first_present(row, "Eng_Query", default=query) or query),
                query_type=str(qtype), is_selected=sel, answer=str(eng_answer),
            ))
    return out


def stream_language(repo: str, lang: str, split: str, limit: int,
                    token: str | None = None) -> Iterator[dict]:
    """Yield up to `limit` raw rows for one language, trying each rung of the ladder."""
    from datasets import load_dataset

    # Rung 1: documented per-language config
    try:
        ds = load_dataset(repo, lang, split=split, streaming=True, token=token)
        log.info("loader: rung1 config-name OK for %s/%s", repo, lang)
        yield from islice(ds, limit)
        return
    except Exception as e:  # noqa: BLE001 - we intentionally fall through the ladder
        log.warning("loader: rung1 failed for %s/%s: %s", repo, lang, type(e).__name__)

    # Rung 2: single default config, filter by language column
    try:
        ds = load_dataset(repo, split=split, streaming=True, token=token)
        log.info("loader: rung2 default-config OK for %s", repo)
        count = 0
        for row in ds:
            tl = row.get("target_lang") or row.get("language") or row.get("lang")
            if tl is None or str(tl).lower().startswith(lang):
                yield row
                count += 1
                if count >= limit:
                    return
        return
    except Exception as e:  # noqa: BLE001
        log.warning("loader: rung2 failed for %s: %s", repo, type(e).__name__)

    # Rung 3: direct file download for the specific language.
    # This is the rung that actually fires for MSMARCO-XI in practice.
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo, repo_type="dataset", token=token)
    split_tag = "val" if split.startswith("val") else "train"
    code = XI_FILE_CODE.get(lang, lang)

    # Exact match on the known naming convention first (validation/hinval.parquet),
    # then fall back to a looser scan for repos that use 2-letter dirs (IndicMSMARCO).
    cands = [f for f in files if f.endswith(f"{code}{split_tag}.parquet")]
    if not cands:
        cands = [f for f in files
                 if f.startswith((f"{lang}/", f"{split_tag}/"))
                 and split_tag in f.lower()
                 and f.endswith((".parquet", ".jsonl", ".json"))]
    if not cands:
        raise RuntimeError(
            f"loader: no file found for {repo} lang={lang} ({code}) split={split}")

    path = hf_hub_download(repo, cands[0], repo_type="dataset", token=token)
    log.info("loader: rung3 direct-file OK -> %s", cands[0])
    fmt = "parquet" if path.endswith(".parquet") else "json"
    ds = load_dataset(fmt, data_files=path, split="train", streaming=True)
    yield from islice(ds, limit)


def load_passages(profile: str = "dev", langs: list[str] | None = None,
                  rows_per_lang: int | None = None,
                  token: str | None = None) -> list[PassageRecord]:
    """Load and explode passages for a corpus profile, deduplicating as we go."""
    cfg = CORPUS_PROFILES[profile]
    langs = langs or cfg["langs"]
    rows = rows_per_lang or cfg["rows_per_lang"]

    seen: dict[int, PassageRecord] = {}
    for lang in langs:
        got = 0
        try:
            for row in stream_language(cfg["repo"], lang, cfg["split"], rows, token):
                for rec in explode_row(row, lang):
                    if not rec.text or len(rec.text) < 20:
                        continue
                    # Hash-dedupe on normalized text; keep dup_count as a BM25 prior.
                    key = hash(" ".join(rec.text.lower().split()))
                    if key in seen:
                        seen[key].dup_count += 1
                    else:
                        seen[key] = rec
                got += 1
        except Exception as e:  # noqa: BLE001
            log.error("loader: language %s failed entirely: %s", lang, e)
            continue
        log.info("loader: %s -> %d rows", lang, got)

    return list(seen.values())
