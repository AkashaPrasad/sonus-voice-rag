"""Find questions the deployed corpus can actually answer.

Two stages, because retrieval confidence alone is not enough:

  1. Sample real queries from the index (every passage carries the query it was
     collected for) and keep those that retrieve strongly.
  2. Send the survivors to the live API and keep only those that come back with
     a substantive, cited answer.

Stage 2 matters. A query can score 0.9 and still produce "the context does not
state X" -- technically grounded, useless as a demo. Those are filtered out.

Re-run after any index change: the answerable set is a property of the corpus,
not of the code.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# Answers that technically ground but tell the user nothing.
HEDGE = re.compile(
    r"does not (state|specify|mention|provide)|not (stated|specified|mentioned) in|"
    r"context does not|संदर्भ में .*नहीं|उपलब्ध संदर्भ|प्रसंग में",
    re.IGNORECASE,
)


def sample_candidates(index_path: Path, per_lang: int, min_conf: float) -> dict:
    from app.main import load_index
    from harness.pipeline import Pipeline
    from retrieval.embedder import Embedder

    embedder = Embedder()
    index, parents = load_index(index_path)
    pipe = Pipeline(index, embedder)

    random.seed(7)
    by_lang: dict[str, list] = {}
    for _, p in parents.items():
        q = (p.get("query") or "").strip()
        if 14 <= len(q) <= 60:
            by_lang.setdefault(p.get("lang", "?"), []).append(q)

    out: dict[str, list] = {}
    for lang, queries in by_lang.items():
        random.shuffle(queries)
        keep = []
        for q in queries[:per_lang]:
            r = pipe.run(q, use_cache=False)
            if r["confidence"] >= min_conf and not r.get("abstained"):
                keep.append({"q": q, "conf": round(r["confidence"], 3)})
        out[lang] = sorted(keep, key=lambda x: -x["conf"])
    return out


def vet(candidates: dict, api: str, limit: int) -> dict:
    good: dict[str, list] = {}
    with httpx.Client(timeout=90) as client:
        for lang, items in candidates.items():
            kept = []
            for item in items[:limit]:
                try:
                    r = client.post(api, json={"query": item["q"], "mode": "quality",
                                               "use_cache": False}).json()
                except httpx.HTTPError:
                    continue
                answer = (r.get("answer") or "").strip()
                if r.get("abstained") or r.get("blocked"):
                    continue
                if HEDGE.search(answer) or len(answer) < 45:
                    continue
                kept.append({**item, "a": answer})
            good[lang] = kept
            print(f"  {lang}: {len(kept)} answerable of {min(len(items), limit)} tested")
    return good


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "index"))
    ap.add_argument("--api", default="https://vaani-api-production.up.railway.app/ask")
    ap.add_argument("--sample", type=int, default=400, help="queries sampled per language")
    ap.add_argument("--vet", type=int, default=45, help="candidates sent to the live API")
    ap.add_argument("--min-conf", type=float, default=0.68)
    ap.add_argument("--out", default=str(ROOT / "bench" / "demo_questions.json"))
    a = ap.parse_args()

    print("sampling corpus queries…")
    cands = sample_candidates(Path(a.index), a.sample, a.min_conf)
    for lang, items in cands.items():
        print(f"  {lang}: {len(items)} candidates")

    print("vetting against the live API…")
    good = vet(cands, a.api, a.vet)
    Path(a.out).write_text(json.dumps(good, indent=2, ensure_ascii=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
