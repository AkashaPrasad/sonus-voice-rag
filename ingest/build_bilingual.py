"""Build a bilingual (Indic + English) index directly from MSMARCO-XI shards.

The generic loader streams whole rows; here we read the parquet shard directly
so the container pulls one file per language instead of the full dataset.

Indexing BOTH sides of each parallel passage pair is what makes English queries
work at all: the deployed dev index was Indic-only, so an English question had
nothing in its own language to match and retrieved noise.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ingest.chunkers.strategies import chunk_record  # noqa: E402
from ingest.loader import XI_FILE_CODE, explode_row  # noqa: E402

log = logging.getLogger("build_bilingual")


def build(langs: list[str], rows_per_lang: int, strategy: str, out: Path) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from retrieval.embedder import Embedder
    from retrieval.engine import HybridIndex

    t0 = time.perf_counter()
    seen: dict[int, object] = {}

    for lang in langs:
        code = XI_FILE_CODE.get(lang, lang)
        path = hf_hub_download("ai4bharat/MSMARCO-XI", f"validation/{code}val.parquet",
                               repo_type="dataset")
        pf = pq.ParquetFile(path)
        got = 0
        for batch in pf.iter_batches(batch_size=500):
            for row in batch.to_pylist():
                for rec in explode_row(row, lang):
                    if not rec.text or len(rec.text) < 20:
                        continue
                    key = hash(" ".join(rec.text.lower().split()))
                    if key in seen:
                        seen[key].dup_count += 1
                    else:
                        seen[key] = rec
                got += 1
            if got >= rows_per_lang:
                break
        log.info("%s: %d rows", lang, got)

    recs = list(seen.values())
    chunks = []
    for r in recs:
        chunks.extend(chunk_record(r, strategy))
    log.info("passages=%d chunks=%d", len(recs), len(chunks))

    embedder = Embedder()
    index = HybridIndex()
    stats = index.build(chunks, embedder)

    parents = {r.passage_id: {"text": r.text, "lang": r.lang, "answer": r.answer,
                              "query": r.parent_query, "query_type": r.query_type}
               for r in recs}

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "index.pkl", "wb") as f:
        pickle.dump({"chunk_ids": index.chunk_ids, "parent_ids": index.parent_ids,
                     "texts": index.texts, "langs": index.langs, "metas": index.metas,
                     "codes": index.codes, "scale": index.scale, "vectors": index.vectors,
                     "corpus_tokens": index._corpus_tokens, "parents": parents},
                    f, protocol=pickle.HIGHEST_PROTOCOL)

    from collections import Counter
    manifest = {
        "corpus_profile": "demo-bilingual",
        "languages": sorted(set(index.langs)),
        "source_languages": langs,
        "n_passages": len(recs),
        "n_chunks": len(chunks),
        "passages_by_lang": dict(Counter(r.lang for r in recs)),
        "chunk_strategy": strategy,
        "embed_model": embedder.model_name,
        "embed_dim": embedder.dim,
        "quantization": "int8",
        "mean_chunk_tokens": round(sum(c.n_tokens for c in chunks) / max(len(chunks), 1), 1),
        "build_seconds": round(time.perf_counter() - t0, 2),
        "index_bytes": (out / "index.pkl").stat().st_size,
        "int8_bytes": stats["int8_bytes"],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("manifest: %s", json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="hi,ta")
    ap.add_argument("--rows", type=int, default=2500)
    ap.add_argument("--strategy", default="s5_hierarchical")
    ap.add_argument("--out", default=str(ROOT / "index"))
    a = ap.parse_args()
    build([x for x in a.langs.split(",") if x], a.rows, a.strategy, Path(a.out))


if __name__ == "__main__":
    main()
