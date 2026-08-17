"""Build and persist the retrieval index.

Runs at build time, never at boot: the container must be serving in under ~20s,
and embedding a few hundred thousand chunks does not fit in that window.
Emits index/manifest.json as the proof-of-work artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ingest.chunkers.strategies import chunk_record  # noqa: E402
from ingest.loader import load_passages  # noqa: E402

log = logging.getLogger("build_index")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001 - manifest must build outside a git checkout
        return "unknown"


def build(profile: str, langs: list[str], rows: int, strategy: str,
          out_dir: Path, token: str | None = None) -> dict:
    from retrieval.embedder import Embedder
    from retrieval.engine import HybridIndex

    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading passages profile=%s langs=%s rows=%d", profile, langs, rows)
    t0 = time.perf_counter()
    recs = load_passages(profile=profile, langs=langs, rows_per_lang=rows, token=token)
    t_load = time.perf_counter() - t0
    if not recs:
        raise SystemExit("no passages loaded -- check dataset access")

    log.info("chunking with %s", strategy)
    t0 = time.perf_counter()
    chunks = []
    for r in recs:
        chunks.extend(chunk_record(r, strategy))
    t_chunk = time.perf_counter() - t0

    log.info("embedding + indexing %d chunks", len(chunks))
    embedder = Embedder()
    index = HybridIndex()
    stats = index.build(chunks, embedder)

    # Parent passage store: child chunks are swapped for full passages at
    # generation time, so keep the originals addressable by passage_id.
    parents = {r.passage_id: {"text": r.text, "lang": r.lang, "answer": r.answer,
                              "query": r.parent_query, "query_type": r.query_type}
               for r in recs}

    with open(out_dir / "index.pkl", "wb") as f:
        pickle.dump({
            "chunk_ids": index.chunk_ids, "parent_ids": index.parent_ids,
            "texts": index.texts, "langs": index.langs, "metas": index.metas,
            "codes": index.codes, "scale": index.scale, "vectors": index.vectors,
            "corpus_tokens": index._corpus_tokens, "parents": parents,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "corpus_profile": profile,
        "languages": langs,
        "rows_per_lang": rows,
        "n_passages": len(recs),
        "n_chunks": len(chunks),
        "chunk_strategy": strategy,
        "embed_model": embedder.model_name,
        "embed_dim": embedder.dim,
        "quantization": "int8",
        "mean_chunk_tokens": round(sum(c.n_tokens for c in chunks) / max(len(chunks), 1), 1),
        "load_seconds": round(t_load, 2),
        "chunk_seconds": round(t_chunk, 2),
        "build_seconds": round(time.perf_counter() - t_start, 2),
        "index_bytes": (out_dir / "index.pkl").stat().st_size,
        "int8_bytes": stats["int8_bytes"],
        "float32_bytes": stats["float32_bytes"],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        **{k: v for k, v in stats.items() if k.endswith("_s")},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("manifest: %s", json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.getenv("CORPUS_PROFILE", "dev"))
    ap.add_argument("--langs", default=os.getenv("CORPUS_LANGS", "hi,ta,te,bn"))
    ap.add_argument("--rows", type=int, default=int(os.getenv("CORPUS_ROWS_PER_LANG", "500")))
    ap.add_argument("--strategy", default=os.getenv("CHUNK_STRATEGY", "s1_recursive"))
    ap.add_argument("--out", default=os.getenv("INDEX_PATH", str(ROOT / "index")))
    args = ap.parse_args()

    build(args.profile, [x for x in args.langs.split(",") if x], args.rows,
          args.strategy, Path(args.out), os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()
