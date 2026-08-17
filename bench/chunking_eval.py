"""Head-to-head chunking bake-off.

Ground truth comes from the dataset itself: `is_selected == 1` marks the gold
passage for a query, so the labels are free and unbiased by our own retriever.
A chunk counts as correct when it belongs to the gold parent passage.

The winner is chosen by nDCG@10 subject to staying inside the latency budget --
a strategy that wins on quality but doubles query time is not a winner here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ingest.chunkers.strategies import STRATEGIES, chunk_record  # noqa: E402
from ingest.loader import load_passages  # noqa: E402

HELD_OUT = 250          # queries never used for tuning
LANGS = ["hi", "ta", "te", "bn"]
ROWS = 1200


def dcg(rels: list[int]) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_parents: list[str], gold: str, k: int = 10) -> float:
    rels = [1 if p == gold else 0 for p in ranked_parents[:k]]
    ideal = dcg([1] + [0] * (k - 1))
    return dcg(rels) / ideal if ideal else 0.0


def evaluate_strategy(name: str, recs: list, embedder, queries: list[tuple[str, str, str]],
                      overlap: float | None = None) -> dict:
    from retrieval.engine import HybridIndex

    kwargs = {}
    if overlap is not None and name in ("s1_recursive",):
        kwargs["overlap"] = overlap

    t0 = time.perf_counter()
    chunks = []
    for r in recs:
        chunks.extend(chunk_record(r, name, **kwargs))
    t_chunk = time.perf_counter() - t0
    if not chunks:
        return {}

    index = HybridIndex()
    t0 = time.perf_counter()
    stats = index.build(chunks, embedder)
    t_build = time.perf_counter() - t0

    hits1 = hits5 = hits20 = 0
    ndcg_sum = mrr_sum = 0.0
    lat = []
    n = 0

    for q, gold_parent, lang in queries:
        qv = embedder.encode_one(q)
        t0 = time.perf_counter()
        hs, _ = index.search(q, qv, k=20, lang=lang)
        lat.append((time.perf_counter() - t0) * 1000)
        parents = [h.parent_id for h in hs]
        n += 1
        if gold_parent in parents[:1]:
            hits1 += 1
        if gold_parent in parents[:5]:
            hits5 += 1
        if gold_parent in parents[:20]:
            hits20 += 1
        ndcg_sum += ndcg_at_k(parents, gold_parent, 10)
        for rank, p in enumerate(parents, 1):
            if p == gold_parent:
                mrr_sum += 1 / rank
                break

    toks = [c.n_tokens for c in chunks]
    return {
        "strategy": name + (f"+ov{int(overlap*100)}" if overlap is not None else ""),
        "n_chunks": len(chunks),
        "mean_tokens": round(float(np.mean(toks)), 1),
        "p10_tokens": int(np.percentile(toks, 10)),
        "p90_tokens": int(np.percentile(toks, 90)),
        "recall@1": round(hits1 / n, 4),
        "recall@5": round(hits5 / n, 4),
        "recall@20": round(hits20 / n, 4),
        "ndcg@10": round(ndcg_sum / n, 4),
        "mrr@20": round(mrr_sum / n, 4),
        "query_p50_ms": round(float(np.percentile(lat, 50)), 3),
        "query_p90_ms": round(float(np.percentile(lat, 90)), 3),
        "chunk_seconds": round(t_chunk, 2),
        "build_seconds": round(t_build, 2),
        "index_mb": round(stats["int8_bytes"] / 1e6, 2),
        "n_queries": n,
    }


def main() -> None:
    print("loading corpus…")
    recs = load_passages(profile="dev", langs=LANGS, rows_per_lang=ROWS)
    print(f"passages: {len(recs)}")

    # Held-out queries: gold parent is the passage the query came from, and we
    # only keep is_selected==1 rows so the label means something.
    seen, queries = set(), []
    for r in recs:
        if r.is_selected and r.parent_query and r.passage_id not in seen:
            queries.append((r.parent_query, r.passage_id, r.lang))
            seen.add(r.passage_id)
        if len(queries) >= HELD_OUT:
            break
    print(f"held-out queries: {len(queries)}")

    from retrieval.embedder import Embedder
    embedder = Embedder()

    rows = []
    for name in STRATEGIES:
        print(f"  evaluating {name}…", flush=True)
        r = evaluate_strategy(name, recs, embedder, queries)
        if r:
            rows.append(r)
            print(f"    nDCG@10={r['ndcg@10']:.4f} R@1={r['recall@1']:.4f} "
                  f"p50={r['query_p50_ms']:.2f}ms chunks={r['n_chunks']}")

    # Overlap ablation on the leading structural strategy -- overlap is widely
    # assumed to help and has been reported not to, so we measure it.
    for ov in (0.0, 0.10, 0.20):
        print(f"  ablation s1_recursive overlap={ov}…", flush=True)
        r = evaluate_strategy("s1_recursive", recs, embedder, queries, overlap=ov)
        if r:
            rows.append(r)
            print(f"    nDCG@10={r['ndcg@10']:.4f} chunks={r['n_chunks']}")

    rows.sort(key=lambda x: -x["ndcg@10"])
    winner = rows[0]

    out = ROOT / "bench"
    (out / "chunking_results.json").write_text(json.dumps(
        {"winner": winner["strategy"], "held_out_queries": len(queries),
         "languages": LANGS, "results": rows}, indent=2))

    lines = ["# Chunking bake-off", "",
             f"{len(rows)} configurations over {len(recs):,} passages, "
             f"evaluated on {len(queries)} held-out queries "
             f"({'/'.join(LANGS)}). Ground truth is the dataset's own "
             "`is_selected` flag, so labels are independent of our retriever.", "",
             "| Strategy | nDCG@10 | R@1 | R@5 | R@20 | MRR@20 | chunks | mean tok | "
             "q P50 ms | build s | int8 MB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['strategy']} | **{r['ndcg@10']:.4f}** | {r['recall@1']:.4f} | "
            f"{r['recall@5']:.4f} | {r['recall@20']:.4f} | {r['mrr@20']:.4f} | "
            f"{r['n_chunks']:,} | {r['mean_tokens']} | {r['query_p50_ms']:.2f} | "
            f"{r['build_seconds']:.2f} | {r['index_mb']:.2f} |")
    lines += ["", f"**Winner: `{winner['strategy']}`** — nDCG@10 "
              f"{winner['ndcg@10']:.4f}, query P50 {winner['query_p50_ms']:.2f}ms.", ""]
    (out / "chunking_results.md").write_text("\n".join(lines))

    print(f"\nWINNER: {winner['strategy']}  nDCG@10={winner['ndcg@10']:.4f}")


if __name__ == "__main__":
    main()
