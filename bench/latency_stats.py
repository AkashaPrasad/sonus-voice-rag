"""Per-stage latency table with a pass/fail verdict against the budget.

Reports avg/P50/P95/P99 per stage, because that is what a reviewer reads first.
The companion `latency_bench.py` produces the full distribution and the
committed JSON; this one is the at-a-glance console summary.

Everything measured here is the in-process pipeline: retrieval, guardrails, and
extractive answering. The LLM compose step is reported separately and is never
folded into the budget number -- a network call to another continent cannot be
inside a 200ms server-side contract, so mixing them would be dishonest.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# Displayed in pipeline order. `embed` covers query vectorization; `search` sums
# the dense and sparse sweeps; `rerank` is fusion + MMR.
GROUPS = {
    "guard_in": ["guard_in"],
    "cache": ["cache_probe"],
    "embed": ["embed"],
    "search": ["dense", "sparse"],
    "rerank": ["fuse", "rerank"],
    "guard_out": ["guard_retrieval", "guard_out"],
    "extract": ["extract"],
}

DEFAULT_QUERIES = [
    "cell organelles definition", "what is lantus insulin",
    "what type of attack ip spoofing", "cold symptoms with pink eye",
    "the moon's what affect the oceans tides on earth",
    "what is the average salary for neonatal nurses",
    "नागरिक को परिभाषित करें", "कोरिया का दूसरा सबसे बड़ा शहर क्या है?",
    "मानक पैलेट आयाम क्या हैं", "असामान्य मनोविज्ञान का अध्ययन क्या है",
    "हाइड्रलिक यंत्र क्या है", "what is my bank balance",
]


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def run(index_path: Path, n: int, warmup: int) -> dict:
    from app.main import load_index
    from harness.pipeline import Pipeline, SemanticCache
    from retrieval.embedder import Embedder

    embedder = Embedder()
    index, parents = load_index(index_path)
    pipe = Pipeline(index, embedder, SemanticCache())

    # Prefer real corpus queries so the measurement reflects real retrieval work
    # rather than a handful of hand-picked strings.
    pool = [p["query"].strip() for p in parents.values()
            if (p.get("query") or "").strip() and 12 <= len(p["query"].strip()) <= 60]
    queries = (pool[:: max(1, len(pool) // n)][:n] or DEFAULT_QUERIES) if pool else DEFAULT_QUERIES

    for i in range(warmup):
        pipe.run(queries[i % len(queries)], use_cache=False)

    rows, totals = [], []
    for i in range(n):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        res = pipe.run(q, use_cache=False)
        totals.append((time.perf_counter() - t0) * 1000)
        rows.append(res["timings"])

    stages: dict[str, list[float]] = {}
    for name, keys in GROUPS.items():
        stages[name] = [sum(r.get(k, 0.0) for k in keys) for r in rows]
    stages["total"] = totals

    return {
        "n": n,
        "index": str(index_path),
        "n_chunks": len(index.chunk_ids),
        "stages": {
            name: {
                "avg": round(st.mean(v), 2),
                "p50": round(pct(v, 50), 2),
                "p95": round(pct(v, 95), 2),
                "p99": round(pct(v, 99), 2),
                "max": round(max(v), 2),
            }
            for name, v in stages.items()
        },
    }


def render(stats: dict, budget: float) -> str:
    total = stats["stages"]["total"]
    passed = total["p95"] < budget
    w = max(len(k) for k in stats["stages"]) + 2

    lines = [
        f"Ran {stats['n']} queries over {stats['n_chunks']:,} chunks",
        "",
        f"{'stage'.ljust(w)}{'avg':>9}{'p50':>9}{'p95':>9}{'p99':>9}   (ms)",
    ]
    for name, s in stats["stages"].items():
        if name == "total":
            lines.append("-" * (w + 36))
        lines.append(
            f"{name.ljust(w)}{s['avg']:>9.2f}{s['p50']:>9.2f}"
            f"{s['p95']:>9.2f}{s['p99']:>9.2f}"
        )
    lines += [
        "",
        f"Latency budget: {budget:.1f}ms | p95 total: {total['p95']:.2f}ms",
        f"{'PASS' if passed else 'FAIL'}: "
        f"{'within' if passed else 'over'} budget "
        f"({'<' if passed else '>'}{budget:.0f}ms)",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "index"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--budget", type=float, default=200.0)
    ap.add_argument("--json", default=str(ROOT / "bench" / "latency_stats.json"))
    a = ap.parse_args()

    stats = run(Path(a.index), a.n, a.warmup)
    report = render(stats, a.budget)
    print("\n" + report + "\n")

    stats["budget_ms"] = a.budget
    stats["verdict"] = "PASS" if stats["stages"]["total"]["p95"] < a.budget else "FAIL"
    Path(a.json).write_text(json.dumps(stats, indent=2))
    Path(a.json).with_suffix(".txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
