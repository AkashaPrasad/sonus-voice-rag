"""Latency benchmark: >=300 stratified queries x N runs, with per-stage breakdown.

Reports T_retrieval and T_pipeline separately, and labels whether it ran
in-process (server-side truth) or over the network (what a user in India
experiences). Conflating those two is how submissions end up claiming
impossible numbers.
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

STAGES = ["guard_in", "cache_probe", "embed", "dense", "sparse", "fuse",
          "rerank", "guard_retrieval", "extract", "guard_out"]

# Out-of-corpus probes: personal, future, and fictional questions the corpus
# cannot answer, used to exercise the abstention path under load.
OOD = [
    "मेरे बैंक खाते में कितने पैसे हैं", "what is my wifi password",
    "என் பூனையின் பெயர் என்ன", "আমার বাড়ির ঠিকানা কি",
    "నా బ్యాంక్ నిల్వ ఎంత", "who won the 2027 quidditch world cup",
    "tell me tomorrow's lottery numbers", "how many unicorns live in my garden",
]


def pct(a: list[float], q: float) -> float:
    return float(np.percentile(a, q)) if a else 0.0


def build_query_set(parents: dict, n: int) -> list[dict]:
    """Stratify by language x length x in/out-of-corpus."""
    in_corpus = []
    for pid, p in parents.items():
        q = (p.get("query") or "").strip()
        if q:
            in_corpus.append({"q": q, "lang": p.get("lang", ""), "kind": "in_corpus",
                              "length": "short" if len(q) < 40 else "long"})
    # Deterministic stride keeps the sample reproducible across runs.
    step = max(1, len(in_corpus) // max(n - len(OOD) * 2, 1))
    sampled = in_corpus[::step][: n - len(OOD) * 2]
    out = sampled + [{"q": q, "lang": "", "kind": "out_of_corpus", "length": "short"}
                     for q in OOD * 2]
    return out[:n]


def run_local(n: int, runs: int, index_path: Path) -> dict:
    from app.main import load_index
    from harness.pipeline import Pipeline, SemanticCache
    from retrieval.embedder import Embedder

    embedder = Embedder()
    index, parents = load_index(index_path)
    queries = build_query_set(parents, n)
    print(f"query set: {len(queries)} "
          f"(in_corpus={sum(1 for q in queries if q['kind']=='in_corpus')}, "
          f"ood={sum(1 for q in queries if q['kind']=='out_of_corpus')})")

    all_runs = []
    for r in range(runs):
        cache = SemanticCache()
        pipe = Pipeline(index, embedder, cache)
        for q in queries[:40]:                      # warm before measuring
            pipe.run(q["q"], use_cache=False)

        rows = []
        for q in queries:
            t0 = time.perf_counter()
            res = pipe.run(q["q"], use_cache=False)  # cache-cold
            wall = (time.perf_counter() - t0) * 1000
            tm = res["timings"]
            rows.append({
                "kind": q["kind"], "lang": q["lang"], "length": q["length"],
                "t_pipeline": wall,
                "t_retrieval": sum(tm.get(s, 0.0) for s in
                                   ["embed", "dense", "sparse", "fuse", "rerank"]),
                "abstained": res.get("abstained", False),
                "stages": {s: tm.get(s, 0.0) for s in STAGES},
            })
        # cache-warm pass
        warm = []
        for q in queries[:100]:
            pipe.run(q["q"], use_cache=True)
            t0 = time.perf_counter()
            pipe.run(q["q"], use_cache=True)
            warm.append((time.perf_counter() - t0) * 1000)

        all_runs.append({"rows": rows, "warm": warm, "cache": cache.stats()})
        print(f"  run {r+1}/{runs}: n={len(rows)} "
              f"P50={pct([x['t_pipeline'] for x in rows],50):.2f}ms")

    return {"queries": len(queries), "runs": runs, "data": all_runs}


def summarize(res: dict) -> dict:
    per_run_p50 = []
    pipe_all, retr_all, warm_all = [], [], []
    stage_tot = {s: [] for s in STAGES}
    by_kind: dict[str, list[float]] = {}

    for run in res["data"]:
        rows = run["rows"]
        p = [x["t_pipeline"] for x in rows]
        per_run_p50.append(pct(p, 50))
        pipe_all += p
        retr_all += [x["t_retrieval"] for x in rows]
        warm_all += run["warm"]
        for x in rows:
            by_kind.setdefault(x["kind"], []).append(x["t_pipeline"])
            for s in STAGES:
                stage_tot[s].append(x["stages"][s])

    return {
        "n_per_run": res["queries"], "runs": res["runs"],
        "T_pipeline": {"P50": pct(pipe_all, 50), "P70": pct(pipe_all, 70),
                       "P90": pct(pipe_all, 90), "P100": max(pipe_all),
                       "n": len(pipe_all)},
        "T_retrieval": {"P50": pct(retr_all, 50), "P70": pct(retr_all, 70),
                        "P90": pct(retr_all, 90), "P100": max(retr_all),
                        "n": len(retr_all)},
        "T_cache_warm": {"P50": pct(warm_all, 50), "P70": pct(warm_all, 70),
                         "P100": max(warm_all), "n": len(warm_all)},
        "run_to_run_P50": {"mean": st.mean(per_run_p50),
                           "stdev": st.stdev(per_run_p50) if len(per_run_p50) > 1 else 0.0,
                           "values": per_run_p50},
        "stages_P50": {s: pct(v, 50) for s, v in stage_tot.items()},
        "stages_P90": {s: pct(v, 90) for s, v in stage_tot.items()},
        "by_kind_P50": {k: pct(v, 50) for k, v in by_kind.items()},
    }


def write_markdown(summary: dict, path: Path, budget: float = 200.0) -> None:
    t, r, w = summary["T_pipeline"], summary["T_retrieval"], summary["T_cache_warm"]
    rr = summary["run_to_run_P50"]
    lines = [
        "# Latency report", "",
        f"`{summary['runs']}` independent runs x `{summary['n_per_run']}` stratified queries, "
        f"measured in-process (server-side truth, cache-cold unless stated).", "",
        "| Metric | P50 | P70 | P90 | P100 | n |",
        "|---|---|---|---|---|---|",
        f"| T_retrieval | {r['P50']:.2f} | {r['P70']:.2f} | {r['P90']:.2f} | {r['P100']:.2f} | {r['n']} |",
        f"| T_pipeline | {t['P50']:.2f} | {t['P70']:.2f} | {t['P90']:.2f} | {t['P100']:.2f} | {t['n']} |",
        f"| T_pipeline (cache-warm) | {w['P50']:.2f} | {w['P70']:.2f} | — | {w['P100']:.2f} | {w['n']} |",
        "",
        f"**Budget {budget:.0f}ms — P50 {'PASS' if t['P50'] < budget else 'FAIL'}, "
        f"P70 {'PASS' if t['P70'] < budget else 'FAIL'}.**", "",
        f"Run-to-run P50: mean {rr['mean']:.2f}ms, stdev {rr['stdev']:.3f}ms "
        f"({', '.join(f'{v:.2f}' for v in rr['values'])}). A single run is not evidence, "
        "so the spread is reported.", "",
        "## Per-stage contribution", "",
        "| Stage | P50 (ms) | P90 (ms) |", "|---|---|---|",
    ]
    for s in STAGES:
        lines.append(f"| {s} | {summary['stages_P50'][s]:.3f} | {summary['stages_P90'][s]:.3f} |")
    lines += ["", "## By query kind (P50)", "", "| Kind | P50 (ms) |", "|---|---|"]
    for k, v in summary["by_kind_P50"].items():
        lines.append(f"| {k} | {v:.2f} |")
    lines += ["", "## What is in each number", "",
              "- **T_retrieval** — embed + dense + sparse + fuse + rerank.",
              "- **T_pipeline** — the brief's 200ms budget: transcript in to grounded "
              "answer out, including all four guardrail layers. No external network.",
              "- **T_quality_ttft** — the LLM refinement, measured separately "
              "(~830ms P50 from India to Groq). It cannot fit in 200ms and is never "
              "counted inside T_pipeline.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--index", default=str(ROOT / "index"))
    ap.add_argument("--out", default=str(ROOT / "bench"))
    args = ap.parse_args()

    res = run_local(args.n, args.runs, Path(args.index))
    summary = summarize(res)

    out = Path(args.out)
    (out / "latency_report.json").write_text(json.dumps(
        {"summary": summary,
         "raw": [{"rows": r["rows"], "cache": r["cache"]} for r in res["data"]]},
        indent=2))
    write_markdown(summary, out / "latency_report.md")

    t = summary["T_pipeline"]
    print(f"\nT_pipeline  P50={t['P50']:.2f}ms P70={t['P70']:.2f}ms P100={t['P100']:.2f}ms")
    print(f"wrote {out/'latency_report.json'} and .md")


if __name__ == "__main__":
    main()
