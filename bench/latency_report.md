# Latency report

`3` independent runs x `320` stratified queries, measured in-process (server-side truth, cache-cold unless stated).

| Metric | P50 | P70 | P90 | P100 | n |
|---|---|---|---|---|---|
| T_retrieval | 0.63 | 0.65 | 0.71 | 0.84 | 960 |
| T_pipeline | 1.56 | 1.79 | 2.06 | 2.59 | 960 |
| T_pipeline (cache-warm) | 0.03 | 0.04 | — | 1.31 | 300 |

**Budget 200ms — P50 PASS, P70 PASS.**

Run-to-run P50: mean 1.56ms, stdev 0.010ms (1.57, 1.56, 1.55). A single run is not evidence, so the spread is reported.

## Per-stage contribution

| Stage | P50 (ms) | P90 (ms) |
|---|---|---|
| guard_in | 0.019 | 0.030 |
| cache_probe | 0.000 | 0.000 |
| embed | 0.050 | 0.083 |
| dense | 0.410 | 0.436 |
| sparse | 0.069 | 0.088 |
| fuse | 0.074 | 0.087 |
| rerank | 0.024 | 0.028 |
| guard_retrieval | 0.001 | 0.002 |
| extract | 0.696 | 1.151 |
| guard_out | 0.179 | 0.242 |

## By query kind (P50)

| Kind | P50 (ms) |
|---|---|
| in_corpus | 1.58 |
| out_of_corpus | 0.71 |

## What is in each number

- **T_retrieval** — embed + dense + sparse + fuse + rerank.
- **T_pipeline** — the brief's 200ms budget: transcript in to grounded answer out, including all four guardrail layers. No external network.
- **T_quality_ttft** — the LLM refinement, measured separately (~830ms P50 from India to Groq). It cannot fit in 200ms and is never counted inside T_pipeline.
