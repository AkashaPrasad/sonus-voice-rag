# Architecture

## The one decision everything follows from

The brief asks for the whole pipeline under 200ms. Every other choice here is
downstream of taking that seriously rather than redefining it.

Physics first: Singapore→US is ~180–200ms RTT before a model emits a token. We
measured **828ms P50** for a Groq call from India. So an LLM cannot be inside a
200ms server budget, and any submission claiming otherwise is either measuring
something else or not measuring at all.

That leaves two options: miss the target, or make the in-budget path answer
without a model. We chose the second, which works here because **MS MARCO is a
span-answer dataset** — the answer is literally a sentence or two inside a
retrieved passage. Extraction is the task's native answer form.

```
                         ┌─────────── T_pipeline: 1.56ms P50 ───────────┐
mic ─ VAD ─ STT ─┐       │                                              │
                 ├─ guard_in ─ cache ─ embed ─ retrieve ─ guard_ret ─ extract ─ guard_out ─┐
keyboard ────────┘       │                                              │                  │
                         └──────────────────────────────────────────────┘                  ▼
                                                                                    stream to UI
                                    same context ──→ Groq (828ms P50) ──→ crossfade "refined"
```

The quality path is real but **reported separately**. If it fails, times out,
or returns `INSUFFICIENT_CONTEXT`, the extractive answer stays on screen.

---

## Stage graph

`backend/harness/pipeline.py`. Not a wrapper around one prompt — a typed
sequence where every stage writes its own duration into a shared `RunContext`.

That single timing source is what makes the latency claim checkable: the HUD in
the browser, the `/metrics` endpoint, and `bench/latency_bench.py` all read the
same numbers. There is no separate instrumented path that could disagree with
the demo.

`RunContext` also carries `deadline_ms` and `remaining_ms()`, so a stage can
short-circuit to its fallback rather than blow the SLA.

**Error ladder:** retry → alternate provider → extractive → abstain. A user
never receives a 500.

---

## Retrieval

| Component | Choice | Why |
|---|---|---|
| Embedding | `potion-multilingual-128M` (static, 256-dim) | Token lookup + pooling, no forward pass. **0.028ms P50.** Removes 30–80ms vs any transformer. |
| Dense | flat int8 matrix, float32 rescore | At 10⁴–10⁵ chunks a BLAS sweep beats HNSW and has no recall cliff. **0.41ms P50.** |
| Sparse | `bm25s` | Numpy-native, no JVM, no server. |
| Fusion | RRF (k=60, α=0.6) | Rank-based, so BM25's unbounded scores never need calibrating against cosine. |
| Rerank | feature-based | Cross-encoder costs 30–80ms — 15–40% of budget — for a ranking this approximates. |
| Diversity | MMR (λ=0.7) | MS MARCO reuses passages heavily; without it the top-3 is often one passage three ways. |

**int8 with float32 rescoring** is not a compromise: the int8 sweep is a
prefilter, and the top candidates are rescored in float32 so quantization error
can never reorder the final ranking. Memory drops 4.0× (measured).

### Why in-process

A remote vector DB adds 10–40ms per query. Against a 1.56ms pipeline that is a
10–25× regression for a corpus that fits in RAM. The index is baked into the
Docker image, so boot is a 3.4s load — not a build, and not a download.

---

## Chunking

Seven strategies behind one `chunk(text, meta) -> list[str]` contract so the
benchmark can swap them without special-casing. Winner chosen by nDCG@10 subject
to the latency budget: **`s5_hierarchical`, 0.6087** (see README for the table).

### Indic segmentation is the load-bearing part

Chonkie's default splitters — and most tutorials — split on `.`. Devanagari,
Bengali, Gujarati, Punjabi, and Odia terminate sentences with the **danda `।`**;
Urdu uses **`۔`**. A Hindi passage contains no periods, so `.`-splitting returns
**one chunk** and every downstream strategy degenerates to fixed-size.

`ingest/chunkers/indic.py` handles danda, double danda, Arabic terminators, and
protects decimals. Token estimation is per-script too, since Indic text packs
more characters per token than Latin.

---

## Guardrails

Four layers, cheap → expensive, ~0.28ms total on the fast path.

The highest-value rail is **Layer 3, retrieval confidence**, and it costs
0.001ms because the scores already exist. The corpus itself tells you when a
question isn't about the corpus. Everything else is defence in depth.

Layer 2 runs in-process on purpose: Llama Guard benchmarks ~459ms P95, which is
2× the entire budget. It belongs in offline audit.

### Thresholds are measured

`ABSTAIN_THRESHOLD=0.44` comes from a sweep over 300 verbatim + 13 paraphrase
in-domain queries against 14 same-language out-of-domain queries. We ship 0.44
rather than the F1-optimal 0.48 because same-language off-topic queries score
0.438 mean — tightening rejects real questions for little gain.

This is a genuine precision/recall tradeoff with no clean separation, and the
in-domain control rate (71.4%) reflects it honestly.

---

## What is deliberately not here

**Cross-lingual answering.** Measured, not assumed: for English queries against
Indic passages, in-domain cosine (0.117–0.490) and off-topic (0.126–0.471)
overlap almost entirely, and BM25 scores *higher* for off-topic. No threshold
separates them. A permissive floor trades a false-positive problem for a
hallucination problem. Cross-lingual queries abstain, and it is tracked as its
own benchmark category so the gap stays visible.

**Streaming STT.** The `STTProvider` protocol has a `stream()` slot; only
`transcribe_once` is implemented. This is the largest remaining `T_e2e_voice`
win and it is unbuilt, not hidden.

**HNSW.** Correct past ~10⁶ vectors. At 9,196 chunks it would be slower and add
a dependency.

---

## Deployment

- **Frontend** — Cloudflare Pages, 4KB gzipped JS, global edge.
- **Backend** — Railway, Singapore. Pinned deliberately: TTFB from India was
  ~380ms against a ~160ms handshake, which showed the container had landed in a
  US region. The fast path makes zero external calls, so client RTT is the only
  network cost that matters.
- **Index** — baked into the image. `gc.freeze()` after 24 warmup queries keeps
  collector pauses out of the P100 tail.

The frontend probes API candidates at boot rather than hardcoding a host, so a
pending TLS certificate on the branded domain cannot take the demo down.
