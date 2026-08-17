# Sonus · voice → grounded answer

**Live demo: [sonus.spacesdrive.cc](https://sonus.spacesdrive.cc)** · API: [`/health`](https://vaani-api-production.up.railway.app/health)

Voice-first multilingual question answering grounded in the AI4Bharat
**MSMARCO-XI** corpus. Ask in Hindi, Tamil, Telugu, or Bengali — by voice or
keyboard — and get an answer that is extracted from a real passage, with the
supporting span highlighted, or an explicit refusal when the corpus cannot
support one.

The interface shows a live per-stage latency breakdown. Those milliseconds are
the server's own measurements, the same ones `bench/latency_bench.py` records —
not a client-side estimate.

---

## The latency contract

The brief asks for the full pipeline under 200ms. That is achievable, but only
if you are precise about what is inside the budget. We publish three separate
numbers instead of one flattering one.

| Metric | P50 | P70 | P90 | P100 | n |
|---|---|---|---|---|---|
| **T_retrieval** | 0.62 ms | 0.65 ms | 0.70 ms | 0.82 ms | 960 |
| **T_pipeline** (the 200ms budget) | **1.56 ms** | **1.79 ms** | 2.03 ms | 2.67 ms | 960 |
| T_pipeline (cache-warm) | 0.03 ms | 0.04 ms | — | 1.16 ms | 300 |
| T_quality_ttft (LLM refinement) | ~828 ms | — | ~1016 ms | — | 6 |

320 stratified queries × 3 independent runs. Run-to-run P50 spread: mean
1.56 ms, **stdev 0.010 ms**. A single run is not evidence, so the spread is
reported. Full data in [`bench/latency_report.md`](bench/latency_report.md).

**What is in each number**

- `T_retrieval` — embed → dense → sparse → fuse → rerank.
- `T_pipeline` — transcript received to grounded answer produced, including all
  four guardrail layers. **Zero external network calls.** This is the brief's
  200ms budget and it passes at P50 and P70 with ~100× headroom.
- `T_quality_ttft` — the optional LLM refinement. Measured at **828 ms P50 from
  India**, and that is the honest reason it is *not* inside `T_pipeline`.

### Why the LLM is not in the budget

A Singapore→US round trip costs ~180–200 ms before a model emits its first
token. We measured 828 ms P50 end-to-end to Groq. **A cross-continent LLM call
cannot fit in a 200 ms server budget**, so we do not pretend it does.

Instead the system answers twice:

1. **Fast path** — extractive span selection, entirely in-process. MS MARCO is
   natively a span-answer dataset, so extraction is the task's own answer form,
   not a degraded shortcut. This is what lands in 1.56 ms.
2. **Quality path** — the same retrieved context sent to Groq, streamed in
   afterwards over SSE and crossfaded into the UI with a `refined` badge. If it
   fails, times out, or returns `INSUFFICIENT_CONTEXT`, the extractive answer
   stays on screen. The screen never blanks.

Per-stage P50 (from the same run):

| Stage | P50 (ms) |
|---|---|
| guard_in | 0.020 |
| embed | 0.048 |
| dense | 0.407 |
| sparse | 0.068 |
| fuse | 0.073 |
| rerank | 0.023 |
| extract | 0.694 |
| guard_out | 0.181 |

---

## Architecture

```
mic → VAD → STT (Sarvam) ─┐
                          ├→ guard_in → [semantic cache] → embed (static, 0.03ms)
keyboard ─────────────────┘                                      │
                                                                 ▼
                              hybrid retrieve: int8 dense + BM25 → RRF → MMR
                                                                 │
                                          guard_retrieval (confidence gate)
                                                                 │
                    ┌────────────────────────────────────────────┴──────────┐
             FAST PATH (in budget)                          QUALITY PATH (after)
             extractive span + citations                    Groq gpt-oss-20b, streamed
                    └────────────────────────────────────────────┬──────────┘
                                                                 ▼
                                             guard_out (groundedness, numerics)
```

**Everything on the fast path is in-process.** No vector-DB network hop, no
remote reranker, no model inference beyond a static embedding lookup.

---

## Chunking bake-off

Ten configurations, evaluated on **250 held-out queries** across hi/ta/te/bn.
Ground truth comes from the dataset's own `is_selected` flag, so the labels are
independent of our retriever.

| Strategy | nDCG@10 | R@1 | R@20 | chunks | mean tok | q P50 |
|---|---|---|---|---|---|---|
| **s5_hierarchical** | **0.6087** | 0.5680 | 0.7200 | 9,196 | 68.2 | 2.38 ms |
| s6_metadata_aware | 0.5691 | 0.5520 | 0.6640 | 4,167 | 151.5 | 1.91 ms |
| s0_fixed *(baseline)* | 0.5643 | 0.5480 | 0.6720 | 4,002 | 157.8 | 1.84 ms |
| s2_semantic | 0.5643 | 0.5480 | 0.6720 | 4,001 | 157.8 | 1.89 ms |
| s4_contextual | 0.5623 | 0.5520 | 0.6760 | 4,389 | 146.8 | 1.99 ms |
| s1_recursive | 0.5596 | 0.5480 | 0.6760 | 4,389 | 143.8 | 1.94 ms |
| s3_late | 0.5596 | 0.5480 | 0.6760 | 4,389 | 143.8 | 1.97 ms |

### Why the winner won

`s5_hierarchical` splits passages into **sentence-level child chunks (~68
tokens)** and restores the full parent passage for answering. It beats the
fixed-size baseline by **7.9% nDCG@10**.

The reason is specific to this corpus: MS MARCO answers are *spans*, usually one
or two sentences inside a passage that is mostly about something else. A
512-token chunk dilutes that span across ~150 tokens of unrelated text, and a
static bag-of-embeddings model averages it away. A 68-token chunk keeps the
answer sentence dominant in its own vector. It costs 2× the chunks and 0.5 ms of
query time, which is affordable at 1.56 ms total.

### Two findings worth stating

**Overlap did nothing.** Ablating 0% / 10% / 20% on `s1_recursive` produced
*identical* nDCG@10 (0.5596) and identical chunk counts. On this corpus, overlap
is pure index cost. We report it because the assumption that overlap helps is
widespread and, here, wrong.

**We caught our own leak.** `s4_contextual` initially scored nDCG@10 **0.878**,
far ahead of everything else. It was situating each chunk with its parent query
— and the evaluation queries *are* those parent queries, so every chunk embedded
a copy of the query it would be scored against. Corrected to use the passage's
own leading clause, it scores **0.5623**, in line with its peers. The inflated
number was leakage, not quality. See `test_contextual_does_not_leak_parent_query`.

---

## How an answer is decided

Retrieval confidence is a spectrum, not a switch, and the first version treated
it as a switch. A single cosine cut made the system refuse most ordinary
questions -- it answered Hindi and refused nearly all English. Three bands
replaced it:

| Top cosine | Band | What happens |
|---|---|---|
| `< 0.18` | nothing relevant | abstain immediately, no LLM call |
| `0.18 - 0.44` | weak evidence | passages go to the LLM, which reads them and answers or says `INSUFFICIENT_CONTEXT` |
| `>= 0.44` | confident | extractive span, then the LLM composes it into prose |

The key insight is that **a bag-of-embeddings score is a weak relevance judge
and a model reading the passage is a strong one.** Retrieval is tuned for
recall; the LLM supplies precision. Grounding is never relaxed: the model sees
only retrieved passages, and Layer 4 checks the answer against them.

This also catches a failure a threshold cannot. `what is my bank balance`
retrieves banking passages at cosine 0.625 -- high enough to answer on -- but
none of them contain *your* balance. The model refuses; the cosine would not
have. Two prompt rules were earned the same way:

- Irrelevant retrieved passages are normal and are not grounds to refuse.
  Without this the model refused whenever the top-6 contained noise.
- Questions about the user personally must refuse even when a passage shares
  the topic. Without this, `what did I eat for breakfast` answered *"I drank a
  carton of ENU protein shake"* by adopting a first-person passage as the
  user's own.

**Provider:** DeepSeek `deepseek-v4-flash` (the smaller of the two published
models -- the answer is constrained to supplied context, so latency matters more
than reasoning headroom), with Groq `gpt-oss-20b` as automatic fallback.

Measured on the deployed index: **13/15** on a mixed suite -- every ordinary
question answered in both languages, every personal, fictional, injection, and
unsafe query refused.

## Guardrails

Four layers, cheap → expensive. **114 cases: 100 adversarial + 14 in-domain
controls.**

- **Block rate (adversarial): 96.0%**
- **False-positive rate (in-domain): 28.6%**

Both numbers are reported because a guardrail that blocks everything scores
100% on the first one and is useless.

| Category | Handled | n |
|---|---|---|
| injection_en | 16/16 | 100% |
| injection_indic | 6/6 | 100% |
| unsafe_weapons / self_harm / cyber / illegal | 20/20 | 100% |
| empty_noise | 10/10 | 100% |
| off_topic_fictional / future | 12/12 | 100% |
| off_topic_personal | 16/20 | 80% |
| pii | 4/4 | 100% |
| cross_lingual_known_gap | 6/6 | 100% |
| **in_domain_control** | **10/14** | **71.4%** |

**Layer 1 — input** (~0.02 ms): injection patterns *including Indic-script
variants*, encoded-blob detection, noise/empty rejection, and PII redaction for
Aadhaar / PAN / Indian phone / UPI.

**Layer 2 — safety** (in-process): weapons, self-harm, cyber, illegal. Llama
Guard benchmarks ~459 ms P95 — more than twice our entire budget — so it belongs
in offline audit, not the hot path.

**Layer 3 — retrieval confidence** (~0.001 ms): the highest-value rail in the
system and effectively free. If nothing in the corpus scores above threshold,
the system abstains. The corpus itself tells you when a question isn't about it.

**Layer 4 — output** (~0.18 ms): groundedness overlap, plus numeric consistency —
any number in the answer must appear in the context. Numeric hallucinations are
the most damaging and the cheapest to catch.

### The abstain threshold is calibrated, not guessed

Swept over 300 verbatim corpus queries + 13 same-language paraphrases against 14
same-language out-of-domain queries → [`bench/abstain_calibration.json`](bench/abstain_calibration.json).

We ship **0.44**, not the F1-optimal 0.48. Same-language off-topic queries score
0.438 mean, so tightening to 0.48 rejects real questions for little gain. The
tradeoff is a deliberate choice and it is why the in-domain control rate is 71%
rather than higher — we would rather abstain than fabricate.

---

## Known limitations

**Cross-lingual retrieval on a single-language index does not work, and we
measured it rather than assuming it.** When the index held only Indic passages,
English queries had nothing in their own language to match: in-domain cosine
(0.117-0.490) and off-topic (0.126-0.471) overlapped almost entirely, and BM25
scored *higher* for off-topic. No threshold separated them.

The fix was not a better threshold, it was a better corpus. MSMARCO-XI ships
parallel English/Indic passage pairs, so we index **both sides**: English
queries now match English passages directly. `what is a corporation` went from
cosine 0.339 (retrieving chilli-seed and Wikipedia-stub noise) to 0.887.

Also honest about:
- The deployed index is **254,670 chunks / 177,735 passages** (88.9k Hindi +
  88.8k English), not the full 55.6 GB corpus. Depth beat breadth: an earlier
  build spread 2,500 rows across three languages and left topical holes, so
  `what is xylem` retrieved Xanax and JavaScript passages and the system
  correctly refused. One language pair with deep coverage answers far more
  real questions than three with gaps.
- Tamil, Telugu, and Bengali are supported by the pipeline but are not in the
  deployed index for that reason. `--langs hi,ta,te,bn` rebuilds with them.
- Confident hits still take the extractive path (1.56 ms); everything else is
  composed by the LLM and costs ~1-3 s. The latency table measures the
  extractive pipeline, which is what the 200 ms budget covers.
- STT is REST, not streaming. The `STTProvider` protocol has a `stream()` slot;
  the realtime WebSocket path is not implemented.

---

## What we tried and rejected

**Qdrant HNSW** → replaced with a flat int8 matrix. At ~10⁴–10⁵ chunks an
exhaustive BLAS sweep is faster than graph traversal (0.41 ms dense P50), has no
recall cliff, and removes a dependency. HNSW earns its place past ~10⁶ vectors.

**Cross-encoder reranking on the hot path** → rejected. A transformer forward
pass over 50 candidates costs 30–80 ms, which is 15–40% of the budget for a
ranking that feature-based rescoring already approximates.

**LangChain** → rejected. Its abstraction overhead is measurable against a
1.56 ms budget and it obscures exactly the tracing story that matters here.

**Llama Guard on the hot path** → rejected on published latency (~459 ms P95),
which is 2× the entire pipeline budget.

**Cloudflare Workers + Vectorize for the backend** → rejected. The free tier
allows 10 ms CPU per invocation; we cannot run embedding and fusion in that.
Vectorize's free tier is 5M stored *dimensions* ≈ 19.5k vectors at 256-dim,
against our 9,196 chunks today and 400k at demo scale. Cloudflare hosts the
frontend, where it is excellent.

**Next.js** → replaced with Vite. This is one client-rendered page with no SSR,
routing, or server components. Vite ships a 4 KB gzipped bundle to Pages with
less build surface.

**Chonkie's default splitters** → not used for boundaries. They split on `.`,
which produces **one chunk** for a Devanagari passage. Our segmenter handles
danda (`।`), double danda (`॥`), and Arabic terminators (`۔` `؟`).

---

## Two bugs worth reading about

**The tokenizer was destroying Indic retrieval.** Python's `\w` excludes Unicode
combining marks (Mn/Mc), which carry vowels in Indic scripts. A `\w+` split
shattered every word at each matra — `हिरलूम` → `['ह','रल','म']` — so BM25 was
matching consonant debris. Tokenizing by Unicode character category lifted
**Recall@1 from ~0 to 0.465** on 200 Hindi queries. Silent, and it would have
looked like "the embedding model is weak."

**Runt chunks.** Per-paragraph packing emitted 3-token fragments — the documented
reason semantic chunking underperforms recursive in published benchmarks.
Sub-threshold chunks now fold into a neighbour.

---

## Run it in 5 minutes

```bash
git clone <repo> && cd sonus-voice-rag

# backend
uv venv --python 3.12 && source .venv/bin/activate
uv pip install fastapi "uvicorn[standard]" uvloop orjson pydantic "httpx[http2]" \
               model2vec numpy bm25s datasets huggingface-hub python-multipart

python ingest/build_index.py --profile dev --langs hi,ta,te,bn \
       --rows 1000 --strategy s5_hierarchical --out index

export GROQ_API_KEYS="key1,key2"     # optional: enables the quality path
export INDEX_PATH="$PWD/index"
uvicorn backend.app.main:app --port 8000

# frontend (separate shell)
cd frontend && npm install
echo "VITE_API_BASE=http://localhost:8000" > .env.local
npm run dev
```

Verify:

```bash
pytest backend/tests -q                       # 41 tests
python bench/latency_bench.py --n 320 --runs 3
python bench/guardrail_eval.py
python bench/chunking_eval.py
```

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEYS` | — | Comma-separated. Rotated; a 429/401 cools that key down. |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | `llama-3.1-8b-instant` is retired and 404s. |
| `SARVAM_API_KEY` | — | STT. Falls back to Groq Whisper. |
| `SARVAM_STT_MODEL` | `saaras:v3` | |
| `ABSTAIN_THRESHOLD` | `0.44` | Calibrated. Do not guess this. |
| `GROUNDEDNESS_THRESHOLD` | `0.55` | Layer 4 overlap floor. |
| `CORPUS_PROFILE` | `dev` | `dev` \| `demo` \| `full` |
| `SEMANTIC_CACHE_THRESHOLD` | `0.97` | Cosine for a semantic cache hit. |
| `INDEX_PATH` | `/app/index` | Index is baked into the image, not built at boot. |

---

## Deployment

- **Frontend** — Cloudflare Pages at `sonus.spacesdrive.cc`, 4 KB gzipped JS.
- **Backend** — Railway, Singapore (`asia-southeast1-eqsg3a`). Pinned there
  deliberately: TTFB from India was ~380 ms against a ~160 ms handshake, which
  showed the container had landed in a US region.
- **Index** — baked into the Docker image at build time. Boot is a 3.4 s load
  plus 24 warmup queries, then `gc.freeze()` to keep collector pauses out of the
  tail. No cold starts, no boot-time downloads.

## Repo layout

```
backend/
  app/main.py            FastAPI, lifespan warmup, SSE dual-path
  harness/pipeline.py    stage graph, RunContext timings, semantic cache
  retrieval/             static embedder, int8 hybrid index, RRF, MMR
  answer/                extractive span selection, Groq client + key rotation
  guardrails/rails.py    four layers
  stt/providers.py       Sarvam primary, Groq Whisper fallback
  tests/                 41 tests
ingest/
  loader.py              HF fallback ladder, both real schemas
  chunkers/              Indic segmentation + 7 strategies
bench/                   latency, guardrail, chunking, calibration
frontend/                Vite + vanilla TS, latency HUD
```
