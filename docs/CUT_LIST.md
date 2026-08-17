# Cut list

Features deliberately not shipped, and why. Recorded so the gaps are visible
rather than discovered.

| Cut | Reason |
|---|---|
| Streaming STT (WebSocket) | `STTProvider` has the `stream()` slot; only `transcribe_once` is implemented. This is the biggest remaining `T_e2e_voice` win. REST STT works today. |
| Speculative retrieval on partial transcripts | Depends on streaming STT. |
| Client-side Silero VAD | Browser `MediaRecorder` start/stop covers the demo; VAD is a perceived-latency optimisation, not a correctness one. |
| LLM-generated contextual chunk summaries (S4) | The deterministic variant is implemented and benchmarked. The LLM variant needs one call per chunk; `context_fn` is the hook. |
| Cerebras / Sarvam LLM bake-off | Groq is integrated with key rotation and measured. A three-way bake-off was cut for time; the client is provider-shaped, not Groq-shaped. |
| OpenTelemetry export | `RunContext` records per-stage timings and every stage is span-shaped, but no OTLP exporter is wired. `/metrics` serves the same data. |
| Deterministic trace replay | `RunContext` carries `trace_id` and events; `replay.py` is not built. |
| Full 55.6GB corpus | Deployed index is the `dev` profile (9,196 chunks). `CORPUS_PROFILE=demo\|full` scales it; past ~10⁶ chunks the flat sweep needs HNSW. |
| Playwright E2E suite | The live site was verified in-browser (load, query, abstention, citations, mobile 375px, zero console errors) but the specs are not committed as a suite. |
| Lighthouse CI report | Not run. The bundle is 4KB gzipped and the page is static, but the report is not committed, so no score is claimed. |
