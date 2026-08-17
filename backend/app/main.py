"""Sonus API.

The index is loaded once at startup and warmed with synthetic queries so the
first real request does not pay JIT, page-fault, or allocation costs. gc is
frozen after warmup to keep collector pauses out of the P100 tail.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import pickle
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import ORJSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("sonus")

STATE: dict = {"ready": False, "manifest": {}, "boot_ms": 0.0}

WARMUP_QUERIES = [
    "भारत की राजधानी क्या है", "what is a corporation", "தமிழ் மொழி",
    "টমেটো কি", "టమాటో అంటే ఏమిటి", "how does photosynthesis work",
    "हरी चाय के फायदे", "what is xylem",
]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    lang: str | None = None
    k: int = Field(3, ge=1, le=10)
    cross_lingual: bool = True
    use_cache: bool = True
    mode: str = Field("strict", pattern="^(strict|quality)$")


def load_index(path: Path):
    from retrieval.engine import HybridIndex
    import bm25s

    with open(path / "index.pkl", "rb") as f:
        d = pickle.load(f)

    idx = HybridIndex()
    idx.chunk_ids = d["chunk_ids"]
    idx.parent_ids = d["parent_ids"]
    idx.texts = d["texts"]
    idx.langs = d["langs"]
    idx.metas = d["metas"]
    idx.codes = d["codes"]
    idx.scale = d["scale"]
    idx.vectors = d["vectors"]
    idx._corpus_tokens = d["corpus_tokens"]
    idx._bm25 = bm25s.BM25()
    idx._bm25.index(d["corpus_tokens"], show_progress=False)
    return idx, d.get("parents", {})


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    index_path = Path(os.getenv("INDEX_PATH", str(ROOT / "index")))

    from harness.pipeline import Pipeline, SemanticCache
    from retrieval.embedder import Embedder

    log.info("loading embedder")
    embedder = Embedder(os.getenv("EMBED_MODEL", "minishlab/potion-multilingual-128M"))

    log.info("loading index from %s", index_path)
    index, parents = load_index(index_path)
    manifest_file = index_path / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}

    cache = SemanticCache(int(os.getenv("SEMANTIC_CACHE_SIZE", "2048")),
                          float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.97")))
    pipeline = Pipeline(index, embedder, cache,
                        abstain_threshold=float(os.getenv("ABSTAIN_THRESHOLD", "0.44")),
                        groundedness_threshold=float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.55")))

    # Warm every hot code path before serving: first-call costs otherwise land
    # on a real user and dominate P100.
    log.info("warming up")
    for q in WARMUP_QUERIES * 3:
        pipeline.run(q, use_cache=False)
    cache.hits = cache.misses = 0

    groq = None
    from answer.generative import GroqClient, groq_keys_from_env
    keys = groq_keys_from_env()
    if keys:
        groq = GroqClient(keys, os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                          int(os.getenv("LLM_TIMEOUT_MS", "4000")))
        await groq.start()
        log.info("groq enabled with %d keys", len(keys))
    else:
        log.warning("no GROQ keys -- quality path disabled, extractive still works")

    from stt.providers import build_stt
    stt_primary, stt_fallback = build_stt()
    log.info("stt primary=%s fallback=%s",
             type(stt_primary).__name__ if stt_primary else None,
             type(stt_fallback).__name__ if stt_fallback else None)

    # Freeze surviving objects so the collector stops scanning them.
    gc.collect()
    gc.freeze()

    STATE.update(pipeline=pipeline, index=index, embedder=embedder, cache=cache,
                 parents=parents, manifest=manifest, groq=groq, ready=True,
                 stt=stt_primary, stt_fallback=stt_fallback,
                 boot_ms=round((time.perf_counter() - t0) * 1000, 1))
    log.info("ready in %.1fms  chunks=%d", STATE["boot_ms"], len(index.chunk_ids))
    yield
    if groq:
        await groq.close()


app = FastAPI(title="Sonus", version="1.0.0", default_response_class=ORJSONResponse,
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("CORS_ORIGINS", "*").split(",") if o] or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    if not STATE.get("ready"):
        raise HTTPException(503, "warming up")
    return {"status": "ok", "ready": True, "boot_ms": STATE["boot_ms"],
            "manifest": STATE["manifest"], "cache": STATE["cache"].stats(),
            "groq_enabled": STATE.get("groq") is not None}


@app.post("/ask")
async def ask(req: AskRequest):
    if not STATE.get("ready"):
        raise HTTPException(503, "warming up")
    pipeline = STATE["pipeline"]
    # Retrieval is CPU-bound and sub-millisecond; a thread hop would cost more
    # than it saves, so run it inline on the event loop.
    result = pipeline.run(req.query, lang=req.lang, k=req.k,
                          cross_lingual=req.cross_lingual, use_cache=req.use_cache)
    return result


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Fast extractive answer first, then the refined LLM answer over SSE."""
    if not STATE.get("ready"):
        raise HTTPException(503, "warming up")
    pipeline = STATE["pipeline"]
    groq = STATE.get("groq")

    async def gen():
        fast = pipeline.run(req.query, lang=req.lang, k=req.k,
                            cross_lingual=req.cross_lingual, use_cache=req.use_cache)
        yield f"event: fast\ndata: {json.dumps(fast, ensure_ascii=False)}\n\n"

        if groq and not fast.get("abstained") and not fast.get("blocked") \
                and req.mode == "quality" and fast.get("passages"):
            t0 = time.perf_counter()
            res = await groq.complete(req.query, [p["text"] for p in fast["passages"]])
            payload = {
                "answer": res.text, "ok": res.ok, "mode": "generative",
                "ttft_ms": round(res.ttft_ms, 1),
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                "insufficient": res.insufficient, "error": res.error,
            }
            # Never blank a good extractive answer because the LLM failed.
            if not res.ok or res.insufficient:
                payload["fallback"] = "kept_extractive"
            yield f"event: refined\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    """Transcribe a clip. Falls back to the secondary provider on any failure."""
    if not STATE.get("ready"):
        raise HTTPException(503, "warming up")
    primary, fallback = STATE.get("stt"), STATE.get("stt_fallback")
    if primary is None:
        raise HTTPException(503, "no STT provider configured")

    data = await audio.read()
    if len(data) > 12_000_000:
        raise HTTPException(413, "audio too large")
    mime = audio.content_type or "audio/webm"

    for provider in (primary, fallback):
        if provider is None:
            continue
        try:
            t = await provider.transcribe_once(data, mime)
            return {"text": t.text, "language": t.language, "provider": t.provider,
                    "latency_ms": round(t.provider_latency_ms, 1)}
        except Exception as e:  # noqa: BLE001 - try the next provider, never 500
            log.warning("stt provider %s failed: %s", type(provider).__name__, e)
    raise HTTPException(502, "all STT providers failed")


@app.get("/metrics")
async def metrics():
    if not STATE.get("ready"):
        raise HTTPException(503, "warming up")
    return {"cache": STATE["cache"].stats(), "manifest": STATE["manifest"],
            "boot_ms": STATE["boot_ms"],
            "n_chunks": len(STATE["index"].chunk_ids)}


@app.get("/sample-queries")
async def sample_queries():
    """Real queries drawn from the indexed corpus, for the UI's chips."""
    parents = STATE.get("parents", {})
    out, seen = [], set()
    for pid, p in parents.items():
        q, lang = p.get("query"), p.get("lang")
        if q and lang not in seen:
            out.append({"query": q, "lang": lang})
            seen.add(lang)
        if len(out) >= 6:
            break
    return {"queries": out}
