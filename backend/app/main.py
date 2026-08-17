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
    # strict   -- extractive fast path, ~2ms, no LLM
    # quality  -- flash composes from 6 passages (default)
    # accurate -- pro model, deeper retrieval, second-pass grounding verifier.
    #             Slower on purpose; correctness is the objective.
    mode: str = Field("quality", pattern="^(strict|quality|accurate)$")


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

    deepseek = None
    ds_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if ds_key:
        from answer.deepseek import DeepSeekClient
        deepseek = DeepSeekClient(ds_key, os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                                  int(os.getenv("DEEPSEEK_TIMEOUT_MS", "12000")))
        await deepseek.start()
        log.info("deepseek enabled model=%s", deepseek.model)

    prompt_guard = None
    if keys:
        from guardrails.promptguard import PromptGuardClient
        prompt_guard = PromptGuardClient(keys)
        await prompt_guard.start()
        log.info("prompt-guard enabled model=%s", prompt_guard.model)

    if not (groq or deepseek):
        log.warning("no LLM configured -- extractive path only")

    from stt.providers import build_stt
    stt_primary, stt_fallback = build_stt()
    log.info("stt primary=%s fallback=%s",
             type(stt_primary).__name__ if stt_primary else None,
             type(stt_fallback).__name__ if stt_fallback else None)

    # Freeze surviving objects so the collector stops scanning them.
    gc.collect()
    gc.freeze()

    STATE.update(pipeline=pipeline, index=index, embedder=embedder, cache=cache,
                 parents=parents, manifest=manifest, groq=groq, deepseek=deepseek,
                 prompt_guard=prompt_guard, ready=True,
                 stt=stt_primary, stt_fallback=stt_fallback,
                 boot_ms=round((time.perf_counter() - t0) * 1000, 1))
    log.info("ready in %.1fms  chunks=%d", STATE["boot_ms"], len(index.chunk_ids))
    yield
    if groq:
        await groq.close()
    if deepseek:
        await deepseek.close()
    if prompt_guard:
        await prompt_guard.close()


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
            "groq_enabled": STATE.get("groq") is not None,
            "deepseek_enabled": STATE.get("deepseek") is not None,
            "prompt_guard_enabled": STATE.get("prompt_guard") is not None}


ACCURATE_MODEL = os.getenv("DEEPSEEK_ACCURATE_MODEL", "deepseek-v4-pro")


async def _compose(query: str, passages: list[str], model: str | None = None):
    """Run the quality-path LLM, preferring DeepSeek and falling back to Groq."""
    ds, groq = STATE.get("deepseek"), STATE.get("groq")
    if ds:
        r = await ds.complete(query, passages, model=model)
        if r.ok:
            return r, "deepseek"
        log.warning("deepseek failed: %s", r.error)
    if groq:
        r = await groq.complete(query, passages)
        if r.ok:
            return r, "groq"
        log.warning("groq failed: %s", r.error)
    return None, None


async def _compose_accurate(query: str, passages: list[str]) -> tuple:
    """Accurate mode: compose with the larger model, then verify the result.

    Two independent calls rather than one careful call. The composer is invested
    in the answer it just wrote; a separate verifier with a narrow question
    ("is every claim supported?") catches overreach the composer does not see in
    its own output. On UNSUPPORTED we retry once with the verifier's objection
    fed back, and abstain if the retry is still unsupported -- a wrong answer is
    worse than no answer in this mode.
    """
    ds = STATE.get("deepseek")
    if ds is None:
        return await _compose(query, passages) + (None,)

    res = await ds.complete(query, passages, max_tokens=700, model=ACCURATE_MODEL)
    if not res.ok or res.insufficient:
        return res, "deepseek", None

    supported, reason = await ds.verify(query, res.text, passages, model=ACCURATE_MODEL)
    if supported:
        return res, "deepseek", {"verified": True, "verifier": reason}

    retry_passages = passages + [
        f"VERIFIER OBJECTION to a previous draft: {reason}. "
        "Answer again using only what the passages state, or reply "
        "INSUFFICIENT_CONTEXT."
    ]
    retry = await ds.complete(query, retry_passages, max_tokens=700, model=ACCURATE_MODEL)
    if retry.ok and not retry.insufficient:
        ok2, reason2 = await ds.verify(query, retry.text, passages, model=ACCURATE_MODEL)
        if ok2:
            return retry, "deepseek", {"verified": True, "verifier": reason2, "retried": True}
        return retry, "deepseek", {"verified": False, "verifier": reason2, "retried": True}
    return retry, "deepseek", {"verified": False, "verifier": reason, "retried": True}


@app.post("/ask")
async def ask(req: AskRequest):
    if not STATE.get("ready"):
        raise HTTPException(503, "warming up")
    pipeline = STATE["pipeline"]
    # Retrieve deeper than the UI displays: the LLM filters the extras, and a
    # relevant passage at rank 4-6 is common. Retrieval is sub-millisecond, so
    # the extra depth is effectively free.
    #
    # Retrieval is CPU-bound and sub-millisecond; a thread hop would cost more
    # than it saves, so run it inline on the event loop.
    accurate = req.mode == "accurate"

    # Accurate mode adds Meta's Prompt Guard 2 on top of the in-process regex
    # pack. The regex layer already ran inside pipeline.run(); this catches
    # phrasings no pattern list anticipated, in any script.
    if accurate:
        pg = STATE.get("prompt_guard")
        if pg is not None:
            v = await pg.check(req.query)
            if v.ok and v.is_injection:
                from answer.extractive import ABSTAIN_TEXT
                from guardrails.rails import detect_lang
                lang = req.lang or detect_lang(req.query)
                return {"answer": ABSTAIN_TEXT.get(lang, ABSTAIN_TEXT["en"]),
                        "mode": "blocked", "abstained": True, "blocked": True,
                        "block_category": "prompt_injection",
                        "block_layer": "L2_prompt_guard",
                        "block_reason": f"prompt-guard score {v.score:.4f}",
                        "confidence": 0.0, "lang": lang, "citations": [],
                        "passages": [], "timings": {"prompt_guard": round(v.latency_ms, 1)},
                        "total_ms": round(v.latency_ms, 1), "answer_mode": "accurate"}
    # Accurate mode reads more of the corpus before deciding. Retrieval is
    # sub-millisecond, so the extra depth costs nothing measurable and gives the
    # model a real chance to find the supporting passage.
    top_k = 12 if accurate else max(req.k, 6)
    result = pipeline.run(req.query, lang=req.lang, k=top_k,
                          cross_lingual=req.cross_lingual,
                          use_cache=req.use_cache and not accurate)

    # Compose with the LLM whenever we have passages and are not already
    # refusing. Two distinct jobs, both better done by a model that reads the
    # text than by a cosine score:
    #
    #  * weak evidence  -- decide whether these passages answer the question
    #  * confident hit  -- turn the extracted span into a fluent, cited answer,
    #                      and catch the case where retrieval scored well but
    #                      the passage does not actually address the question
    #                      (measured: "what is my bank balance" scores 0.625
    #                      against banking passages that never mention *your*
    #                      balance).
    #
    # Grounding is preserved: the model sees only retrieved passages and is
    # instructed to reply INSUFFICIENT_CONTEXT when they do not answer.
    # strict is the extractive-only contract: no LLM, ~2ms, which is what the
    # latency table measures. Composing there would quietly turn a 2ms mode into
    # a 2s one. Weak evidence is the exception -- there is no verified span to
    # return, so strict abstains rather than guessing.
    should_compose = (
        req.mode != "strict"
        and not result.get("blocked")
        and not result.get("abstained")
        and bool(result.get("passages"))
        and (STATE.get("deepseek") or STATE.get("groq"))
    )

    if req.mode == "strict" and result.get("weak_evidence"):
        from answer.extractive import ABSTAIN_TEXT
        lang = result.get("lang", "en")
        result.update(answer=ABSTAIN_TEXT.get(lang, ABSTAIN_TEXT["en"]),
                      mode="abstain", abstained=True,
                      abstain_reason="weak_evidence_no_llm")

    if should_compose:
        extractive_answer = result.get("answer", "")
        t0 = time.perf_counter()
        # Send more context than the UI shows. Retrieval reliably places a
        # relevant passage in the top few but not always at rank 1, and the
        # model is a better filter of the extras than a cosine cutoff is.
        ctx_passages = [p["text"] for p in result["passages"][:12 if accurate else 6]]

        verification = None
        if accurate:
            res, provider, verification = await _compose_accurate(req.query, ctx_passages)
        else:
            res, provider = await _compose(req.query, ctx_passages)
        result["timings"]["llm"] = round((time.perf_counter() - t0) * 1000, 1)
        result["answer_mode"] = req.mode
        if verification:
            result["verification"] = verification

        # In accurate mode an answer the verifier rejected twice is withheld:
        # the whole point of the mode is that a wrong answer costs more than none.
        if accurate and verification and verification.get("verified") is False:
            from answer.extractive import ABSTAIN_TEXT
            lang = result.get("lang", "en")
            result.update(answer=ABSTAIN_TEXT.get(lang, ABSTAIN_TEXT["en"]),
                          mode="abstain", abstained=True, citations=[],
                          abstain_reason="failed_verification")
            result["total_ms"] = round(result["total_ms"] + result["timings"]["llm"], 3)
            return result

        if res and res.ok and not res.insufficient:
            result.update(answer=res.text, mode="generative", abstained=False,
                          provider=provider,
                          citations=[{"passage_id": p["passage_id"], "text": p["text"],
                                      "char_start": 0, "char_end": len(p["text"]),
                                      "score": p["cosine"], "lang": p["lang"]}
                                     for p in result["passages"][:1]])
        elif res and res.insufficient:
            # The model read the passages and found no answer. That is a far more
            # reliable abstention signal than a similarity threshold.
            from answer.extractive import ABSTAIN_TEXT
            lang = result.get("lang", "en")
            result.update(answer=ABSTAIN_TEXT.get(lang, ABSTAIN_TEXT["en"]),
                          mode="abstain", abstained=True, citations=[],
                          abstain_reason="llm_insufficient_context")
        elif result.get("weak_evidence"):
            # Provider down and nothing verified to fall back to.
            from answer.extractive import ABSTAIN_TEXT
            lang = result.get("lang", "en")
            result.update(answer=ABSTAIN_TEXT.get(lang, ABSTAIN_TEXT["en"]),
                          mode="abstain", abstained=True,
                          abstain_reason="llm_unavailable")
        else:
            # Provider down but the extractive span was already verified by the
            # groundedness rail -- keep it rather than blanking the answer.
            result["answer"] = extractive_answer
            result["llm_error"] = (res.error if res else "no provider")

        result["total_ms"] = round(result["total_ms"] + result["timings"]["llm"], 3)
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
