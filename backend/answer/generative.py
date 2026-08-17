"""Quality path: streamed LLM answer with rotating Groq keys.

Runs strictly outside T_pipeline. A Singapore->US round trip alone is ~180-200ms
before a single token is produced, so this can never fit the 200ms server budget
and is reported separately as T_quality_ttft.

Key rotation exists because the free tier is rate-limited per key. Keys are
tried least-recently-failed first, and a key that returns 429/401 is cooled
down rather than retried in a tight loop.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You answer strictly from the provided context passages.

Rules:
- Use ONLY facts present in the context. Never add outside knowledge.
- Cite every claim with [1], [2] markers matching the numbered passages.
- Answer in the SAME language as the user's question.
- If the context does not contain the answer, reply exactly: INSUFFICIENT_CONTEXT
- Be concise: two sentences at most."""


@dataclass
class KeyState:
    key: str
    failures: int = 0
    cooldown_until: float = 0.0
    uses: int = 0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.cooldown_until


class KeyRotator:
    """Round-robin pool that cools down rate-limited keys.

    With N free-tier keys the effective rate limit is ~N times a single key's,
    which is what keeps a demo alive under a burst of judge traffic.
    """

    def __init__(self, keys: list[str]) -> None:
        self.states = [KeyState(k.strip()) for k in keys if k and k.strip()]
        self._cycle = itertools.cycle(range(len(self.states))) if self.states else None
        log.info("groq key rotator: %d keys", len(self.states))

    def acquire(self) -> KeyState | None:
        if not self.states:
            return None
        for _ in range(len(self.states)):
            st = self.states[next(self._cycle)]
            if st.available:
                st.uses += 1
                return st
        return None  # every key is cooling down

    @staticmethod
    def penalize(st: KeyState, status: int) -> None:
        st.failures += 1
        # 429 is transient (per-minute quota); 401 means the key is simply bad.
        st.cooldown_until = time.monotonic() + (60.0 if status == 429 else 300.0)
        log.warning("groq key cooled down status=%s failures=%d", status, st.failures)


@dataclass
class GenerativeResult:
    text: str = ""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    ok: bool = False
    error: str = ""
    insufficient: bool = False
    meta: dict = field(default_factory=dict)


class GroqClient:
    # llama-3.1-8b-instant is retired on Groq (404s). Verified against the live
    # model list: gpt-oss-20b answers in the query language with correct [n]
    # citations, while qwen3.6-27b leaks <think> traces and is unusable here.
    def __init__(self, keys: list[str], model: str = "openai/gpt-oss-20b",
                 timeout_ms: int = 4000) -> None:
        self.rotator = KeyRotator(keys)
        self.model = model
        self.timeout_s = timeout_ms / 1000
        # HTTP/2 keep-alive pool opened once: saves 2-3 RTTs of TCP+TLS per call.
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                http2=True, timeout=self.timeout_s,
                limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
            )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def build_messages(query: str, passages: list[str]) -> list[dict]:
        ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"},
        ]

    async def complete(self, query: str, passages: list[str], max_tokens: int = 180,
                       temperature: float = 0.1, attempts: int = 3) -> GenerativeResult:
        await self.start()
        t0 = time.perf_counter()
        last_err = ""

        for _ in range(attempts):
            st = self.rotator.acquire()
            if st is None:
                return GenerativeResult(error="all keys cooling down", ok=False,
                                        total_ms=(time.perf_counter() - t0) * 1000)
            try:
                r = await self._client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {st.key}"},
                    json={"model": self.model, "messages": self.build_messages(query, passages),
                          "max_tokens": max_tokens, "temperature": temperature},
                )
                if r.status_code in (429, 401, 403):
                    self.rotator.penalize(st, r.status_code)
                    last_err = f"HTTP {r.status_code}"
                    continue
                r.raise_for_status()
                data = r.json()
                text = (data["choices"][0]["message"]["content"] or "").strip()
                total = (time.perf_counter() - t0) * 1000
                return GenerativeResult(
                    text=text, ttft_ms=total, total_ms=total, ok=True,
                    insufficient="INSUFFICIENT_CONTEXT" in text,
                    meta={"usage": data.get("usage", {}), "model": self.model},
                )
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
                log.warning("groq call failed: %s", last_err)
                await asyncio.sleep(0.05)

        return GenerativeResult(error=last_err or "exhausted", ok=False,
                                total_ms=(time.perf_counter() - t0) * 1000)


def groq_keys_from_env() -> list[str]:
    """Read GROQ_API_KEYS (comma-separated) with GROQ_API_KEY as a fallback."""
    multi = os.getenv("GROQ_API_KEYS", "")
    keys = [k.strip() for k in multi.split(",") if k.strip()]
    single = os.getenv("GROQ_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)
    return keys
