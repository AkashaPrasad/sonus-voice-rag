"""DeepSeek client for the quality path.

Uses `deepseek-v4-flash`, the smaller/cheaper of the two published models, which
is the right trade for this workload: the answer is constrained to a short span
of supplied context, so reasoning headroom matters far less than latency.

DeepSeek is OpenAI-compatible, so this mirrors the Groq client's shape and both
sit behind the same call signature in the route.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# The context may be in a different language than the question -- that is the
# normal case for this corpus, and answering across that gap is precisely what
# the model is here for. It must still never invent content.
SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.

The passages come from a retrieval system, so expect some of them to be
irrelevant. That is normal and is NOT a reason to refuse: ignore the passages
that do not apply and answer from the ones that do. A single relevant passage
is enough.

Rules:
- Use ONLY facts stated in the context. Never add outside knowledge.
- The context may be in a different language than the question. Translate as
  needed and answer in the SAME language as the question.
- Cite the passages you used with [1], [2] markers.
- Reply with exactly INSUFFICIENT_CONTEXT only when NO passage contains
  information relevant to the question. If the passages are partially relevant,
  answer with what they do support rather than refusing.
- The corpus is general reference text. It knows nothing about the person
  asking. If the question is about the user personally -- their balance, their
  password, their possessions, what they did -- reply INSUFFICIENT_CONTEXT even
  when a passage happens to discuss the same topic. A passage written in the
  first person is the author's statement, never the user's.
- Be concise: two sentences maximum."""


@dataclass
class DeepSeekResult:
    text: str = ""
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    ok: bool = False
    error: str = ""
    insufficient: bool = False
    meta: dict = field(default_factory=dict)


class DeepSeekClient:
    def __init__(self, api_key: str, model: str = "deepseek-v4-flash",
                 timeout_ms: int = 12000) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_ms / 1000
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout_s,
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

    async def complete(self, query: str, passages: list[str], max_tokens: int = 400,
                       temperature: float = 0.1) -> DeepSeekResult:
        """Compose a grounded answer.

        `max_tokens` is generous because v4-flash spends completion tokens on
        internal reasoning before emitting text; a tight cap truncates the
        answer to an empty string.
        """
        await self.start()
        t0 = time.perf_counter()
        try:
            r = await self._client.post(
                DEEPSEEK_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model,
                      "messages": self.build_messages(query, passages),
                      "max_tokens": max_tokens, "temperature": temperature},
            )
            if r.status_code >= 400:
                return DeepSeekResult(error=f"HTTP {r.status_code}: {r.text[:160]}",
                                      total_ms=(time.perf_counter() - t0) * 1000)
            data = r.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            total = (time.perf_counter() - t0) * 1000
            return DeepSeekResult(
                text=text, ttft_ms=total, total_ms=total, ok=bool(text),
                insufficient="INSUFFICIENT_CONTEXT" in text,
                meta={"usage": data.get("usage", {}), "model": self.model},
            )
        except httpx.HTTPError as e:
            return DeepSeekResult(error=f"{type(e).__name__}: {e}",
                                  total_ms=(time.perf_counter() - t0) * 1000)
