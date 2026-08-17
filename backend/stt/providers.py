"""Speech-to-text behind one protocol, so providers are swappable.

Sarvam is primary: it is India-hosted (low RTT from an Asian backend), covers
22 Indian languages, and handles code-mixing natively -- all of which match this
corpus. Groq Whisper is the implemented fallback, and both return the same
Transcript shape so the route never branches on provider.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

log = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


@dataclass
class Transcript:
    text: str
    is_final: bool = True
    language: str | None = None
    confidence: float | None = None
    provider_latency_ms: float = 0.0
    provider: str = ""


class STTProvider(Protocol):
    async def transcribe_once(self, audio: bytes, mime: str) -> Transcript: ...


class SarvamSTT:
    def __init__(self, api_key: str, model: str = "saaras:v3") -> None:
        self.api_key = api_key
        self.model = model

    async def transcribe_once(self, audio: bytes, mime: str = "audio/webm") -> Transcript:
        t0 = time.perf_counter()
        ext = "webm" if "webm" in mime else ("wav" if "wav" in mime else "mp3")
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": self.api_key},
                files={"file": (f"a.{ext}", audio, mime)},
                data={"model": self.model},
            )
            r.raise_for_status()
            d = r.json()
        return Transcript(
            text=(d.get("transcript") or "").strip(),
            language=d.get("language_code"),
            provider_latency_ms=(time.perf_counter() - t0) * 1000,
            provider="sarvam",
        )


class GroqWhisperSTT:
    """Fallback. Rotates keys so it inherits the same quota resilience."""

    def __init__(self, keys: list[str], model: str = "whisper-large-v3-turbo") -> None:
        self.keys = keys
        self.model = model
        self._i = 0

    async def transcribe_once(self, audio: bytes, mime: str = "audio/webm") -> Transcript:
        t0 = time.perf_counter()
        last = None
        for _ in range(max(len(self.keys), 1)):
            key = self.keys[self._i % len(self.keys)]
            self._i += 1
            try:
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.post(
                        GROQ_STT_URL,
                        headers={"Authorization": f"Bearer {key}"},
                        files={"file": ("a.webm", audio, mime)},
                        data={"model": self.model, "response_format": "json"},
                    )
                    if r.status_code in (429, 401):
                        last = f"HTTP {r.status_code}"
                        continue
                    r.raise_for_status()
                    d = r.json()
                return Transcript(
                    text=(d.get("text") or "").strip(),
                    provider_latency_ms=(time.perf_counter() - t0) * 1000,
                    provider="groq-whisper",
                )
            except httpx.HTTPError as e:
                last = str(e)
        raise RuntimeError(f"groq whisper failed: {last}")


def build_stt() -> tuple[STTProvider | None, STTProvider | None]:
    """Return (primary, fallback) from the environment."""
    primary = fallback = None
    sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
    if sarvam_key:
        primary = SarvamSTT(sarvam_key, os.getenv("SARVAM_STT_MODEL", "saaras:v3"))

    from answer.generative import groq_keys_from_env
    gk = groq_keys_from_env()
    if gk:
        fallback = GroqWhisperSTT(gk)

    if primary is None:
        primary, fallback = fallback, None
    return primary, fallback
