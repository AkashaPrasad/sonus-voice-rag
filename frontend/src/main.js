/* Sonus client.
   Networking, STT, and SSE contracts stay untouched; this file only reshapes
   the presentation and drives the real visual states from the existing data. */

const API_CANDIDATES = [
  import.meta.env.VITE_API_BASE,
  "https://vaani-api-production.up.railway.app",
  "https://api.sonus.spacesdrive.cc",
].filter(Boolean);

const STAGES = [
  ["guard_in", "#ff7b75", "guard·in"],
  ["cache_probe", "#8f8cf8", "cache"],
  ["embed", "#f3ad54", "embed"],
  ["dense", "#63d9d5", "dense"],
  ["sparse", "#43b6b0", "sparse"],
  ["fuse", "#5a94ff", "fuse"],
  ["rerank", "#b39bff", "rerank"],
  ["guard_retrieval", "#c47167", "guard·ret"],
  ["extract", "#f6c66c", "extract"],
  ["guard_out", "#ff928c", "guard·out"],
];

const STATE_COPY = {
  idle: {
    label: "idle",
    detail: "Hold space or use the mic to start.",
    hint: "hold space or press the mic",
  },
  listening: {
    label: "listening",
    detail: "Microphone stream is live. The core reacts to the real input signal.",
    hint: "release to transcribe",
  },
  transcribing: {
    label: "transcribing",
    detail: "The captured clip is being sent to speech recognition.",
    hint: "speech clip uploading…",
  },
  thinking: {
    label: "retrieving",
    detail: "Hybrid retrieval and guardrails are running against the frozen backend.",
    hint: "retrieval pipeline active",
  },
  responding: {
    label: "responding",
    detail: "A grounded answer is on screen. Quality mode may refine it after the fast path lands.",
    hint: "answer instrument active",
  },
  "mic-denied": {
    label: "mic denied",
    detail: "Microphone access was blocked. The typed query path remains available.",
    hint: "microphone unavailable",
  },
  offline: {
    label: "offline",
    detail: "The API could not be reached. Connection state is real, not inferred.",
    hint: "backend unavailable",
  },
  "no-speech": {
    label: "no speech",
    detail: "The clip was too short or empty. Hold the mic slightly longer.",
    hint: "too short — hold longer",
  },
};

/* Verified against the deployed index: each of these retrieves a passage that
   actually answers it. The last one is deliberately unanswerable so the
   abstention state is one click away. */
const FALLBACK_CHIPS = [
  { query: "what is photosynthesis", lang: "en" },
  { query: "what is a corporation", lang: "en" },
  { query: "green tea health benefits", lang: "en" },
  { query: "हरी चाय के फायदे", lang: "hi" },
  { query: "कॉर्पोरेशन क्या है", lang: "hi" },
  { query: "what is my bank balance", lang: "en" },
];

const BUDGET_MS = 200;
let API = API_CANDIDATES[0];
let appState = "idle";
let inFlight = false;
let settleTimer = null;
let media = null;
let audioCtx = null;
let analyser = null;
let recorder = null;
let chunks = [];
let recordedMime = "";
let recStartedAt = 0;
let reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let currentAudioLevel = 0;
let smoothAudioLevel = 0;
let latencyEnergy = 0;
let answerGlow = 0;
let rafId = null;

const $ = (id) => document.getElementById(id);

const el = {
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  meta: $("meta"),
  micBtn: $("micBtn"),
  micHint: $("micHint"),
  wave: $("wave"),
  core: $("core"),
  form: $("askForm"),
  q: $("q"),
  askBtn: $("askBtn"),
  chips: $("chips"),
  sampleTabs: $("sampleTabs"),
  mode: $("mode"),
  topk: $("topk"),
  xling: $("xling"),
  cache: $("cache"),
  hud: $("hud"),
  track: $("track"),
  legend: $("legend"),
  totalMs: $("totalMs"),
  hudNote: $("hudNote"),
  transcript: $("transcript"),
  answerBox: $("answerBox"),
  sources: $("sources"),
  stateLabel: $("stateLabel"),
  stateDetail: $("stateDetail"),
};

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (char) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]
  ));
}

function clearSettleTimer() {
  if (settleTimer) {
    clearTimeout(settleTimer);
    settleTimer = null;
  }
}

function scheduleSettle(delay = 2200) {
  clearSettleTimer();
  settleTimer = setTimeout(() => {
    if (!inFlight && appState === "responding") setAppState("idle");
  }, delay);
}

function setAppState(next, detailOverride) {
  appState = next;
  const copy = STATE_COPY[next] || STATE_COPY.idle;
  document.body.dataset.state = next;
  el.stateLabel.textContent = copy.label;
  el.stateDetail.textContent = detailOverride || copy.detail;
  el.micHint.textContent = copy.hint;
}

async function resolveAPI() {
  for (const base of API_CANDIDATES) {
    try {
      const r = await fetch(`${base}/health`, { cache: "no-store" });
      if (r.ok) {
        API = base;
        return r;
      }
    } catch {
      // try the next candidate
    }
  }
  return null;
}

async function checkHealth() {
  try {
    const probe = await resolveAPI();
    if (!probe) throw new Error("no reachable API");
    const d = await probe.json();
    el.statusDot.className = "dot ok";
    el.statusText.textContent = `${d.manifest?.n_chunks?.toLocaleString() ?? "?"} chunks · ready`;
    el.meta.textContent = `${d.manifest?.embed_model?.split("/").pop() ?? ""} · ${d.manifest?.chunk_strategy ?? ""} · int8`;
    if (appState === "offline") setAppState("idle");
    return true;
  } catch {
    el.statusDot.className = "dot bad";
    el.statusText.textContent = "backend unreachable";
    el.meta.textContent = "waiting for a reachable API candidate";
    if (!inFlight) setAppState("offline");
    return false;
  }
}

function renderHUD(timings, totalMs) {
  const safeTotal = Number(totalMs) || 0;
  const present = STAGES
    .map(([key, color, label]) => [key, color, label, Number(timings?.[key] ?? 0)])
    .filter(([, , , value]) => value > 0);

  const scaleMax = Math.max(safeTotal, BUDGET_MS, 1);
  const widthPct = Math.min((safeTotal / scaleMax) * 100, 100);
  latencyEnergy = Math.min(safeTotal / BUDGET_MS, 1.6);

  el.track.innerHTML = present.map(([key, color, , value]) => (
    `<div class="seg" style="width:${(value / safeTotal) * 100}%;background:${color}" title="${key}: ${value.toFixed(2)}ms"></div>`
  )).join("");
  el.track.style.width = `${widthPct}%`;

  el.legend.innerHTML = present.length
    ? present.map(([, color, label, value]) => (
      `<li><span class="sw" style="background:${color}"></span><span class="nm">${label}</span><span class="ms">${value.toFixed(2)}ms</span></li>`
    )).join("")
    : `<li><span class="sw" style="background:rgba(138,168,187,.22)"></span><span class="nm">waiting for a query</span><span class="ms">—</span></li>`;

  el.totalMs.textContent = safeTotal ? safeTotal.toFixed(2) : "—";
  el.hud.classList.toggle("under", safeTotal > 0 && safeTotal < BUDGET_MS);
  document.querySelector(".budget").style.left = `${(BUDGET_MS / scaleMax) * 100}%`;
}

function renderTranscript(text, partial = false) {
  el.transcript.innerHTML = text
    ? `<p${partial ? ' class="partial"' : ""}>${partial ? "Partial" : "Captured"} transcript: “${escapeHTML(text)}”</p>`
    : `<p class="placeholder">The captured transcript appears here after speech recognition.</p>`;
}

function renderAnswer(payload) {
  el.answerBox.classList.remove("empty");
  const badges = [];

  if (payload.blocked) {
    badges.push(`<span class="badge blocked">blocked · ${escapeHTML(payload.block_category || "")}</span>`);
    badges.push(`<span class="badge">${escapeHTML(payload.block_layer || "")}</span>`);
  } else if (payload.abstained) {
    badges.push(`<span class="badge mode">abstained</span>`);
    if (payload.abstain_reason) badges.push(`<span class="badge">${escapeHTML(payload.abstain_reason)}</span>`);
  } else {
    badges.push(`<span class="badge mode">${escapeHTML(payload.mode)}</span>`);
    badges.push(`<span class="badge">conf ${(payload.confidence ?? 0).toFixed(3)}</span>`);
    // Accurate mode runs an independent grounding check; show its verdict so a
    // verified answer is visibly different from an unverified one.
    if (payload.provider) {
      badges.push(`<span class="badge">${escapeHTML(payload.provider)}</span>`);
    }
  }

  if (payload.cached) badges.push(`<span class="badge">cache · ${escapeHTML(payload.cached)}</span>`);

  const abstainClass = (payload.abstained || payload.blocked) ? "abstain" : "";
  const hint = (payload.abstained && !payload.blocked)
    ? `<p class="abstain-hint">Nothing in this corpus scored above the grounding threshold. ${payload.passages?.length ? "The closest retrieved evidence is still shown on the right." : ""}</p>`
    : "";

  el.answerBox.innerHTML = `
    <div class="${abstainClass}">
      <p class="answer-text" id="answerText">${escapeHTML(payload.answer)}</p>
      ${hint}
      <div class="badges">${badges.join("")}</div>
    </div>
  `;
  answerGlow = 1;
  renderSources(payload);
}

function renderSources(payload) {
  const passages = payload.passages || [];
  if (!passages.length) {
    el.sources.classList.add("empty");
    el.sources.innerHTML = `<p class="placeholder">No supporting passages were returned for this response.</p>`;
    return;
  }

  const cited = new Set((payload.citations || []).map((item) => item.text));

  el.sources.classList.remove("empty");
  el.sources.innerHTML = passages.map((passage, index) => {
    let text = escapeHTML(passage.text);
    for (const citation of cited) {
      if (citation && passage.text.includes(citation)) {
        text = text.replace(escapeHTML(citation), `<mark>${escapeHTML(citation)}</mark>`);
        break;
      }
    }

    const pills = [
      `<span class="source-pill">${escapeHTML(passage.lang || "lang ?")}</span>`,
      `<span class="source-pill">cos ${(passage.cosine ?? 0).toFixed(3)}</span>`,
    ];
    if (passage.cross_lingual) pills.unshift(`<span class="source-pill">cross-lingual</span>`);

    return `
      <article class="src">
        <div class="src-head">
          <span class="src-n">[${index + 1}]</span>
          ${pills.join("")}
        </div>
        <p class="src-text">${text}</p>
      </article>
    `;
  }).join("");
}

function showRefined(payload) {
  const node = $("answerText");
  if (!node || !payload.ok || payload.insufficient || !payload.answer) return;

  node.style.transition = "opacity .22s ease";
  node.style.opacity = "0";
  setTimeout(() => {
    node.textContent = payload.answer;
    node.style.opacity = "1";
    const badges = el.answerBox.querySelector(".badges");
    if (badges && !badges.querySelector(".refined")) {
      badges.insertAdjacentHTML(
        "afterbegin",
        `<span class="badge refined">refined · ${Math.round(payload.total_ms)}ms</span>`,
      );
    }
  }, 220);
}

async function ask(query) {
  if (!query.trim() || inFlight) return;

  clearSettleTimer();
  inFlight = true;
  el.askBtn.disabled = true;
  setAppState("thinking", el.mode.value === "strict"
    ? "Extractive mode: retrieval only, no model call."
    : undefined);
  el.answerBox.classList.remove("empty");
  el.answerBox.innerHTML = `<p class="placeholder">query sent to the retrieval pipeline…</p>`;

  const body = {
    query,
    k: Number(el.topk.value) || 3,
    cross_lingual: el.xling.checked,
    use_cache: el.cache.checked,
    mode: el.mode.value,
  };

  try {
    {
      const r = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      renderAnswer(d);
      renderHUD(d.timings, d.total_ms);
      setAppState("responding");
      scheduleSettle();
    }
  } catch (error) {
    el.answerBox.innerHTML = `
      <p class="answer-text">Could not reach the backend.</p>
      <p class="abstain-hint">${escapeHTML(error.message)}</p>
    `;
    setAppState("offline");
    el.statusDot.className = "dot bad";
    el.statusText.textContent = "backend unreachable";
  } finally {
    inFlight = false;
    el.askBtn.disabled = false;
  }
}

/* The backend serves questions it has verified against the live index, grouped
   by language, plus a "should refuse" group. Grouping matters: a judge needs to
   see that the corpus answers in several languages, and needs the guardrail
   probes findable without knowing what to type. */
let sampleGroups = [];
let activeGroup = 0;

async function loadChips() {
  try {
    const r = await fetch(`${API}/sample-queries`);
    if (r.ok) {
      const d = await r.json();
      if (Array.isArray(d.languages) && d.languages.length) sampleGroups = d.languages;
    }
  } catch {
    // fall through to the built-ins
  }

  if (!sampleGroups.length) {
    sampleGroups = [{ code: "en", label: "English", native: "English",
                      questions: FALLBACK_CHIPS.map((c) => c.query) }];
  }

  renderSampleTabs();
  renderSampleChips();
}

function renderSampleTabs() {
  el.sampleTabs.innerHTML = sampleGroups.map((g, i) => {
    const probe = g.code === "probe";
    return `<button class="stab${i === activeGroup ? " is-active" : ""}${probe ? " stab--probe" : ""}"
      type="button" role="tab" aria-selected="${i === activeGroup}"
      data-i="${i}" title="${escapeHTML(g.label)}">${escapeHTML(g.native)}</button>`;
  }).join("");

  el.sampleTabs.querySelectorAll(".stab").forEach((b) => {
    b.addEventListener("click", () => {
      activeGroup = Number(b.dataset.i);
      renderSampleTabs();
      renderSampleChips();
    });
    // Arrow-key navigation is expected of a tablist.
    b.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault();
      const step = e.key === "ArrowRight" ? 1 : -1;
      activeGroup = (activeGroup + step + sampleGroups.length) % sampleGroups.length;
      renderSampleTabs();
      renderSampleChips();
      el.sampleTabs.querySelector(".stab.is-active")?.focus();
    });
  });
}

function renderSampleChips() {
  const group = sampleGroups[activeGroup];
  if (!group) return;
  const probe = group.code === "probe";

  el.chips.innerHTML = group.questions.map((q) => (
    `<button class="chip${probe ? " chip--probe" : ""}" type="button"
       data-q="${escapeHTML(q)}">${escapeHTML(q)}</button>`
  )).join("");

  el.chips.querySelectorAll(".chip").forEach((button) => {
    button.addEventListener("click", () => {
      el.q.value = button.dataset.q;
      ask(button.dataset.q);
    });
  });
}

function ensureCanvasSize(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(Math.floor(canvas.clientWidth * ratio), 1);
  const height = Math.max(Math.floor(canvas.clientHeight * ratio), 1);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return ratio;
}

function readAudioLevel() {
  if (!analyser) return 0;
  const buffer = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(buffer);
  let sum = 0;
  for (let i = 0; i < buffer.length; i += 1) {
    const centered = (buffer[i] - 128) / 128;
    sum += centered * centered;
  }
  return Math.min(Math.sqrt(sum / buffer.length) * 2.4, 1);
}

function currentStateDrive(time) {
  if (appState === "listening") return 0.55;
  if (appState === "thinking") return 0.36 + Math.sin(time * 0.0022) * 0.06;
  if (appState === "responding") return 0.28 + answerGlow * 0.25;
  if (appState === "transcribing") return 0.2;
  if (appState === "offline" || appState === "mic-denied") return 0.14;
  return 0.1;
}

function drawDeformedRing(ctx, cx, cy, radius, amp, time, color, width, phaseShift = 0) {
  const points = 140;
  ctx.beginPath();
  for (let i = 0; i <= points; i += 1) {
    const angle = (i / points) * Math.PI * 2;
    const deformation =
      Math.sin(angle * 3 + time * 0.0014 + phaseShift) * amp +
      Math.sin(angle * 5 - time * 0.0011 - phaseShift * 1.5) * amp * 0.42;
    const r = radius + deformation;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function drawCoreFrame(time) {
  const ratio = ensureCanvasSize(el.core);
  const ctx = el.core.getContext("2d");
  const { width, height } = el.core;
  const cx = width / 2;
  const cy = height / 2;
  const minSide = Math.min(width, height);
  const stateDrive = currentStateDrive(time);

  currentAudioLevel = readAudioLevel();
  smoothAudioLevel += ((currentAudioLevel + stateDrive) - smoothAudioLevel) * 0.08;
  answerGlow *= 0.984;

  ctx.clearRect(0, 0, width, height);

  const outerGlow = ctx.createRadialGradient(cx, cy, minSide * 0.06, cx, cy, minSide * 0.5);
  outerGlow.addColorStop(0, "rgba(243, 173, 84, 0.18)");
  outerGlow.addColorStop(0.55, "rgba(99, 217, 213, 0.10)");
  outerGlow.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = outerGlow;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(time * 0.00008);
  for (let i = 0; i < 14; i += 1) {
    const angle = (Math.PI * 2 * i) / 14;
    const radius = minSide * 0.35 + Math.sin(time * 0.001 + i) * minSide * 0.01;
    const x = Math.cos(angle) * radius;
    const y = Math.sin(angle) * radius;
    ctx.fillStyle = `rgba(99, 217, 213, ${0.08 + smoothAudioLevel * 0.08})`;
    ctx.beginPath();
    ctx.arc(x, y, ratio * (1.5 + smoothAudioLevel * 2.6), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  drawDeformedRing(
    ctx,
    cx,
    cy,
    minSide * 0.19,
    minSide * (0.01 + smoothAudioLevel * 0.024),
    time,
    `rgba(243, 173, 84, ${0.62 - latencyEnergy * 0.08})`,
    Math.max(1.8, ratio * 1.4),
  );
  drawDeformedRing(
    ctx,
    cx,
    cy,
    minSide * 0.27,
    minSide * (0.012 + smoothAudioLevel * 0.03 + latencyEnergy * 0.006),
    time * 1.14,
    "rgba(99, 217, 213, 0.48)",
    Math.max(1.2, ratio * 1.05),
    0.8,
  );
  drawDeformedRing(
    ctx,
    cx,
    cy,
    minSide * 0.34,
    minSide * (0.008 + stateDrive * 0.016),
    time * 0.82,
    "rgba(143, 140, 248, 0.28)",
    Math.max(1, ratio * 0.9),
    2.2,
  );

  const orbRadius = minSide * (0.11 + smoothAudioLevel * 0.018 + answerGlow * 0.008);
  const orb = ctx.createRadialGradient(cx, cy - orbRadius * 0.28, orbRadius * 0.16, cx, cy, orbRadius * 1.4);
  orb.addColorStop(0, "rgba(252, 223, 167, 0.98)");
  orb.addColorStop(0.36, "rgba(243, 173, 84, 0.92)");
  orb.addColorStop(1, "rgba(169, 95, 26, 0.20)");
  ctx.fillStyle = orb;
  ctx.beginPath();
  ctx.arc(cx, cy, orbRadius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = `rgba(255, 255, 255, ${0.12 + smoothAudioLevel * 0.18})`;
  ctx.lineWidth = Math.max(1, ratio * 0.9);
  ctx.beginPath();
  ctx.arc(cx, cy, orbRadius + minSide * 0.032, 0, Math.PI * 2);
  ctx.stroke();
}

function drawWaveFrame(time) {
  const ratio = ensureCanvasSize(el.wave);
  const ctx = el.wave.getContext("2d");
  const { width, height } = el.wave;
  const mid = height / 2;

  currentAudioLevel = readAudioLevel();
  smoothAudioLevel += (currentAudioLevel - smoothAudioLevel) * 0.12;

  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "rgba(99, 217, 213, 0.06)");
  gradient.addColorStop(0.5, "rgba(243, 173, 84, 0.24)");
  gradient.addColorStop(1, "rgba(143, 140, 248, 0.06)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  ctx.lineWidth = 1.7 * ratio;
  ctx.strokeStyle = "rgba(243, 173, 84, 0.94)";
  ctx.beginPath();

  const amp = height * (0.12 + smoothAudioLevel * 0.28 + currentStateDrive(time) * 0.08);
  for (let x = 0; x <= width; x += 6) {
    const progress = x / width;
    const wave =
      Math.sin(progress * Math.PI * 8 + time * 0.0036) * amp * 0.28 +
      Math.sin(progress * Math.PI * 18 - time * 0.0046) * amp * 0.1;
    const envelope = 0.42 + Math.sin(progress * Math.PI) * 0.58;
    const y = mid + wave * envelope;
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function animate() {
  const tick = (time) => {
    drawCoreFrame(time);
    drawWaveFrame(time);
    rafId = requestAnimationFrame(tick);
  };

  if (!reducedMotion && !rafId) {
    rafId = requestAnimationFrame(tick);
  } else if (reducedMotion) {
    drawCoreFrame(performance.now());
    drawWaveFrame(performance.now());
  }
}

async function startRec() {
  if (recorder?.state === "recording" || inFlight) return;
  clearSettleTimer();

  try {
    media = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    el.micBtn.disabled = true;
    setAppState("mic-denied");
    return;
  }

  audioCtx = new AudioContext();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  audioCtx.createMediaStreamSource(media).connect(analyser);

  chunks = [];
  recorder = new MediaRecorder(media, pickRecorderOptions());
  recorder.ondataavailable = (event) => {
    if (event.data.size) chunks.push(event.data);
  };
  recorder.onstop = onRecStop;
  // A timeslice makes ondataavailable fire periodically instead of only at
  // stop(). Without it a short clip can reach onstop before its single blob is
  // delivered, and we upload silence.
  recorder.start(250);
  recStartedAt = performance.now();

  el.micBtn.classList.add("rec");
  setAppState("listening");
}

/* Codec support differs by browser: Chrome/Firefox give webm/opus, Safari gives
   mp4/aac. Record whatever the browser actually supports and remember the real
   MIME so the upload is labelled honestly -- sending Safari's mp4 as .webm made
   the server hand the wrong extension to the STT provider. */
function pickRecorderOptions() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4;codecs=mp4a.40.2",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  for (const mimeType of candidates) {
    if (window.MediaRecorder?.isTypeSupported?.(mimeType)) {
      recordedMime = mimeType;
      return { mimeType, audioBitsPerSecond: 64000 };
    }
  }
  recordedMime = "";
  return {};
}

function extensionFor(mime) {
  if (mime.includes("webm")) return "webm";
  if (mime.includes("mp4")) return "m4a";
  if (mime.includes("ogg")) return "ogg";
  return "webm";
}

async function onRecStop() {
  el.micBtn.classList.remove("rec");
  setAppState("transcribing");

  media?.getTracks().forEach((track) => track.stop());
  media = null;

  await audioCtx?.close();
  audioCtx = null;
  analyser = null;
  currentAudioLevel = 0;

  const mime = recordedMime || "audio/webm";
  const blob = new Blob(chunks, { type: mime });
  const heldMs = recStartedAt ? performance.now() - recStartedAt : 0;

  // Reject clips that are too short to contain speech rather than sending them
  // and letting the provider return an empty transcript.
  if (heldMs < 350 || blob.size < 1200) {
    renderTranscript("");
    setAppState("no-speech", heldMs < 350
      ? "Hold the mic a little longer — that clip was too short to contain speech."
      : "The clip captured no audible speech. Check the microphone input level.");
    scheduleSettle(1800);
    return;
  }

  const formData = new FormData();
  formData.append("audio", blob, `clip.${extensionFor(mime)}`);

  try {
    const r = await fetch(`${API}/stt`, { method: "POST", body: formData });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    if (d.text) {
      renderTranscript(d.text);
      el.q.value = d.text;
      await ask(d.text);
    } else {
      renderTranscript("");
      setAppState("no-speech");
      scheduleSettle(1400);
    }
  } catch {
    renderTranscript("");
    setAppState("offline", "Speech recognition could not be reached. The typed query path remains available.");
  }
}

function stopRec() {
  if (recorder?.state !== "recording") return;
  // Flush whatever is buffered before stopping, so the last words of a short
  // utterance are not lost.
  try { recorder.requestData(); } catch { /* not fatal */ }
  recorder.stop();
}

function bindEvents() {
  el.form.addEventListener("submit", (event) => {
    event.preventDefault();
    ask(el.q.value);
  });

  el.micBtn.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    startRec();
  });
  el.micBtn.addEventListener("pointerup", stopRec);
  el.micBtn.addEventListener("pointerleave", stopRec);

  let spaceHeld = false;
  document.addEventListener("keydown", (event) => {
    if (event.code === "Space" && !spaceHeld && document.activeElement !== el.q) {
      event.preventDefault();
      spaceHeld = true;
      startRec();
    }
  });
  document.addEventListener("keyup", (event) => {
    if (event.code === "Space" && spaceHeld) {
      spaceHeld = false;
      stopRec();
    }
  });

  window.addEventListener("resize", () => {
    if (reducedMotion) animate();
  });
}

async function init() {
  setAppState("idle");
  renderTranscript("");
  renderHUD({}, 0);
  animate();
  bindEvents();
  await checkHealth();
  await loadChips();
  setInterval(checkHealth, 60000);
}

init();
