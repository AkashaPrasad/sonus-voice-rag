/* Sonus client.
   The HUD renders the server's own per-stage timings, so what a judge sees is
   the same measurement the benchmark records -- not a client-side estimate. */

/* The branded API domain is preferred, but its certificate is issued
   asynchronously after the DNS record lands. Probe it once at boot and fall
   back to the Railway service domain so the demo never depends on cert timing. */
const API_CANDIDATES = [
  import.meta.env.VITE_API_BASE,
  "https://api.sonus.spacesdrive.cc",
  "https://vaani-api-production.up.railway.app",
].filter(Boolean);

let API = API_CANDIDATES[0];
const BUDGET_MS = 200;

async function resolveAPI() {
  for (const base of API_CANDIDATES) {
    try {
      const r = await fetch(`${base}/health`, { cache: "no-store" });
      if (r.ok) { API = base; return r; }
    } catch { /* try the next candidate */ }
  }
  return null;
}

const $ = (id) => document.getElementById(id);

const el = {
  statusDot: $("statusDot"), statusText: $("statusText"),
  micBtn: $("micBtn"), micHint: $("micHint"), wave: $("wave"),
  form: $("askForm"), q: $("q"), askBtn: $("askBtn"), chips: $("chips"),
  mode: $("mode"), topk: $("topk"), xling: $("xling"), cache: $("cache"),
  hud: $("hud"), track: $("track"), legend: $("legend"), totalMs: $("totalMs"),
  hudNote: $("hudNote"), transcript: $("transcript"),
  answerBox: $("answerBox"), sources: $("sources"), meta: $("meta"),
};

/* stage → colour. Order matches pipeline execution order. */
const STAGES = [
  ["guard_in",        "#e8615d", "guard·in"],
  ["cache_probe",     "#7a6cf0", "cache"],
  ["embed",           "#e8a33d", "embed"],
  ["dense",           "#4ecdc4", "dense"],
  ["sparse",          "#3d9a94", "sparse"],
  ["fuse",            "#5b8def", "fuse"],
  ["rerank",          "#9d7bea", "rerank"],
  ["guard_retrieval", "#d4736f", "guard·ret"],
  ["extract",         "#f0c674", "extract"],
  ["guard_out",       "#c25a56", "guard·out"],
];

/* ── health ─────────────────────────────────────────── */
async function checkHealth() {
  try {
    const probe = await resolveAPI();
    if (!probe) throw new Error("no reachable API");
    const d = await probe.json();
    el.statusDot.className = "dot ok";
    el.statusText.textContent = `${d.manifest?.n_chunks?.toLocaleString() ?? "?"} chunks · ready`;
    el.meta.textContent =
      `${d.manifest?.embed_model?.split("/").pop() ?? ""} · ${d.manifest?.chunk_strategy ?? ""} · int8`;
    return true;
  } catch {
    el.statusDot.className = "dot bad";
    el.statusText.textContent = "backend unreachable";
    return false;
  }
}

/* ── latency HUD ────────────────────────────────────── */
function renderHUD(timings, totalMs) {
  const present = STAGES
    .map(([k, c, label]) => [k, c, label, Number(timings?.[k] ?? 0)])
    .filter(([, , , v]) => v > 0);

  // Scale against the budget so the 200ms marker is meaningful. A request that
  // overruns the budget rescales instead of overflowing the track.
  const scaleMax = Math.max(totalMs, BUDGET_MS);
  const widthPct = Math.min((totalMs / scaleMax) * 100, 100);

  el.track.innerHTML = present
    .map(([k, c, , v]) =>
      `<div class="seg" style="width:${(v / totalMs) * 100}%;background:${c}" title="${k}: ${v}ms"></div>`)
    .join("");
  el.track.style.width = `${widthPct}%`;

  el.legend.innerHTML = present
    .map(([, c, label, v]) =>
      `<li><span class="sw" style="background:${c}"></span>
        <span class="nm">${label}</span><span class="ms">${v.toFixed(2)}ms</span></li>`)
    .join("");

  el.totalMs.textContent = totalMs.toFixed(2);
  el.hud.classList.toggle("under", totalMs < BUDGET_MS);
  document.querySelector(".budget").style.left = `${(BUDGET_MS / scaleMax) * 100}%`;
}

/* ── answer rendering ───────────────────────────────── */
function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderAnswer(d) {
  el.answerBox.classList.remove("empty");
  const badges = [];

  if (d.blocked) {
    badges.push(`<span class="badge blocked">blocked · ${escapeHTML(d.block_category || "")}</span>`);
    badges.push(`<span class="badge">${escapeHTML(d.block_layer || "")}</span>`);
  } else if (d.abstained) {
    badges.push(`<span class="badge mode">abstained</span>`);
    if (d.abstain_reason) badges.push(`<span class="badge">${escapeHTML(d.abstain_reason)}</span>`);
  } else {
    badges.push(`<span class="badge mode">${escapeHTML(d.mode)}</span>`);
    badges.push(`<span class="badge">conf ${(d.confidence ?? 0).toFixed(3)}</span>`);
  }
  if (d.cached) badges.push(`<span class="badge">cache · ${escapeHTML(d.cached)}</span>`);

  const abstainCls = (d.abstained || d.blocked) ? " abstain" : "";
  const hint = (d.abstained && !d.blocked)
    ? `<p class="abstain-hint">Nothing in this corpus scored above the grounding threshold.
       ${d.passages?.length ? "The closest passages are shown below." : ""}</p>`
    : "";

  el.answerBox.innerHTML =
    `<div class="${abstainCls.trim()}">
       <p class="answer-text" id="answerText">${escapeHTML(d.answer)}</p>
       ${hint}
       <div class="badges">${badges.join("")}</div>
     </div>`;

  renderSources(d);
}

function renderSources(d) {
  const ps = d.passages || [];
  if (!ps.length) { el.sources.hidden = true; return; }
  const cited = new Set((d.citations || []).map((c) => c.text));

  el.sources.hidden = false;
  el.sources.innerHTML =
    `<h3>Sources</h3>` +
    ps.map((p, i) => {
      let text = escapeHTML(p.text);
      // Highlight the exact span the answer was drawn from.
      for (const c of cited) {
        if (c && p.text.includes(c)) {
          text = text.replace(escapeHTML(c), `<mark>${escapeHTML(c)}</mark>`);
          break;
        }
      }
      return `<div class="src">
        <div class="src-head">
          <span class="src-n">[${i + 1}]</span>
          <span class="src-lang${p.cross_lingual ? " src-xl" : ""}">${escapeHTML(p.lang)}${p.cross_lingual ? " · cross-lingual" : ""}</span>
          <span>cos ${(p.cosine ?? 0).toFixed(3)}</span>
        </div>
        <p class="src-text">${text}</p>
      </div>`;
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
      badges.insertAdjacentHTML("afterbegin",
        `<span class="badge refined">refined · ${Math.round(payload.total_ms)}ms</span>`);
    }
  }, 220);
}

/* ── ask ────────────────────────────────────────────── */
let inFlight = false;

async function ask(query) {
  if (!query.trim() || inFlight) return;
  inFlight = true;
  el.askBtn.disabled = true;
  el.answerBox.classList.remove("empty");
  el.answerBox.innerHTML = `<p class="placeholder">searching…</p>`;

  const body = {
    query,
    k: Number(el.topk.value) || 3,
    cross_lingual: el.xling.checked,
    use_cache: el.cache.checked,
    mode: el.mode.value,
  };

  try {
    if (el.mode.value === "quality") {
      await askStream(body);
    } else {
      const r = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      renderAnswer(d);
      renderHUD(d.timings, d.total_ms);
    }
  } catch (e) {
    el.answerBox.innerHTML =
      `<p class="answer-text">Could not reach the backend.</p>
       <p class="abstain-hint">${escapeHTML(e.message)}</p>`;
    el.statusDot.className = "dot bad";
    el.statusText.textContent = "backend unreachable";
  } finally {
    inFlight = false;
    el.askBtn.disabled = false;
  }
}

async function askStream(body) {
  const r = await fetch(`${API}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);

  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let event = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";

    for (const block of parts) {
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) {
          const raw = line.slice(5).trim();
          if (!raw) continue;
          let d; try { d = JSON.parse(raw); } catch { continue; }
          if (event === "fast") { renderAnswer(d); renderHUD(d.timings, d.total_ms); }
          else if (event === "refined") showRefined(d);
        }
      }
    }
  }
}

/* ── sample chips ───────────────────────────────────── */
const FALLBACK_CHIPS = [
  { query: "हिरलूम टमाटर का क्या अर्थ है", lang: "hi" },
  { query: "what is a corporation", lang: "en" },
  { query: "what is my bank balance", lang: "en" },
];

async function loadChips() {
  let items = FALLBACK_CHIPS;
  try {
    const r = await fetch(`${API}/sample-queries`);
    if (r.ok) {
      const d = await r.json();
      if (d.queries?.length) items = d.queries.slice(0, 5).concat(FALLBACK_CHIPS[2]);
    }
  } catch { /* fall back to the built-ins */ }

  el.chips.innerHTML = items
    .map((q) => `<button class="chip" type="button" data-q="${escapeHTML(q.query)}">${escapeHTML(q.query)}</button>`)
    .join("");
  el.chips.querySelectorAll(".chip").forEach((b) =>
    b.addEventListener("click", () => { el.q.value = b.dataset.q; ask(b.dataset.q); }));
}

/* ── mic + waveform ─────────────────────────────────── */
let media = null, audioCtx = null, analyser = null, rafId = null, recorder = null, chunks = [];

function drawWave() {
  const c = el.wave, ctx = c.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== c.clientWidth * dpr) {
    c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
  }
  const buf = new Uint8Array(analyser.frequencyBinCount);

  const loop = () => {
    analyser.getByteTimeDomainData(buf);
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.lineWidth = 1.6 * dpr;
    ctx.strokeStyle = "#e8a33d";
    ctx.beginPath();
    const slice = c.width / buf.length;
    for (let i = 0; i < buf.length; i++) {
      const y = (buf[i] / 128.0) * (c.height / 2);
      i ? ctx.lineTo(i * slice, y) : ctx.moveTo(0, y);
    }
    ctx.stroke();
    rafId = requestAnimationFrame(loop);
  };
  loop();
}

async function startRec() {
  try {
    media = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    el.micHint.textContent = "mic blocked — use the text box below";
    el.micBtn.disabled = true;
    return;
  }
  audioCtx = new AudioContext();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  audioCtx.createMediaStreamSource(media).connect(analyser);

  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) drawWave();

  chunks = [];
  recorder = new MediaRecorder(media);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = onRecStop;
  recorder.start();

  el.micBtn.classList.add("rec");
  el.micHint.textContent = "listening…";
}

async function onRecStop() {
  el.micBtn.classList.remove("rec");
  el.micHint.textContent = "transcribing…";
  cancelAnimationFrame(rafId);
  media?.getTracks().forEach((t) => t.stop());
  audioCtx?.close();

  const blob = new Blob(chunks, { type: "audio/webm" });
  if (blob.size < 1200) { el.micHint.textContent = "too short — hold longer"; return; }

  const fd = new FormData();
  fd.append("audio", blob, "clip.webm");
  try {
    const r = await fetch(`${API}/stt`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    if (d.text) {
      el.transcript.hidden = false;
      el.transcript.textContent = `“${d.text}”`;
      el.q.value = d.text;
      el.micHint.textContent = `hold space or press the mic`;
      ask(d.text);
    } else {
      el.micHint.textContent = "no speech detected";
    }
  } catch (e) {
    el.micHint.textContent = "transcription unavailable — type instead";
  }
}

function stopRec() {
  if (recorder?.state === "recording") recorder.stop();
}

/* ── wiring ─────────────────────────────────────────── */
el.form.addEventListener("submit", (e) => { e.preventDefault(); ask(el.q.value); });

el.micBtn.addEventListener("pointerdown", (e) => { e.preventDefault(); startRec(); });
el.micBtn.addEventListener("pointerup", stopRec);
el.micBtn.addEventListener("pointerleave", stopRec);

let spaceHeld = false;
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !spaceHeld && document.activeElement !== el.q) {
    e.preventDefault(); spaceHeld = true; startRec();
  }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && spaceHeld) { spaceHeld = false; stopRec(); }
});

checkHealth();
loadChips();
setInterval(checkHealth, 60000);
