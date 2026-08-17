/* Edge proxy: api.sonus.spacesdrive.cc -> Railway origin.
 *
 * Railway issues custom-domain certificates only after its own ownership
 * validation completes, which left the branded hostname serving Railway's
 * wildcard cert. Fronting the origin with a Worker gives us a Cloudflare-issued
 * certificate for the branded name immediately and keeps the origin hostname an
 * implementation detail we can change without touching DNS.
 */

const ORIGIN = "https://vaani-api-production.up.railway.app";

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const target = new URL(url.pathname + url.search, ORIGIN);

    // Preflight is answered at the edge so it never costs an origin round trip.
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    const res = await fetch(target, {
      method: request.method,
      headers: stripHopByHop(request.headers),
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      redirect: "follow",
    });

    const headers = new Headers(res.headers);
    for (const [k, v] of Object.entries(corsHeaders())) headers.set(k, v);
    // SSE must not be buffered by any intermediary.
    if ((headers.get("content-type") || "").includes("text/event-stream")) {
      headers.set("cache-control", "no-cache");
      headers.set("x-accel-buffering", "no");
    }
    return new Response(res.body, { status: res.status, headers });
  },
};

function corsHeaders() {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type",
    "access-control-max-age": "86400",
  };
}

function stripHopByHop(h) {
  const out = new Headers(h);
  for (const k of ["host", "cf-connecting-ip", "cf-ray", "cf-visitor", "x-forwarded-host"]) {
    out.delete(k);
  }
  return out;
}
