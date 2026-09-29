// A GPU box calls this when its job ends (or fails) to destroy itself, so the Vast key never
// goes on the box. Cloudflare Pages Function. POST /destroy {"instance_id": N} with header X-Kill-Token.
// Only instances labelled openjevx-*-DESTROY-AFTER can be destroyed.
const VAST = "https://console.vast.ai/api/v0/instances/";

function same(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

export async function onRequestPost({ request: req, env }) {
  {
    if (!same(req.headers.get("X-Kill-Token") || "", env.KILL_TOKEN)) return new Response("forbidden", { status: 403 });
    let id;
    try { id = Number((await req.json()).instance_id); } catch { return new Response("bad json", { status: 400 }); }
    if (!Number.isInteger(id) || id <= 0) return new Response("bad instance_id", { status: 400 });
    const auth = { Authorization: "Bearer " + env.VAST_API_KEY };
    const info = await fetch(VAST + id + "/", { headers: auth });
    if (info.status === 404) return Response.json({ id, destroyed: false, reason: "not found" });
    if (!info.ok) return Response.json({ id, destroyed: false, reason: "vast " + info.status }, { status: 502 });
    const inst = (await info.json()).instances;
    if (!inst) return Response.json({ id, destroyed: false, reason: "not found" });
    const label = inst.label || "";
    if (!/^openjevx-.*-DESTROY-AFTER$/.test(label)) return Response.json({ id, destroyed: false, reason: "label" }, { status: 403 });
    const del = await fetch(VAST + id + "/", { method: "DELETE", headers: auth });
    return Response.json({ id, label, destroyed: del.ok, status: del.status });
  }
}
