// openjevx.json on install and upgrade: the user's values always win; only keys the user does not have yet are added.
export const defaults = { listen: "127.0.0.1:21118", device: "auto" };

// mergeConfig returns the text to write, or null when the existing file needs no change (it is never rewritten then).
// A file that is not a JSON object is left alone: better a config the server rejects loudly than a silent overwrite.
export function mergeConfig(existing, wanted = defaults) {
  if (existing == null) return JSON.stringify(wanted, null, 2) + "\n";
  let cfg;
  try { cfg = JSON.parse(existing); } catch { return null; }
  if (cfg === null || typeof cfg !== "object" || Array.isArray(cfg)) return null;
  const missing = Object.keys(wanted).filter((k) => !(k in cfg));
  if (missing.length === 0) return null;
  for (const k of missing) cfg[k] = wanted[k];
  return JSON.stringify(cfg, null, 2) + "\n";
}
