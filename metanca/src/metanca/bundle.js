// Read the tiny binary bundle format (see blog/export/_bundle.py).
// parseBundle is environment-agnostic; loaders below handle Node vs browser.

export function parseBundle(manifest, floatArray) {
  const out = {};
  for (const [name, meta] of Object.entries(manifest.tensors)) {
    const shape = meta.shape;
    const n = shape.length ? shape.reduce((a, b) => a * b, 1) : 1;
    out[name] = { data: floatArray.subarray(meta.offset, meta.offset + n), shape };
  }
  return out;
}

// Browser: fetch <prefix>.json and <prefix>.bin
export async function loadBundle(prefix) {
  const manifest = await (await fetch(prefix + ".json")).json();
  const buf = await (await fetch(prefix + ".bin")).arrayBuffer();
  return parseBundle(manifest, new Float32Array(buf));
}
