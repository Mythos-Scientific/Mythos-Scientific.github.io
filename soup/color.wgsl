// Per-cell genome-identity color. Same k-mer LSH signature as before (similar genomes -> similar color), but
// mapped through a smooth cosine palette (moderate saturation, warm<->cool) and temporally smoothed (EMA)
// so it eases between frames instead of zapping. colorState holds the previous smoothed color per cell.
struct P { n: u32, gw: u32, gh: u32, pad: u32, };
@group(0) @binding(0) var<storage, read> cells: array<u32>;
@group(0) @binding(1) var<uniform> u: P;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read_write> colorState: array<u32>;
const STRIDE: u32 = 129u;

fn sgn(h: u32) -> f32 { return select(-1.0, 1.0, ((h >> 31u) & 1u) == 1u); }
fn palette(t: f32) -> vec3<f32> {   // Inigo-Quilez cosine palette — smooth, harmonious, not oversaturated
  return vec3<f32>(0.52, 0.50, 0.56) + vec3<f32>(0.36, 0.36, 0.34) * cos(6.28318 * (t + vec3<f32>(0.0, 0.15, 0.30)));
}
fn packc(c: vec3<f32>) -> u32 {
  return u32(clamp(c.x, 0.0, 1.0) * 255.0) | (u32(clamp(c.y, 0.0, 1.0) * 255.0) << 8u) | (u32(clamp(c.z, 0.0, 1.0) * 255.0) << 16u);
}
fn unpackc(v: u32) -> vec3<f32> { return vec3<f32>(f32(v & 0xffu), f32((v >> 8u) & 0xffu), f32((v >> 16u) & 0xffu)) / 255.0; }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.gw || g.y >= u.gh) { return; }
  let cell = g.y * u.gw + g.x;
  let base = cell * STRIDE;
  var acc = vec3<f32>(0.0, 0.0, 0.0);
  for (var i: u32 = 0u; i < 125u; i = i + 1u) {
    let km = ((cells[base + i] & 0xffu) << 24u) | ((cells[base + i + 1u] & 0xffu) << 16u)
           | ((cells[base + i + 2u] & 0xffu) << 8u) | (cells[base + i + 3u] & 0xffu);
    acc.x = acc.x + sgn(km * 2654435761u);
    acc.y = acc.y + sgn(km * 40503u);
    acc.z = acc.z + sgn(km * 2246822519u);
  }
  acc = acc / sqrt(125.0);
  let t = 0.5 + 0.5 * tanh(acc.x * 1.4);                          // primary axis -> palette position (hue)
  let bright = 0.80 + 0.20 * (0.5 + 0.5 * tanh(acc.y * 1.4));     // subtle brightness variation for identity
  let hueShift = 0.09 * tanh(acc.z * 1.4);
  let tgt = palette(t + hueShift) * bright;
  let prev = unpackc(colorState[cell]);
  let sm = mix(prev, tgt, 0.14);                                 // temporal EMA — smooth, non-zappy transitions
  colorState[cell] = packc(sm);
  textureStore(outTex, vec2<i32>(i32(g.x), i32(g.y)), vec4<f32>(sm, 1.0));
}
