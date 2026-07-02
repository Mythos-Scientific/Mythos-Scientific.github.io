// WebGPU backend for the MetaNCA MLP rule. Implements the backend interface used
// by rule_core.updateStep: attn(dir, focus, neigh, qpos, kpos) and head(percep).
// The ELU linear attention is 3 compute kernels; the head is a chain of matmuls.
// Validated against the CPU oracle (rule_cpu.js) in Deno; same code runs in-browser.
import { STATE } from "./rule_core.js";

const HEADS = 3, HEAD_DIM = 10, HD30 = HEADS * HEAD_DIM; // 30
const WG = 64;

function concatF32(list) {
  const total = list.reduce((a, x) => a + x.length, 0);
  const out = new Float32Array(total);
  let o = 0;
  for (const x of list) { out.set(x, o); o += x.length; }
  return out;
}

export async function createDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  return await adapter.requestDevice();
}

const COMMON = `
const HEADS = 3u; const HEAD_DIM = 10u; const HD30 = 30u; const STATE = 31u;
fn fmap(x: f32) -> f32 { return select(exp(x), x + 1.0, x > 0.0); }
fn eluf(x: f32) -> f32 { return select(exp(x) - 1.0, x, x > 0.0); }
`;

// K1: project keys -> kphi (feature map + rope) and vv. One thread per (b,l).
const WGSL_KPROJ = COMMON + `
struct P { B:u32, Lq:u32, Lk:u32, pad:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> neigh: array<f32>;   // B*Lk*31
@group(0) @binding(2) var<storage, read> kpos: array<f32>;    // B*Lk*30
@group(0) @binding(3) var<storage, read> Wk: array<f32>;      // 31*30
@group(0) @binding(4) var<storage, read> Wv: array<f32>;
@group(0) @binding(5) var<storage, read_write> kphi: array<f32>; // B*Lk*30
@group(0) @binding(6) var<storage, read_write> vv: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.x + gid.y * nwg.x * ${WG}u; if (idx >= p.B * p.Lk) { return; }
  let nbase = idx * STATE; let pbase = idx * HD30;
  for (var h = 0u; h < HEADS; h = h + 1u) {
    var tmp: array<f32, 10>;
    for (var k = 0u; k < HEAD_DIM; k = k + 1u) {
      let woff = h * HEAD_DIM + k; var sk = 0.0; var sv = 0.0;
      for (var d = 0u; d < STATE; d = d + 1u) {
        let nvv = neigh[nbase + d];
        sk = sk + nvv * Wk[d * HD30 + woff];
        sv = sv + nvv * Wv[d * HD30 + woff];
      }
      tmp[k] = fmap(sk);
      vv[idx * HD30 + h * HEAD_DIM + k] = sv;
    }
    for (var m = 0u; m < 5u; m = m + 1u) {
      let s = kpos[pbase + h * HEAD_DIM + 2u * m];
      let c = kpos[pbase + h * HEAD_DIM + 2u * m + 1u];
      let xe = tmp[2u * m]; let xo = tmp[2u * m + 1u];
      kphi[idx * HD30 + h * HEAD_DIM + 2u * m] = xe * c - xo * s;
      kphi[idx * HD30 + h * HEAD_DIM + 2u * m + 1u] = xe * s + xo * c;
    }
  }
}`;

// K2: reduce -> kv (B*300) and ksum (B*30). One thread per (b, hk) with hk in 0..30.
const WGSL_KREDUCE = COMMON + `
struct P { B:u32, Lq:u32, Lk:u32, pad:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> kphi: array<f32>;
@group(0) @binding(2) var<storage, read> vv: array<f32>;
@group(0) @binding(3) var<storage, read_write> kv: array<f32>;    // B*300
@group(0) @binding(4) var<storage, read_write> ksum: array<f32>;  // B*30
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.x + gid.y * nwg.x * ${WG}u; if (idx >= p.B * HD30) { return; }
  let b = idx / HD30; let hk = idx % HD30; let h = hk / HEAD_DIM;
  var s = 0.0; var acc: array<f32, 10>;
  for (var m = 0u; m < HEAD_DIM; m = m + 1u) { acc[m] = 0.0; }
  for (var l = 0u; l < p.Lk; l = l + 1u) {
    let kp = kphi[(b * p.Lk + l) * HD30 + hk];
    s = s + kp;
    let voff = (b * p.Lk + l) * HD30 + h * HEAD_DIM;
    for (var m = 0u; m < HEAD_DIM; m = m + 1u) { acc[m] = acc[m] + kp * vv[voff + m]; }
  }
  ksum[idx] = s;
  for (var m = 0u; m < HEAD_DIM; m = m + 1u) { kv[(b * HD30 + hk) * HEAD_DIM + m] = acc[m]; }
}`;

// K3: apply -> out (B*Lq*31). One thread per (b,lq). Attention weights packed
// into one storage buffer (offsets below) to stay within 8 storage buffers/stage.
// pack layout (floats): Wq[930] Wout[930] g1[31] b1[31] g2[31] b2[31]
//   Wff1[3100] bff1[100] Wff2[3100] bff2[31]   (ff biases added: main ffn_use_bias=True)
const WGSL_KAPPLY = COMMON + `
struct P { B:u32, Lq:u32, Lk:u32, pad:u32 };
const O_WOUT = 930u; const O_G1 = 1860u; const O_B1 = 1891u; const O_G2 = 1922u;
const O_B2 = 1953u; const O_FF1 = 1984u; const O_BFF1 = 5084u; const O_FF2 = 5184u; const O_BFF2 = 8284u;
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> focus: array<f32>;  // B*Lq*31
@group(0) @binding(2) var<storage, read> qpos: array<f32>;   // B*Lq*30
@group(0) @binding(3) var<storage, read> kv: array<f32>;     // B*300
@group(0) @binding(4) var<storage, read> ksum: array<f32>;   // B*30
@group(0) @binding(5) var<storage, read> w: array<f32>;      // packed weights
@group(0) @binding(6) var<storage, read_write> outp: array<f32>; // B*Lq*31
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.x + gid.y * nwg.x * ${WG}u; if (idx >= p.B * p.Lq) { return; }
  let b = idx / p.Lq;
  let fbase = idx * STATE; let pbase = idx * HD30;
  var qphi: array<f32, 30>;
  for (var h = 0u; h < HEADS; h = h + 1u) {
    var tmp: array<f32, 10>;
    for (var k = 0u; k < HEAD_DIM; k = k + 1u) {
      let woff = h * HEAD_DIM + k; var sq = 0.0;
      for (var d = 0u; d < STATE; d = d + 1u) { sq = sq + focus[fbase + d] * w[d * HD30 + woff]; }
      tmp[k] = fmap(sq);
    }
    for (var m = 0u; m < 5u; m = m + 1u) {
      let s = qpos[pbase + h * HEAD_DIM + 2u * m];
      let c = qpos[pbase + h * HEAD_DIM + 2u * m + 1u];
      let xe = tmp[2u * m]; let xo = tmp[2u * m + 1u];
      qphi[h * HEAD_DIM + 2u * m] = xe * c - xo * s;
      qphi[h * HEAD_DIM + 2u * m + 1u] = xe * s + xo * c;
    }
  }
  var attnflat: array<f32, 30>;
  for (var h = 0u; h < HEADS; h = h + 1u) {
    var den = 1e-6;
    for (var k = 0u; k < HEAD_DIM; k = k + 1u) { den = den + qphi[h * HEAD_DIM + k] * ksum[b * HD30 + h * HEAD_DIM + k]; }
    for (var m = 0u; m < HEAD_DIM; m = m + 1u) {
      var num = 0.0;
      for (var k = 0u; k < HEAD_DIM; k = k + 1u) { num = num + qphi[h * HEAD_DIM + k] * kv[(b * HD30 + h * HEAD_DIM + k) * HEAD_DIM + m]; }
      attnflat[h * HEAD_DIM + m] = num / den;
    }
  }
  var o: array<f32, 31>;
  for (var d = 0u; d < STATE; d = d + 1u) {
    var s = 0.0;
    for (var e = 0u; e < HD30; e = e + 1u) { s = s + attnflat[e] * w[O_WOUT + e * STATE + d]; }
    o[d] = s + focus[fbase + d];
  }
  var mean = 0.0; for (var d = 0u; d < STATE; d = d + 1u) { mean = mean + o[d]; } mean = mean / 31.0;
  var vv2 = 0.0; for (var d = 0u; d < STATE; d = d + 1u) { let t = o[d] - mean; vv2 = vv2 + t * t; } vv2 = vv2 / 31.0;
  var nr: array<f32, 31>; let inv1 = 1.0 / sqrt(vv2 + 1e-6);
  for (var d = 0u; d < STATE; d = d + 1u) { nr[d] = w[O_G1 + d] * (o[d] - mean) * inv1 + w[O_B1 + d]; }
  var ff: array<f32, 31>;
  var hid: array<f32, 100>;
  for (var hh = 0u; hh < 100u; hh = hh + 1u) {
    var s = w[O_BFF1 + hh]; for (var d = 0u; d < STATE; d = d + 1u) { s = s + nr[d] * w[O_FF1 + d * 100u + hh]; }
    hid[hh] = max(s, 0.0);
  }
  for (var d = 0u; d < STATE; d = d + 1u) {
    var s = w[O_BFF2 + d]; for (var hh = 0u; hh < 100u; hh = hh + 1u) { s = s + hid[hh] * w[O_FF2 + hh * STATE + d]; }
    ff[d] = s + nr[d];
  }
  var mean2 = 0.0; for (var d = 0u; d < STATE; d = d + 1u) { mean2 = mean2 + ff[d]; } mean2 = mean2 / 31.0;
  var vr2 = 0.0; for (var d = 0u; d < STATE; d = d + 1u) { let t = ff[d] - mean2; vr2 = vr2 + t * t; } vr2 = vr2 / 31.0;
  let inv2 = 1.0 / sqrt(vr2 + 1e-6);
  for (var d = 0u; d < STATE; d = d + 1u) { outp[fbase + d] = w[O_G2 + d] * (ff[d] - mean2) * inv2 + w[O_B2 + d]; }
}`;

// Head matmul layer: out[r,o] = act(bias[o] + sum_d in[r,d]*K[d,o]). One thread per (r,o).
const WGSL_DENSE = COMMON + `
struct P { N:u32, inD:u32, outD:u32, act:u32 };  // act: 0=elu,1=tanh,2=none
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> ker: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.x + gid.y * nwg.x * ${WG}u; if (idx >= p.N * p.outD) { return; }
  let r = idx / p.outD; let o = idx % p.outD;
  var s = bias[o];
  for (var d = 0u; d < p.inD; d = d + 1u) { s = s + inp[r * p.inD + d] * ker[d * p.outD + o]; }
  if (p.act == 0u) { s = eluf(s); } else if (p.act == 1u) { s = tanh(s); }
  outp[idx] = s;
}`;

export const _shaders = { KPROJ: WGSL_KPROJ, KREDUCE: WGSL_KREDUCE, KAPPLY: WGSL_KAPPLY, DENSE: WGSL_DENSE };
export const WGSIZE = WG;
export { WGSL_KPROJ, WGSL_KREDUCE, WGSL_KAPPLY, WGSL_DENSE, concatF32 };
export function makePipelineExt(device, code) { return makePipeline(device, code); }

function makePipeline(device, code) {
  return device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
}

export class GPURule {
  constructor(device, W) {
    this.device = device;
    this.kproj = makePipeline(device, WGSL_KPROJ);
    this.kreduce = makePipeline(device, WGSL_KREDUCE);
    this.kapply = makePipeline(device, WGSL_KAPPLY);
    this.dense = makePipeline(device, WGSL_DENSE);
    const sb = (arr, lbl) => this._storage(arr, lbl);
    // attention weights per direction
    this.attnW = {};
    for (const [dir, pre] of [["fwd", "attn_forward"], ["bwd", "attn_backward"]]) {
      // K1 needs Wk, Wv separately; K3 weights packed into one buffer.
      const packed = concatF32([
        W[`${pre}.query.kernel`].data, W[`${pre}.out.kernel`].data,
        W[`${pre}.gamma_1`].data, W[`${pre}.beta_1`].data,
        W[`${pre}.gamma_2`].data, W[`${pre}.beta_2`].data,
        W[`${pre}.ff_1.kernel`].data, W[`${pre}.ff_1.bias`].data,
        W[`${pre}.ff_2.kernel`].data, W[`${pre}.ff_2.bias`].data,
      ]);
      this.attnW[dir] = {
        Wk: sb(W[`${pre}.key.kernel`].data, `${dir}.Wk`), Wv: sb(W[`${pre}.value.kernel`].data, `${dir}.Wv`),
        pack: sb(packed, `${dir}.pack`),
      };
    }
    // head layers
    this.headL = [];
    let i = 0;
    while (W[`local_rule_head.Dense_${i}.kernel`]) {
      const k = W[`local_rule_head.Dense_${i}.kernel`], b = W[`local_rule_head.Dense_${i}.bias`];
      this.headL.push({ k: sb(k.data, `head${i}.k`), b: sb(b.data, `head${i}.b`), inD: k.shape[0], outD: k.shape[1] });
      i++;
    }
  }

  _storage(f32, label = "stor") {
    // parseBundle returns subarray views into one big ArrayBuffer; copy to a
    // contiguous array so writeBuffer sizes the copy correctly.
    const src = (f32.byteOffset === 0 && f32.byteLength === f32.buffer.byteLength) ? f32 : f32.slice();
    const buf = this.device.createBuffer({ label, size: Math.max(4, src.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(buf, 0, src);
    return buf;
  }
  _empty(nFloats, label = "tmp") {
    return this.device.createBuffer({ label, size: Math.max(4, nFloats * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  }
  _uniform(ints, label = "uni") {
    const buf = this.device.createBuffer({ label, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, new Uint32Array(ints));
    return buf;
  }
  _bind(pipeline, buffers) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
  }
  _pass(enc, pipeline, buffers, threads) {
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this._bind(pipeline, buffers));
    const wgTotal = Math.ceil(threads / WG);
    const wx = Math.min(wgTotal, 65535);
    const wy = Math.ceil(wgTotal / 65535); // 2D grid: idx = gid.y*nwg.x*WG + gid.x
    pass.dispatchWorkgroups(wx, wy);
    pass.end();
  }
  async _read(buf, nFloats) {
    const staging = this.device.createBuffer({ size: nFloats * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, nFloats * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    return out;
  }

  async attn(dir, focus, neigh, qpos, kpos) {
    const dev = this.device, w = this.attnW[dir];
    const B = focus.shape[0], Lq = focus.shape[1], Lk = neigh.shape[1];
    const uni = this._uniform([B, Lq, Lk, 0]);
    const focusB = this._storage(focus.data), neighB = this._storage(neigh.data);
    const qposB = this._storage(qpos.data), kposB = this._storage(kpos.data);
    const kphi = this._empty(B * Lk * HD30), vv = this._empty(B * Lk * HD30);
    const kv = this._empty(B * HD30 * HEAD_DIM), ksum = this._empty(B * HD30);
    const outB = this._empty(B * Lq * STATE);
    const enc = dev.createCommandEncoder();
    this._pass(enc, this.kproj, [uni, neighB, kposB, w.Wk, w.Wv, kphi, vv], B * Lk);
    this._pass(enc, this.kreduce, [uni, kphi, vv, kv, ksum], B * HD30);
    this._pass(enc, this.kapply, [uni, focusB, qposB, kv, ksum, w.pack, outB], B * Lq);
    dev.queue.submit([enc.finish()]);
    const out = await this._read(outB, B * Lq * STATE);
    [focusB, neighB, qposB, kposB, kphi, vv, kv, ksum, outB, uni].forEach((b) => b.destroy());
    return { data: out, shape: [B, Lq, STATE] };
  }

  async head(percep) {
    const dev = this.device;
    const N = percep.shape[0];
    let curBuf = this._storage(percep.data), curD = percep.shape[1];
    const temps = [curBuf];
    for (let li = 0; li < this.headL.length; li++) {
      const L = this.headL[li];
      const act = li < this.headL.length - 1 ? 0 : 1; // elu hidden, tanh final
      const uni = this._uniform([N, L.inD, L.outD, act]);
      const outB = this._empty(N * L.outD);
      const enc = dev.createCommandEncoder();
      this._pass(enc, this.dense, [uni, curBuf, L.k, L.b, outB], N * L.outD);
      dev.queue.submit([enc.finish()]);
      curBuf = outB; curD = L.outD; temps.push(outB); temps.push(uni);
    }
    const out = await this._read(curBuf, N * curD);
    temps.forEach((b) => b.destroy());
    return { data: out, shape: [N, curD] };
  }
}
