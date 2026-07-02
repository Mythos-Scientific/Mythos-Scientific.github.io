// All-GPU MetaNCA engine: task-net state (weights, hidden, posenc) lives on the GPU
// across steps. Each step runs entirely on-GPU (gather -> attention -> perception ->
// head -> scatter -> apply) as ONE no-readback submission. Weights are read back only
// when the UI needs them (viz/accuracy). Reuses the kernels validated in gpu.js.
//
// Synchronous-update semantics: deltas accumulate into deltaW/deltaHs (each weight
// written by exactly its own param's fwd-focus scatter -> no atomics), then applied
// after all params, so every gather reads the previous step's state.
import { H } from "./posenc.js";
import { STATE, neighborSpec } from "./rule_core.js";
import { addFeat, groupByAxis, concatAxis1 } from "./nd.js";
import { _shaders, WGSIZE as WG, makePipelineExt as makePipeline, concatF32 } from "./gpu.js";

const HD30 = 30;

// gather: materialize vec[count*31] + pos[count*30] from global W/Hs/Pe via u32 idx map.
const WGSL_GATHER = `
struct P { count:u32, a:u32, b:u32, c:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Hs: array<f32>;
@group(0) @binding(3) var<storage, read> Pe: array<f32>;
@group(0) @binding(4) var<storage, read> idx: array<u32>;
@group(0) @binding(5) var<storage, read_write> vecOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> posOut: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * ${WG}u; if (i >= p.count) { return; }
  let g = idx[i];
  vecOut[i * 31u + 0u] = W[g];
  for (var d = 0u; d < 30u; d = d + 1u) {
    vecOut[i * 31u + 1u + d] = Hs[g * 30u + d];
    posOut[i * 30u + d] = Pe[g * 30u + d];
  }
}`;

// perception: percep[cell*93] = [focusVec, aggF/countF, aggB[clip]/countB]
const WGSL_PERCEP = `
struct P { count:u32, a:u32, b:u32, c:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> focusVec: array<f32>;
@group(0) @binding(2) var<storage, read> aggF: array<f32>;
@group(0) @binding(3) var<storage, read> aggB: array<f32>;
@group(0) @binding(4) var<storage, read> bwdCell: array<u32>;  // fwd cell -> bwd-frame cell
@group(0) @binding(5) var<storage, read_write> percep: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let cell = gid.x + gid.y * nwg.x * ${WG}u; if (cell >= p.count) { return; }
  let aB = bwdCell[cell] * 31u;          // agg_bwd read at the same weight's bwd-frame position
  for (var d = 0u; d < 31u; d = d + 1u) {  // NO count division (main)
    percep[cell * 93u + d] = focusVec[cell * 31u + d];
    percep[cell * 93u + 31u + d] = aggF[cell * 31u + d];
    percep[cell * 93u + 62u + d] = aggB[aB + d];
  }
}`;

// scatter: deltaW[idx[cell]] += delta[cell,0]; deltaHs[idx[cell]*30+d] += delta[cell,1+d]
const WGSL_SCATTER = `
struct P { count:u32, a:u32, b:u32, c:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read> delta: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;   // per-cell 0/1 (prop_cells_updated)
@group(0) @binding(4) var<storage, read_write> deltaW: array<f32>;
@group(0) @binding(5) var<storage, read_write> deltaHs: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let cell = gid.x + gid.y * nwg.x * ${WG}u; if (cell >= p.count) { return; }
  let m = mask[cell];                       // only update the masked fraction this step
  let g = idx[cell];
  deltaW[g] = deltaW[g] + m * delta[cell * 31u + 0u];
  for (var d = 0u; d < 30u; d = d + 1u) { deltaHs[g * 30u + d] = deltaHs[g * 30u + d] + m * delta[cell * 31u + 1u + d]; }
}`;

// apply-and-zero: W += deltaW; deltaW = 0 (same kernel for Hs with its own buffers)
const WGSL_APPLY = `
struct P { count:u32, a:u32, b:u32, c:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> delta: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * ${WG}u; if (i >= p.count) { return; }
  dst[i] = dst[i] + delta[i]; delta[i] = 0.0;
}`;

const O_PACK = [0, 930, 1860, 1891, 1922, 1953, 1984, 5084]; // for reference

export class GPUEngine {
  constructor(device, ruleBundle) {
    this.device = device;
    this.prop = 1.0;   // prop_cells_updated; demo sets 0.8 (matches training); 1.0 = all cells (validation)
    this.pipe = {
      kproj: makePipeline(device, _shaders.KPROJ), kreduce: makePipeline(device, _shaders.KREDUCE),
      kapply: makePipeline(device, _shaders.KAPPLY), dense: makePipeline(device, _shaders.DENSE),
      gather: makePipeline(device, WGSL_GATHER), percep: makePipeline(device, WGSL_PERCEP),
      scatter: makePipeline(device, WGSL_SCATTER), apply: makePipeline(device, WGSL_APPLY),
    };
    const W = ruleBundle;
    const sb = (f32) => this._storage(f32);
    this.attnW = {};
    for (const [dir, pre] of [["fwd", "attn_forward"], ["bwd", "attn_backward"]]) {
      this.attnW[dir] = {
        Wk: sb(W[`${pre}.key.kernel`].data), Wv: sb(W[`${pre}.value.kernel`].data),
        pack: sb(concatF32([W[`${pre}.query.kernel`].data, W[`${pre}.out.kernel`].data,
          W[`${pre}.gamma_1`].data, W[`${pre}.beta_1`].data, W[`${pre}.gamma_2`].data,
          W[`${pre}.beta_2`].data, W[`${pre}.ff_1.kernel`].data, W[`${pre}.ff_1.bias`].data,
          W[`${pre}.ff_2.kernel`].data, W[`${pre}.ff_2.bias`].data])),
      };
    }
    this.headL = [];
    let i = 0;
    while (W[`local_rule_head.Dense_${i}.kernel`]) {
      const k = W[`local_rule_head.Dense_${i}.kernel`], b = W[`local_rule_head.Dense_${i}.bias`];
      this.headL.push({ k: sb(k.data), b: sb(b.data), inD: k.shape[0], outD: k.shape[1] });
      i++;
    }
    this.maxHeadW = Math.max(...this.headL.map((l) => l.outD), 93);
  }

  _storage(f32) {
    const src = (f32.byteOffset === 0 && f32.byteLength === f32.buffer.byteLength) ? f32 : f32.slice();
    const buf = this.device.createBuffer({ size: Math.max(4, src.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(buf, 0, src);
    return buf;
  }
  _u32(arr) {
    const buf = this.device.createBuffer({ size: Math.max(4, arr.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, arr);
    return buf;
  }
  _empty(nFloats) { return this.device.createBuffer({ size: Math.max(4, nFloats * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }); }
  _uni(ints) { const b = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); const a = new Uint32Array(8); a.set(ints); this.device.queue.writeBuffer(b, 0, a); return b; }
  _bg(pipe, bufs) { return this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) }); }

  // grouped global-index array for a param's weights, grouped by indexDim, + offset.
  _idxView(shape, count, off, indexDim) {
    const local = new Float32Array(count);
    for (let k = 0; k < count; k++) local[k] = k;
    const grouped = groupByAxis(addFeat(local, shape), indexDim); // [N, W, 1]
    const N = grouped.shape[0], Wd = grouped.shape[1];
    const u = new Uint32Array(N * Wd);
    for (let t = 0; t < N * Wd; t++) u[t] = (grouped.data[t] | 0) + off;
    return { u, N, W: Wd };
  }

  setArch(arch, params, hidden, posenc) {
    // global layout
    this.arch = arch;
    const names = [];
    for (let L = 0; L < arch.length; L++) { names.push(`Dense_${L}.bias`); names.push(`Dense_${L}.kernel`); }
    this.names = names;
    const off = {}; let tot = 0;
    const shapes = {};
    for (const n of names) { off[n] = tot; tot += params[n].data.length; shapes[n] = params[n].shape; }
    this.totalW = tot; this.off = off; this.shapes = shapes;
    // global W/Hs/Pe
    const Wg = new Float32Array(tot), Hg = new Float32Array(tot * H), Pg = new Float32Array(tot * H);
    for (const n of names) {
      Wg.set(params[n].data, off[n]);
      Hg.set(hidden[n].data, off[n] * H);
      Pg.set(posenc[n].data, off[n] * H);
    }
    this.W = this._storage(Wg); this.Hs = this._storage(Hg); this.Pe = this._storage(Pg);
    this.deltaW = this._empty(tot); this.deltaHs = this._empty(tot * H);
    // zero deltas initially
    this.device.queue.writeBuffer(this.deltaW, 0, new Float32Array(tot));
    this.device.queue.writeBuffer(this.deltaHs, 0, new Float32Array(tot * H));

    // per-param plan
    this.plan = [];
    for (const name of names) {
      const shape = params[name].shape, count = params[name].data.length;
      const specF = neighborSpec(name, arch, "fwd"), specB = neighborSpec(name, arch, "bwd");
      const buildDir = (spec) => {
        const f = this._idxView(shape, count, off[name], spec.fdim);
        const nbrViews = spec.nbrs.map(([n, d]) => this._idxView(params[n].shape, params[n].data.length, off[n], d));
        // concat neighbor idx along axis1 (all share N = f.N)
        const N = f.N, totK = nbrViews.reduce((a, v) => a + v.W, 0);
        const neighU = new Uint32Array(N * totK);
        for (let i = 0; i < N; i++) { let w = 0; for (const v of nbrViews) { for (let k = 0; k < v.W; k++) neighU[i * totK + (w + k)] = v.u[i * v.W + k]; w += v.W; } }
        return { focusU: f.u, N: f.N, W: f.W, neighU, K: totK };
      };
      const F = buildDir(specF), B = buildDir(specB);
      const cells = F.N * F.W;
      // bwdCell[c]: for fwd cell c (weight global F.focusU[c]), the bwd-frame cell index
      // holding the same weight -> reads agg_bwd at the correct (transformed) position.
      const g2bwd = new Map();
      for (let bc = 0; bc < B.focusU.length; bc++) g2bwd.set(B.focusU[bc], bc);
      const bwdCellArr = new Uint32Array(cells);
      for (let c = 0; c < cells; c++) bwdCellArr[c] = g2bwd.get(F.focusU[c]);
      const e = (n) => this._empty(n);
      this.plan.push({
        name, fdim: specF.fdim, cells,
        Nf: F.N, Wf: F.W, Kf: F.K, Nb: B.N, Wb: B.W, Kb: B.K,
        focusIdxF: this._u32(F.focusU), neighIdxF: this._u32(F.neighU),
        focusIdxB: this._u32(B.focusU), neighIdxB: this._u32(B.neighU),
        bwdCell: this._u32(bwdCellArr),
        // scratch
        focusVecF: e(F.N * F.W * STATE), focusPosF: e(F.N * F.W * HD30),
        neighVecF: e(F.N * F.K * STATE), neighPosF: e(F.N * F.K * HD30),
        focusVecB: e(B.N * B.W * STATE), focusPosB: e(B.N * B.W * HD30),
        neighVecB: e(B.N * B.K * STATE), neighPosB: e(B.N * B.K * HD30),
        kphiF: e(F.N * F.K * HD30), vvF: e(F.N * F.K * HD30), kvF: e(F.N * HD30 * 10), ksumF: e(F.N * HD30), aggF: e(F.N * F.W * STATE),
        kphiB: e(B.N * B.K * HD30), vvB: e(B.N * B.K * HD30), kvB: e(B.N * HD30 * 10), ksumB: e(B.N * HD30), aggB: e(B.N * B.W * STATE),
        percep: e(cells * 93), head0: e(cells * this.maxHeadW), head1: e(cells * this.maxHeadW),
        maskBuf: e(cells),   // per-cell stochastic update mask (refilled each step)
      });
    }
    this._compile();
  }

  // record a pass into this.passes (bind group + uniform built once, replayed each step)
  _rec(pipe, bufs, threads) {
    const wt = Math.ceil(threads / WG);
    this.passes.push({ pipe, bg: this._bg(pipe, bufs), wx: Math.min(wt, 65535), wy: Math.ceil(wt / 65535) });
  }
  _recAttn(dir, pl) {
    const w = this.attnW[dir];
    const f = dir === "fwd"
      ? { focusVec: pl.focusVecF, focusPos: pl.focusPosF, neighVec: pl.neighVecF, neighPos: pl.neighPosF, kphi: pl.kphiF, vv: pl.vvF, kv: pl.kvF, ksum: pl.ksumF, agg: pl.aggF, B: pl.Nf, Lq: pl.Wf, Lk: pl.Kf }
      : { focusVec: pl.focusVecB, focusPos: pl.focusPosB, neighVec: pl.neighVecB, neighPos: pl.neighPosB, kphi: pl.kphiB, vv: pl.vvB, kv: pl.kvB, ksum: pl.ksumB, agg: pl.aggB, B: pl.Nb, Lq: pl.Wb, Lk: pl.Kb };
    const uni = this._uni([f.B, f.Lq, f.Lk, 0]);
    this._rec(this.pipe.kproj, [uni, f.neighVec, f.neighPos, w.Wk, w.Wv, f.kphi, f.vv], f.B * f.Lk);
    this._rec(this.pipe.kreduce, [uni, f.kphi, f.vv, f.kv, f.ksum], f.B * HD30);
    this._rec(this.pipe.kapply, [uni, f.focusVec, f.focusPos, f.kv, f.ksum, w.pack, f.agg], f.B * f.Lq);
  }
  _compile() {
    this.passes = [];
    for (const pl of this.plan) {
      this._rec(this.pipe.gather, [this._uni([pl.Nf * pl.Wf]), this.W, this.Hs, this.Pe, pl.focusIdxF, pl.focusVecF, pl.focusPosF], pl.Nf * pl.Wf);
      this._rec(this.pipe.gather, [this._uni([pl.Nf * pl.Kf]), this.W, this.Hs, this.Pe, pl.neighIdxF, pl.neighVecF, pl.neighPosF], pl.Nf * pl.Kf);
      this._rec(this.pipe.gather, [this._uni([pl.Nb * pl.Wb]), this.W, this.Hs, this.Pe, pl.focusIdxB, pl.focusVecB, pl.focusPosB], pl.Nb * pl.Wb);
      this._rec(this.pipe.gather, [this._uni([pl.Nb * pl.Kb]), this.W, this.Hs, this.Pe, pl.neighIdxB, pl.neighVecB, pl.neighPosB], pl.Nb * pl.Kb);
      this._recAttn("fwd", pl);
      this._recAttn("bwd", pl);
      this._rec(this.pipe.percep, [this._uni([pl.cells]), pl.focusVecF, pl.aggF, pl.aggB, pl.bwdCell, pl.percep], pl.cells);
      let cur = pl.percep, inD = 93;
      for (let li = 0; li < this.headL.length; li++) {
        const L = this.headL[li], act = li < this.headL.length - 1 ? 0 : 1;
        const out = (li % 2 === 0) ? pl.head0 : pl.head1;
        this._rec(this.pipe.dense, [this._uni([pl.cells, inD, L.outD, act]), cur, L.k, L.b, out], pl.cells * L.outD);
        cur = out; inD = L.outD;
      }
      this._rec(this.pipe.scatter, [this._uni([pl.cells]), pl.focusIdxF, cur, pl.maskBuf, this.deltaW, this.deltaHs], pl.cells);
    }
    this._rec(this.pipe.apply, [this._uni([this.totalW]), this.W, this.deltaW], this.totalW);
    this._rec(this.pipe.apply, [this._uni([this.totalW * H]), this.Hs, this.deltaHs], this.totalW * H);
  }

  _fillMasks() {
    // fresh stochastic per-cell update mask each step (prop_cells_updated, matches training).
    for (const pl of this.plan) {
      const m = new Float32Array(pl.cells);
      if (this.prop >= 1.0) m.fill(1);
      else for (let i = 0; i < pl.cells; i++) m[i] = Math.random() < this.prop ? 1 : 0;
      this.device.queue.writeBuffer(pl.maskBuf, 0, m);
    }
  }

  step(n = 1) {
    for (let s = 0; s < n; s++) {
      this._fillMasks();                       // new random mask each step
      const enc = this.device.createCommandEncoder();
      for (const p of this.passes) {
        const pass = enc.beginComputePass();
        pass.setPipeline(p.pipe); pass.setBindGroup(0, p.bg);
        pass.dispatchWorkgroups(p.wx, p.wy);
        pass.end();
      }
      this.device.queue.submit([enc.finish()]);
    }
  }

  async readWeights() {
    const staging = this.device.createBuffer({ size: this.totalW * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.W, 0, staging, 0, this.totalW * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    const out = {};
    for (const n of this.names) {
      const cnt = this.shapes[n].reduce((a, b) => a * b, 1);
      out[n] = { data: all.subarray(this.off[n], this.off[n] + cnt), shape: this.shapes[n] };
    }
    return out;
  }

  // Read back per-weight hidden states (same layout as W, H floats per weight).
  // Used to carry state across architecture edits.
  async readHidden() {
    const staging = this.device.createBuffer({ size: this.totalW * H * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.Hs, 0, staging, 0, this.totalW * H * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    const out = {};
    for (const n of this.names) {
      const cnt = this.shapes[n].reduce((a, b) => a * b, 1);
      out[n] = { data: all.subarray(this.off[n] * H, (this.off[n] + cnt) * H), shape: [...this.shapes[n], H] };
    }
    return out;
  }
}
