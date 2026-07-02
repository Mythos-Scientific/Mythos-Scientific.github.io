// Faithful JS simulation of the WGSL kernels in gpu.js (same flat-buffer indexing,
// same packed-weight offsets, same per-thread math). Used to validate the WGSL
// logic/layout against the CPU oracle without a GPU. NOT used at runtime.
import { STATE } from "./rule_core.js";

const HEADS = 3, HEAD_DIM = 10, HD30 = 30;
const O_WOUT = 930, O_G1 = 1860, O_B1 = 1891, O_G2 = 1922, O_B2 = 1953,
  O_FF1 = 1984, O_BFF1 = 5084, O_FF2 = 5184, O_BFF2 = 8284;
const fmap = (x) => (x > 0 ? x + 1.0 : Math.exp(x));
const eluf = (x) => (x > 0 ? x : Math.exp(x) - 1.0);

function concatF32(list) {
  const total = list.reduce((a, x) => a + x.length, 0);
  const out = new Float32Array(total); let o = 0;
  for (const x of list) { out.set(x, o); o += x.length; }
  return out;
}

export class GPUSimRule {
  constructor(W) {
    this.attnW = {};
    for (const [dir, pre] of [["fwd", "attn_forward"], ["bwd", "attn_backward"]]) {
      this.attnW[dir] = {
        Wk: W[`${pre}.key.kernel`].data, Wv: W[`${pre}.value.kernel`].data,
        pack: concatF32([
          W[`${pre}.query.kernel`].data, W[`${pre}.out.kernel`].data,
          W[`${pre}.gamma_1`].data, W[`${pre}.beta_1`].data,
          W[`${pre}.gamma_2`].data, W[`${pre}.beta_2`].data,
          W[`${pre}.ff_1.kernel`].data, W[`${pre}.ff_1.bias`].data,
          W[`${pre}.ff_2.kernel`].data, W[`${pre}.ff_2.bias`].data,
        ]),
      };
    }
    this.headL = [];
    let i = 0;
    while (W[`local_rule_head.Dense_${i}.kernel`]) {
      const k = W[`local_rule_head.Dense_${i}.kernel`], b = W[`local_rule_head.Dense_${i}.bias`];
      this.headL.push({ k: k.data, b: b.data, inD: k.shape[0], outD: k.shape[1] });
      i++;
    }
  }

  attn(dir, focus, neigh, qpos, kpos) {
    const w = this.attnW[dir];
    const B = focus.shape[0], Lq = focus.shape[1], Lk = neigh.shape[1];
    const Wk = w.Wk, Wv = w.Wv, pk = w.pack;
    const kphi = new Float32Array(B * Lk * HD30), vv = new Float32Array(B * Lk * HD30);
    // K1
    for (let idx = 0; idx < B * Lk; idx++) {
      const nbase = idx * STATE, pbase = idx * HD30;
      for (let h = 0; h < HEADS; h++) {
        const tmp = new Float32Array(HEAD_DIM);
        for (let k = 0; k < HEAD_DIM; k++) {
          const woff = h * HEAD_DIM + k; let sk = 0, sv = 0;
          for (let d = 0; d < STATE; d++) { const nvv = neigh.data[nbase + d]; sk += nvv * Wk[d * HD30 + woff]; sv += nvv * Wv[d * HD30 + woff]; }
          tmp[k] = fmap(sk); vv[idx * HD30 + h * HEAD_DIM + k] = sv;
        }
        for (let m = 0; m < 5; m++) {
          const s = kpos.data[pbase + h * HEAD_DIM + 2 * m], c = kpos.data[pbase + h * HEAD_DIM + 2 * m + 1];
          const xe = tmp[2 * m], xo = tmp[2 * m + 1];
          kphi[idx * HD30 + h * HEAD_DIM + 2 * m] = xe * c - xo * s;
          kphi[idx * HD30 + h * HEAD_DIM + 2 * m + 1] = xe * s + xo * c;
        }
      }
    }
    // K2
    const kv = new Float32Array(B * HD30 * HEAD_DIM), ksum = new Float32Array(B * HD30);
    for (let idx = 0; idx < B * HD30; idx++) {
      const b = (idx / HD30) | 0, hk = idx % HD30, h = (hk / HEAD_DIM) | 0;
      let s = 0; const acc = new Float32Array(HEAD_DIM);
      for (let l = 0; l < Lk; l++) {
        const kp = kphi[(b * Lk + l) * HD30 + hk]; s += kp;
        const voff = (b * Lk + l) * HD30 + h * HEAD_DIM;
        for (let m = 0; m < HEAD_DIM; m++) acc[m] += kp * vv[voff + m];
      }
      ksum[idx] = s;
      for (let m = 0; m < HEAD_DIM; m++) kv[(b * HD30 + hk) * HEAD_DIM + m] = acc[m];
    }
    // K3
    const out = new Float32Array(B * Lq * STATE);
    for (let idx = 0; idx < B * Lq; idx++) {
      const b = (idx / Lq) | 0, fbase = idx * STATE, pbase = idx * HD30;
      const qphi = new Float32Array(HD30);
      for (let h = 0; h < HEADS; h++) {
        const tmp = new Float32Array(HEAD_DIM);
        for (let k = 0; k < HEAD_DIM; k++) {
          const woff = h * HEAD_DIM + k; let sq = 0;
          for (let d = 0; d < STATE; d++) sq += focus.data[fbase + d] * pk[d * HD30 + woff];
          tmp[k] = fmap(sq);
        }
        for (let m = 0; m < 5; m++) {
          const s = qpos.data[pbase + h * HEAD_DIM + 2 * m], c = qpos.data[pbase + h * HEAD_DIM + 2 * m + 1];
          const xe = tmp[2 * m], xo = tmp[2 * m + 1];
          qphi[h * HEAD_DIM + 2 * m] = xe * c - xo * s;
          qphi[h * HEAD_DIM + 2 * m + 1] = xe * s + xo * c;
        }
      }
      const attnflat = new Float32Array(HD30);
      for (let h = 0; h < HEADS; h++) {
        let den = 1e-6;
        for (let k = 0; k < HEAD_DIM; k++) den += qphi[h * HEAD_DIM + k] * ksum[b * HD30 + h * HEAD_DIM + k];
        for (let m = 0; m < HEAD_DIM; m++) {
          let num = 0;
          for (let k = 0; k < HEAD_DIM; k++) num += qphi[h * HEAD_DIM + k] * kv[(b * HD30 + h * HEAD_DIM + k) * HEAD_DIM + m];
          attnflat[h * HEAD_DIM + m] = num / den;
        }
      }
      const o = new Float32Array(STATE);
      for (let d = 0; d < STATE; d++) { let s = 0; for (let e = 0; e < HD30; e++) s += attnflat[e] * pk[O_WOUT + e * STATE + d]; o[d] = s + focus.data[fbase + d]; }
      let mean = 0; for (let d = 0; d < STATE; d++) mean += o[d]; mean /= 31;
      let v2 = 0; for (let d = 0; d < STATE; d++) { const t = o[d] - mean; v2 += t * t; } v2 /= 31;
      const nr = new Float32Array(STATE); const inv1 = 1 / Math.sqrt(v2 + 1e-6);
      for (let d = 0; d < STATE; d++) nr[d] = pk[O_G1 + d] * (o[d] - mean) * inv1 + pk[O_B1 + d];
      const hid = new Float32Array(100), ff = new Float32Array(STATE);
      for (let hh = 0; hh < 100; hh++) { let s = pk[O_BFF1 + hh]; for (let d = 0; d < STATE; d++) s += nr[d] * pk[O_FF1 + d * 100 + hh]; hid[hh] = Math.max(s, 0); }
      for (let d = 0; d < STATE; d++) { let s = pk[O_BFF2 + d]; for (let hh = 0; hh < 100; hh++) s += hid[hh] * pk[O_FF2 + hh * STATE + d]; ff[d] = s + nr[d]; }
      let mean2 = 0; for (let d = 0; d < STATE; d++) mean2 += ff[d]; mean2 /= 31;
      let vr2 = 0; for (let d = 0; d < STATE; d++) { const t = ff[d] - mean2; vr2 += t * t; } vr2 /= 31;
      const inv2 = 1 / Math.sqrt(vr2 + 1e-6);
      for (let d = 0; d < STATE; d++) out[fbase + d] = pk[O_G2 + d] * (ff[d] - mean2) * inv2 + pk[O_B2 + d];
    }
    return { data: out, shape: [B, Lq, STATE] };
  }

  head(percep) {
    let cur = percep;
    for (let li = 0; li < this.headL.length; li++) {
      const L = this.headL[li], act = li < this.headL.length - 1 ? 0 : 1;
      const N = cur.shape[0], out = new Float32Array(N * L.outD);
      for (let idx = 0; idx < N * L.outD; idx++) {
        const r = (idx / L.outD) | 0, o = idx % L.outD;
        let s = L.b[o];
        for (let d = 0; d < L.inD; d++) s += cur.data[r * L.inD + d] * L.k[d * L.outD + o];
        out[idx] = act === 0 ? eluf(s) : Math.tanh(s);
      }
      cur = { data: out, shape: [N, L.outD] };
    }
    return cur;
  }
}
