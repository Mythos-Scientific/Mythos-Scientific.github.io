// CPU backend (validation oracle) for the MetaNCA MLP rule. The expensive math
// (ELU linear attention + head) lives here; gather/perception/scatter are shared
// in rule_core.js. Validated bit-for-bit vs metanca.update_tasknet.
import { updateStep as coreUpdateStep, STATE } from "./rule_core.js";

export const HEADS = 3;
export const HEAD_DIM = 10;
const LEAKY = 0.01;
const EPS_ATTN = 1e-6;
const EPS_LN = 1e-6;

const elu = (x) => (x > 0 ? x : Math.exp(Math.min(x, 0)) - 1.0);
const featureMap = (x) => elu(x) + 1.0;

function layerNorm(v, n, gamma, beta) {
  let mean = 0; for (let i = 0; i < n; i++) mean += v[i]; mean /= n;
  let varr = 0; for (let i = 0; i < n; i++) { const d = v[i] - mean; varr += d * d; } varr /= n;
  const inv = 1 / Math.sqrt(varr + EPS_LN);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = gamma[i] * (v[i] - mean) * inv + beta[i];
  return out;
}

function ropeHead(x, pos, out) {
  for (let m = 0; m < HEAD_DIM / 2; m++) {
    const sin = pos[2 * m], cos = pos[2 * m + 1];
    const xe = x[2 * m], xo = x[2 * m + 1];
    out[2 * m] = xe * cos - xo * sin;
    out[2 * m + 1] = xe * sin + xo * cos;
  }
}

class Attn {
  constructor(W, prefix) {
    const g = (s) => W[`${prefix}.${s}`].data;
    this.Wq = g("query.kernel"); this.Wk = g("key.kernel"); this.Wv = g("value.kernel");
    this.Wout = g("out.kernel"); this.g1 = g("gamma_1"); this.b1 = g("beta_1");
    this.g2 = g("gamma_2"); this.b2 = g("beta_2"); this.Wff1 = g("ff_1.kernel"); this.Wff2 = g("ff_2.kernel");
    this.bff1 = g("ff_1.bias"); this.bff2 = g("ff_2.bias");
    this.ffH = W[`${prefix}.ff_1.kernel`].shape[1];
  }
  run(focus, neigh, qpos, kpos) {
    const B = focus.shape[0], Lq = focus.shape[1], Lk = neigh.shape[1];
    const out = new Float32Array(B * Lq * STATE);
    const HD = HEAD_DIM, NH = HEADS, ffH = this.ffH;
    const tmp = new Float32Array(HD), dst = new Float32Array(HD);
    for (let b = 0; b < B; b++) {
      const kphi = new Float32Array(Lk * NH * HD), vv = new Float32Array(Lk * NH * HD);
      for (let l = 0; l < Lk; l++) {
        const nbase = (b * Lk + l) * STATE, pbase = (b * Lk + l) * (NH * HD);
        for (let h = 0; h < NH; h++) {
          for (let k = 0; k < HD; k++) {
            let sk = 0, sv = 0; const woff = h * HD + k;
            for (let d = 0; d < STATE; d++) { const nv = neigh.data[nbase + d]; sk += nv * this.Wk[d * NH * HD + woff]; sv += nv * this.Wv[d * NH * HD + woff]; }
            tmp[k] = featureMap(sk); vv[(l * NH + h) * HD + k] = sv;
          }
          ropeHead(tmp, kpos.data.subarray(pbase + h * HD, pbase + h * HD + HD), dst);
          for (let k = 0; k < HD; k++) kphi[(l * NH + h) * HD + k] = dst[k];
        }
      }
      const kv = new Float32Array(NH * HD * HD), ksum = new Float32Array(NH * HD);
      for (let l = 0; l < Lk; l++)
        for (let h = 0; h < NH; h++)
          for (let k = 0; k < HD; k++) {
            const kp = kphi[(l * NH + h) * HD + k]; ksum[h * HD + k] += kp;
            const kvoff = (h * HD + k) * HD, voff = (l * NH + h) * HD;
            for (let m = 0; m < HD; m++) kv[kvoff + m] += kp * vv[voff + m];
          }
      const qphi = new Float32Array(NH * HD);
      for (let lq = 0; lq < Lq; lq++) {
        const fbase = (b * Lq + lq) * STATE, pbase = (b * Lq + lq) * (NH * HD);
        for (let h = 0; h < NH; h++) {
          for (let k = 0; k < HD; k++) {
            let sq = 0; const woff = h * HD + k;
            for (let d = 0; d < STATE; d++) sq += focus.data[fbase + d] * this.Wq[d * NH * HD + woff];
            tmp[k] = featureMap(sq);
          }
          ropeHead(tmp, qpos.data.subarray(pbase + h * HD, pbase + h * HD + HD), dst);
          for (let k = 0; k < HD; k++) qphi[h * HD + k] = dst[k];
        }
        const attnflat = new Float32Array(NH * HD);
        for (let h = 0; h < NH; h++) {
          let den = EPS_ATTN;
          for (let k = 0; k < HD; k++) den += qphi[h * HD + k] * ksum[h * HD + k];
          for (let m = 0; m < HD; m++) { let num = 0; for (let k = 0; k < HD; k++) num += qphi[h * HD + k] * kv[(h * HD + k) * HD + m]; attnflat[h * HD + m] = num / den; }
        }
        const o = new Float32Array(STATE);
        for (let d = 0; d < STATE; d++) { let s = 0; for (let e = 0; e < NH * HD; e++) s += attnflat[e] * this.Wout[e * STATE + d]; o[d] = s + focus.data[fbase + d]; }
        const nr = layerNorm(o, STATE, this.g1, this.b1);
        const hid = new Float32Array(ffH), ff = new Float32Array(STATE);
        for (let hh = 0; hh < ffH; hh++) { let s = this.bff1[hh]; for (let d = 0; d < STATE; d++) s += nr[d] * this.Wff1[d * ffH + hh]; hid[hh] = s > 0 ? s : 0; }
        for (let d = 0; d < STATE; d++) { let s = this.bff2[d]; for (let hh = 0; hh < ffH; hh++) s += hid[hh] * this.Wff2[hh * STATE + d]; ff[d] = s + nr[d]; }
        const res = layerNorm(ff, STATE, this.g2, this.b2);
        for (let d = 0; d < STATE; d++) out[fbase + d] = res[d];
      }
    }
    return { data: out, shape: [B, Lq, STATE] };
  }
}

export class MLPRule {
  constructor(W) {
    this.fwd = new Attn(W, "attn_forward");
    this.bwd = new Attn(W, "attn_backward");
    this.layers = [];
    let i = 0;
    while (W[`local_rule_head.Dense_${i}.kernel`]) {
      this.layers.push({ k: W[`local_rule_head.Dense_${i}.kernel`], b: W[`local_rule_head.Dense_${i}.bias`].data });
      i++;
    }
  }
  attn(dir, focus, neigh, qpos, kpos) { return this[dir].run(focus, neigh, qpos, kpos); }
  head(x) {
    let cur = x; const n = this.layers.length;
    for (let li = 0; li < n; li++) {
      const { k, b } = this.layers[li]; const inD = k.shape[0], outD = k.shape[1], N = cur.shape[0];
      const out = new Float32Array(N * outD);
      for (let r = 0; r < N; r++)
        for (let o = 0; o < outD; o++) {
          let s = b[o]; for (let d = 0; d < inD; d++) s += cur.data[r * inD + d] * k.data[d * outD + o];
          out[r * outD + o] = li < n - 1 ? elu(s) : Math.tanh(s);
        }
      cur = { data: out, shape: [N, outD] };
    }
    return cur;
  }
}

export function updateStep(params, hidden, posenc, rule, arch) {
  return coreUpdateStep(params, hidden, posenc, arch, rule);
}
