// Positional-encoding init for dense-MLP task nets. Mirrors
// metanca.hidden_state and mlp_reference.init_hidden_posenc.
// hidden_states == positional_encodings at init.
//   kernel W_L[i,j] hidden = [posenc10(i), posenc10(j), posenc10(L)]   (H=30)
//   bias   b_L[j]   hidden = [BIAS_CONSTANT x10, posenc10(j), posenc10(L)]

export const H = 30;
export const D_SLOT = 10;
export const BIAS_CONSTANT = -10.0;
export const INPUT_DIM = 784;

// posenc(pos, d): sin/cos sinusoidal encoding -> Float32Array(d).
export function posenc(pos, d = D_SLOT) {
  const out = new Float32Array(d);
  for (let k = 0; k < d; k += 2) {
    const div = Math.exp(k * -(Math.log(10000.0) / d));
    out[k] = Math.sin(pos * div);
    if (k + 1 < d) out[k + 1] = Math.cos(pos * div);
  }
  return out;
}

// Build hidden/posenc dict for arch (list of output dims; input = INPUT_DIM).
// Returns { "Dense_L.kernel": {data,shape:[in,out,H]}, "Dense_L.bias": {data,shape:[out,H]} }
export function initHiddenPosenc(arch) {
  const dims = [INPUT_DIM, ...arch];
  const cache = new Map();
  const pe = (p) => {
    if (!cache.has(p)) cache.set(p, posenc(p, D_SLOT));
    return cache.get(p);
  };
  // bias in-neuron "sentinel" PE at position max(neuron dims) (matches metanca main)
  const biasSentinel = pe(Math.max(...dims));
  const out = {};
  for (let L = 0; L < arch.length; L++) {
    const inL = dims[L], outL = dims[L + 1];
    const peL = pe(L);
    // kernel (in, out, H)
    const K = new Float32Array(inL * outL * H);
    for (let i = 0; i < inL; i++) {
      const pei = pe(i);
      for (let j = 0; j < outL; j++) {
        const pej = pe(j);
        const base = (i * outL + j) * H;
        for (let c = 0; c < 10; c++) {
          K[base + c] = pei[c];
          K[base + 10 + c] = pej[c];
          K[base + 20 + c] = peL[c];
        }
      }
    }
    out[`Dense_${L}.kernel`] = { data: K, shape: [inL, outL, H] };
    // bias (out, H)
    const B = new Float32Array(outL * H);
    for (let j = 0; j < outL; j++) {
      const pej = pe(j);
      const base = j * H;
      for (let c = 0; c < 10; c++) {
        B[base + c] = biasSentinel[c];   // was BIAS_CONSTANT on old branch
        B[base + 10 + c] = pej[c];
        B[base + 20 + c] = peL[c];
      }
    }
    out[`Dense_${L}.bias`] = { data: B, shape: [outL, H] };
  }
  return out;
}
