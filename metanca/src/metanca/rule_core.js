// Shared orchestration for the MetaNCA MLP update: neighbor gathering, perception
// assembly (incl. the bwd-index clip quirk), and scatter. The expensive math
// (attention + head) is delegated to a pluggable `backend` so the CPU oracle and
// the WebGPU runtime share identical, validated glue.
import { H } from "./posenc.js";
import { addFeat, groupByAxis, invertView, concatAxis1, concatLast } from "./nd.js";

export const STATE = 31;
export const PERCEP = 3 * STATE; // 93

// neighbor spec for MLP (matches mlp_reference._neighbor_spec). First entry = self.
export function neighborSpec(name, arch, dir) {
  const nL = arch.length;
  const layer = parseInt(name.split(".")[0].split("_")[1]);
  const ptype = name.split(".")[1];
  let fdim, nbrs;
  if (ptype === "kernel") {
    if (dir === "fwd") {
      fdim = 1; nbrs = [[name, 1], [`Dense_${layer}.bias`, 0]];
      if (layer + 1 < nL) nbrs.push([`Dense_${layer + 1}.kernel`, 0]);
    } else {
      fdim = 0; nbrs = [[name, 0]];
      if (layer > 0) { nbrs.push([`Dense_${layer - 1}.bias`, 0]); nbrs.push([`Dense_${layer - 1}.kernel`, 1]); }
    }
  } else {
    if (dir === "fwd") {
      fdim = 0; nbrs = [[name, 0], [`Dense_${layer}.kernel`, 1]];
      if (layer + 1 < nL) nbrs.push([`Dense_${layer + 1}.kernel`, 0]);
    } else { fdim = 0; nbrs = [[name, 0]]; }
  }
  return { fdim, nbrs };
}

function gather(params, hidden, posenc, name, indexDim) {
  const p = params[name];
  const w = groupByAxis(addFeat(p.data, p.shape), indexDim);
  const h = groupByAxis(hidden[name], indexDim);
  const pe = groupByAxis(posenc[name], indexDim);
  return { w, h, pe };
}

// Build the per-param fwd/bwd inputs (focus + neighbors). Returns plain tensors.
export function buildInputs(params, hidden, posenc, name, arch) {
  const specF = neighborSpec(name, arch, "fwd");
  const specB = neighborSpec(name, arch, "bwd");
  const f = gather(params, hidden, posenc, name, specF.fdim);
  const b = gather(params, hidden, posenc, name, specB.fdim);
  const nf = specF.nbrs.map(([n, d]) => gather(params, hidden, posenc, n, d));
  const nb = specB.nbrs.map(([n, d]) => gather(params, hidden, posenc, n, d));
  return {
    fdim: specF.fdim, bdim: specB.fdim,
    focusFwd: concatLast(f.w, f.h), fp: f.pe,
    neighFwd: concatAxis1(nf.map((g) => concatLast(g.w, g.h))), neighFp: concatAxis1(nf.map((g) => g.pe)),
    focusBwd: concatLast(b.w, b.h), bp: b.pe,
    neighBwd: concatAxis1(nb.map((g) => concatLast(g.w, g.h))), neighBp: concatAxis1(nb.map((g) => g.pe)),
    fw: f.w, fh: f.h, pshape: params[name].shape,
  };
}

// Transform agg_bwd from the backward neuron view into the forward neuron view
// (main's bug fix): invert to param shape via bdim, then re-group via fdim.
export function bwdToFwdFrame(aggB, pshape, bdim, fdim) {
  const paramShaped = invertView(aggB, [...pshape, STATE], bdim, [...pshape, STATE]);
  return groupByAxis(paramShaped, fdim);   // [Nf, Wf, STATE]
}

// Assemble perception (cells, 93): [focus, aggF, aggB] (no count division; aggB already
// transformed to the forward frame so it indexes with the same [i,j]). Row-major.
export function assemblePerception(focusFwd, aggF, aggBfwd) {
  const [Nf, Wf] = focusFwd.shape;
  const cells = Nf * Wf;
  const out = new Float32Array(cells * PERCEP);
  for (let i = 0, c = 0; i < Nf; i++)
    for (let j = 0; j < Wf; j++, c++) {
      const off = (i * Wf + j) * STATE, po = c * PERCEP;
      for (let d = 0; d < STATE; d++) {
        out[po + d] = focusFwd.data[off + d];
        out[po + STATE + d] = aggF.data[off + d];
        out[po + 2 * STATE + d] = aggBfwd.data[off + d];
      }
    }
  return { data: out, shape: [cells, PERCEP] };
}

// Scatter head output (cells,31) deltas into the fwd view, invert to param shape.
export function scatter(inp, out) {
  const { fw, fh, pshape, fdim } = inp;
  const Nf = fw.shape[0], Wf = fw.shape[1];
  const uw = new Float32Array(fw.data), uh = new Float32Array(fh.data);
  let c = 0;
  for (let i = 0; i < Nf; i++)
    for (let j = 0; j < Wf; j++) {
      uw[(i * Wf + j)] += out.data[c * STATE + 0];
      const ho = (i * Wf + j) * H;
      for (let d = 0; d < H; d++) uh[ho + d] += out.data[c * STATE + 1 + d];
      c++;
    }
  return {
    param: invertView({ data: uw, shape: [Nf, Wf, 1] }, [...pshape, 1], fdim, pshape),
    hidden: invertView({ data: uh, shape: [Nf, Wf, H] }, [...pshape, H], fdim, [...pshape, H]),
  };
}

// One update step with a pluggable backend. backend.attn(dir, focus, neigh, qpos, kpos)
// returns the (undivided) aggregation; backend.head(percep) returns (cells,31).
export async function updateStep(params, hidden, posenc, arch, backend) {
  const names = [];
  for (let L = 0; L < arch.length; L++) { names.push(`Dense_${L}.bias`); names.push(`Dense_${L}.kernel`); }
  const newParams = {}, newHidden = {};
  for (const name of names) {
    const inp = buildInputs(params, hidden, posenc, name, arch);
    const aggF = await backend.attn("fwd", inp.focusFwd, inp.neighFwd, inp.fp, inp.neighFp);
    const aggBraw = await backend.attn("bwd", inp.focusBwd, inp.neighBwd, inp.bp, inp.neighBp);
    const aggB = bwdToFwdFrame(aggBraw, inp.pshape, inp.bdim, inp.fdim);
    const percep = assemblePerception(inp.focusFwd, aggF, aggB);
    const out = await backend.head(percep);
    const s = scatter(inp, out);
    newParams[name] = s.param;
    newHidden[name] = s.hidden;
  }
  return { params: newParams, hidden: newHidden };
}
