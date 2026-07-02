// Architecture morphing: carry weights + hidden states across an arch edit.
//
// When the user resizes/adds/removes a hidden layer we do NOT reset the task
// net. Every layer of the new arch that corresponds to a layer of the old arch
// (via `map`) keeps its surviving weights AND their evolved hidden states in
// the overlapping region; only newly created slots (grown widths, inserted
// layers) get fresh values — random weights, posenc-initialized hidden states.
// Positional encodings are always rebuilt for the new arch (they encode the
// weight's *current* coordinates, which the rule reads via RoPE).
import { H, initHiddenPosenc } from "./posenc.js";

// map[newL] = old layer index that new layer newL inherits from, or -1 if the
// layer is brand new. Layer L covers Dense_L.kernel + Dense_L.bias.
export function archEditMap(oldArch, newArch, op) {
  // op: {type:"resize"} | {type:"add", at} | {type:"remove", at}
  const map = new Array(newArch.length);
  for (let L = 0; L < newArch.length; L++) {
    if (op.type === "resize") map[L] = L;
    else if (op.type === "add") map[L] = L < op.at ? L : L === op.at ? -1 : L - 1;
    else map[L] = L < op.at ? L : L + 1; // remove
  }
  return map;
}

export function carryOver(oldParams, oldHidden, newArch, map, randInit) {
  const fresh = randInit(newArch);           // new-slot weights (matches Reset init)
  const freshHP = initHiddenPosenc(newArch); // posenc for new arch + hidden init for new slots
  const params = {}, hidden = {};
  for (let L = 0; L < newArch.length; L++) {
    const oldL = map[L];
    for (const pt of ["kernel", "bias"]) {
      const fp = fresh[`Dense_${L}.${pt}`];
      const dst = { data: new Float32Array(fp.data), shape: fp.shape };
      const fh = freshHP[`Dense_${L}.${pt}`];
      const dstH = { data: new Float32Array(fh.data), shape: fh.shape };
      if (oldL >= 0 && oldParams[`Dense_${oldL}.${pt}`]) {
        const src = oldParams[`Dense_${oldL}.${pt}`];
        const srcH = oldHidden[`Dense_${oldL}.${pt}`];
        if (pt === "kernel") {
          const [inN, outN] = dst.shape, [inO, outO] = src.shape;
          const ci = Math.min(inN, inO), co = Math.min(outN, outO);
          for (let i = 0; i < ci; i++)
            for (let j = 0; j < co; j++) {
              dst.data[i * outN + j] = src.data[i * outO + j];
              for (let d = 0; d < H; d++) dstH.data[(i * outN + j) * H + d] = srcH.data[(i * outO + j) * H + d];
            }
        } else {
          const co = Math.min(dst.shape[0], src.shape[0]);
          dst.data.set(src.data.subarray(0, co));
          dstH.data.set(srcH.data.subarray(0, co * H));
        }
      }
      params[`Dense_${L}.${pt}`] = dst;
      hidden[`Dense_${L}.${pt}`] = dstH;
    }
  }
  return { params, hidden, posenc: freshHP };
}
