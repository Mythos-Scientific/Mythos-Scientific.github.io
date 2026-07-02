// Minimal tensor helpers (flat Float32Array + shape). Mirrors the numpy ops used
// by the MetaNCA MLP reference: grouping (swap_axes_and_reshape) and its inverse,
// specialized to ranks 2 (bias) and 3 (kernel) with a trailing feature dim.

export const prod = (s) => s.reduce((a, b) => a * b, 1);

// Add a trailing length-1 feature axis: shape S -> S+[1].
export function addFeat(data, shape) {
  return { data, shape: [...shape, 1] };
}

// group-by-axis == swap_axes_and_reshape(arr, axis, feat, stride=None).
// Input t has shape [...dims, F] (rank 2 or 3); `axis` is the neuron axis (0 or 1,
// never the trailing feature axis). Returns shape [dims[axis], mid, F].
export function groupByAxis(t, axis) {
  const s = t.shape;
  const R = s.length;
  const F = s[R - 1];
  if (R === 2) {
    // [d0, F], axis must be 0 -> [d0, 1, F] (data unchanged, C-order preserved)
    return { data: t.data, shape: [s[0], 1, F] };
  }
  // R === 3: [a, b, F]
  const [a, b] = s;
  if (axis === 0) return { data: t.data, shape: [a, b, F] };
  // axis === 1: transpose first two dims -> [b, a, F]
  const out = new Float32Array(a * b * F);
  for (let i = 0; i < a; i++)
    for (let j = 0; j < b; j++)
      for (let f = 0; f < F; f++)
        out[(j * a + i) * F + f] = t.data[(i * b + j) * F + f];
  return { data: out, shape: [b, a, F] };
}

// invert_swap_axes_and_reshape(arr, originalShape, indexDim, stride=None).
// arr has shape [N, W, F]; returns a tensor of shape `targetShape`.
export function invertView(arr, originalShapeWithFeat, indexDim, targetShape) {
  const F = originalShapeWithFeat[originalShapeWithFeat.length - 1];
  if (indexDim === 0) {
    // reshape arr -> originalShape (C-order preserved)
    return { data: arr.data, shape: targetShape };
  }
  // indexDim === 1, rank-3 kernel: arr is [out, in, F] -> [in, out, F]
  const [outD, inD] = arr.shape; // arr.shape = [N=out, W=in, F]
  const data = new Float32Array(outD * inD * F);
  for (let j = 0; j < outD; j++)
    for (let i = 0; i < inD; i++)
      for (let f = 0; f < F; f++)
        data[(i * outD + j) * F + f] = arr.data[(j * inD + i) * F + f];
  return { data, shape: targetShape };
}

// concatenate a list of [N, Wk, F] tensors along axis 1 -> [N, sumWk, F].
export function concatAxis1(list) {
  const N = list[0].shape[0];
  const F = list[0].shape[2];
  const totW = list.reduce((a, t) => a + t.shape[1], 0);
  const out = new Float32Array(N * totW * F);
  for (let n = 0; n < N; n++) {
    let wOff = 0;
    for (const t of list) {
      const W = t.shape[1];
      for (let w = 0; w < W; w++)
        for (let f = 0; f < F; f++)
          out[(n * totW + (wOff + w)) * F + f] = t.data[(n * W + w) * F + f];
      wOff += W;
    }
  }
  return { data: out, shape: [N, totW, F] };
}

// concatenate two [N, W, *] tensors along the last axis.
export function concatLast(a, b) {
  const [N, W, Fa] = a.shape;
  const Fb = b.shape[2];
  const F = Fa + Fb;
  const out = new Float32Array(N * W * F);
  for (let n = 0; n < N; n++)
    for (let w = 0; w < W; w++) {
      for (let f = 0; f < Fa; f++) out[(n * W + w) * F + f] = a.data[(n * W + w) * Fa + f];
      for (let f = 0; f < Fb; f++) out[(n * W + w) * F + Fa + f] = b.data[(n * W + w) * Fb + f];
    }
  return { data: out, shape: [N, W, F] };
}
