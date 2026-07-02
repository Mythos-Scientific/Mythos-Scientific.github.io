// Task-net forward pass (dense MLP) + accuracy. Mirrors flax MultiLayerPerceptron:
// nn.Dense(kernel (in,out) + bias), leaky_relu(0.01) on hidden layers, identity out.
const LEAKY = 0.01;

// params: { "Dense_L.kernel": {data,shape:[in,out]}, "Dense_L.bias": {data,shape:[out]} }
// x: {data, shape:[N, inDim]} -> logits {data, shape:[N, outDim]}
export function tasknetForward(params, x, arch) {
  let cur = x;
  const n = arch.length;
  for (let L = 0; L < n; L++) {
    const k = params[`Dense_${L}.kernel`];
    const b = params[`Dense_${L}.bias`].data;
    const inD = k.shape[0], outD = k.shape[1];
    const N = cur.shape[0];
    const out = new Float32Array(N * outD);
    for (let r = 0; r < N; r++) {
      const xoff = r * inD;
      for (let o = 0; o < outD; o++) {
        let s = b[o];
        for (let i = 0; i < inD; i++) s += cur.data[xoff + i] * k.data[i * outD + o];
        out[r * outD + o] = L < n - 1 ? (s > 0 ? s : LEAKY * s) : s;
      }
    }
    cur = { data: out, shape: [N, outD] };
  }
  return cur;
}

export function argmaxRows(logits) {
  const [N, C] = logits.shape;
  const out = new Int32Array(N);
  for (let r = 0; r < N; r++) {
    let best = -Infinity, bi = 0;
    for (let c = 0; c < C; c++) { const v = logits.data[r * C + c]; if (v > best) { best = v; bi = c; } }
    out[r] = bi;
  }
  return out;
}

// labels: Int/Float array of class indices (length N)
export function accuracyFromLabels(logits, labels) {
  const pred = argmaxRows(logits);
  let correct = 0;
  for (let r = 0; r < pred.length; r++) if (pred[r] === labels[r]) correct++;
  return correct / pred.length;
}

// y one-hot {data, shape:[N,C]}
export function accuracyOneHot(logits, y) {
  const pred = argmaxRows(logits);
  const [N, C] = y.shape;
  let correct = 0;
  for (let r = 0; r < N; r++) {
    let best = -Infinity, bi = 0;
    for (let c = 0; c < C; c++) { const v = y.data[r * C + c]; if (v > best) { best = v; bi = c; } }
    if (pred[r] === bi) correct++;
  }
  return correct / N;
}
