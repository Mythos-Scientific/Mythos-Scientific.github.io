// MetaNCA interactive demo controller + visualization.
// Wires the all-GPU engine to weight-field heatmaps, a live accuracy curve, and
// MNIST sample predictions. The local rule (fixed, trained) edits the task net's
// weights each step; the architecture is user-editable and rebuilt on the fly.
import { loadBundle } from "./src/metanca/bundle.js";
import { initHiddenPosenc, INPUT_DIM } from "./src/metanca/posenc.js";
import { tasknetForward, argmaxRows } from "./src/metanca/tasknet.js";
import { createDevice } from "./src/metanca/gpu.js";
import { GPUEngine } from "./src/metanca/engine.js";
import { archEditMap, carryOver } from "./src/metanca/morph.js";

const OUTPUT_DIM = 10;
const ACC_N = 1000;          // images used for the live accuracy readout
const DIGIT_SAMPLES = 24;
const MAX_LAYERS = 4;
const MIN_W = 4, MAX_W = 200;

// crisp line icons for the weight-field controls (inherit color via currentColor)
const SVG_ARROW = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h14M12 5.5l6.5 6.5-6.5 6.5"/></svg>`;
const SVG_PLUS = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M12 5.5v13M5.5 12h13"/></svg>`;
const SVG_X = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/></svg>`;

const $ = (s) => document.querySelector(s);
const els = {};
let engine, mnist, ruleBundle, device;
let arch = [50, 30, OUTPUT_DIM];   // hidden... + output; input 784 implicit
let playing = false, steps = 0, targetRate = 5;  // target steps/sec; Infinity = uncapped ("max")
let accHistory = [];
let lastStepAt = 0, stepRate = 0;   // wall-clock cadence -> measured steps/sec readout

// ---------- weight colormap: teal (neg) <- dark substrate -> coral (pos) ----------
const SUBSTRATE = [13, 16, 23];
const TEAL = [56, 214, 190];
const CORAL = [255, 124, 80];
function colormap(v, scale) {
  let t = Math.max(-1, Math.min(1, v / scale));
  const dir = t >= 0 ? CORAL : TEAL;
  const m = Math.pow(Math.abs(t), 0.42);
  return [
    SUBSTRATE[0] + m * (dir[0] - SUBSTRATE[0]),
    SUBSTRATE[1] + m * (dir[1] - SUBSTRATE[1]),
    SUBSTRATE[2] + m * (dir[2] - SUBSTRATE[2]),
  ];
}

// ---------- He/lecun-style init (matches training distribution) ----------
function randn() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function randInit(arch) {
  const dims = [INPUT_DIM, ...arch];
  const p = {};
  for (let L = 0; L < arch.length; L++) {
    const inD = dims[L], outD = dims[L + 1];
    const std = Math.sqrt(1 / inD);
    const k = new Float32Array(inD * outD);
    for (let i = 0; i < k.length; i++) k[i] = randn() * std;
    p[`Dense_${L}.kernel`] = { data: k, shape: [inD, outD] };
    p[`Dense_${L}.bias`] = { data: new Float32Array(outD), shape: [outD] };
  }
  return p;
}

// ---------- heatmap rendering ----------
function drawHeatmap(canvas, tensor, scale) {
  const [inD, outD] = tensor.shape;
  const off = document.createElement("canvas");
  off.width = outD; off.height = inD;           // columns = output neurons, rows = input
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(outD, inD);
  for (let i = 0; i < inD; i++)
    for (let j = 0; j < outD; j++) {
      const [r, g, b] = colormap(tensor.data[i * outD + j], scale);
      const o = (i * outD + j) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  ctx.putImageData(img, 0, 0);
  const dc = canvas.getContext("2d");
  // heavy row downsampling (e.g. 784 rows) reads better smoothed; crisp otherwise
  dc.imageSmoothingEnabled = inD / canvas.height > 3;
  dc.clearRect(0, 0, canvas.width, canvas.height);
  // fill the canvas — its aspect is pre-clamped to the matrix's in buildLayerField,
  // so ordinary layers keep their true aspect and extreme ones are gently compressed
  dc.drawImage(off, 0, 0, canvas.width, canvas.height);
}

// Build the weight-field DOM *and* the inline architecture controls. Called once
// per arch change (rebuild / morph). Per-step redraws go through drawHeatmaps
// (canvas pixels only), so a slider being dragged is never torn down mid-drag.
let layerCanvases = [];
function buildLayerField() {
  els.layers.innerHTML = "";
  layerCanvases = [];
  const dims = [INPUT_DIM, ...arch];
  const hiddenCount = arch.length - 1;        // editable layers (the output is fixed)
  const inSpans = [];                          // matrix L's input-dim label (driven by arch[L-1])

  // "+" affordance on the arrow before matrix `at`; inserts a fresh hidden layer
  // there. No gap before the first matrix — inserting ahead of the input reads awkward.
  const insertGap = (at) => {
    const g = document.createElement("div");
    g.className = "flow-gap";
    const ar = document.createElement("span"); ar.className = "flow"; ar.innerHTML = SVG_ARROW; g.appendChild(ar);
    if (hiddenCount < MAX_LAYERS) {
      const b = document.createElement("button");
      b.className = "gap-add"; b.innerHTML = SVG_PLUS; b.title = "insert a layer here";
      b.onclick = () => { if (morphing) return; arch.splice(at, 0, 32); morphTo({ type: "add", at }); };
      g.appendChild(b);
    }
    els.layers.appendChild(g);
  };

  for (let L = 0; L < arch.length; L++) {
    if (L > 0) insertGap(L);                    // arrow + "+" between matrices -> insert at arch position L

    const rows = dims[L], cols = dims[L + 1];  // shape known from dims; pixels filled by drawHeatmaps
    const BASE = 260, CLAMP = 2.6;
    const a = Math.max(1 / CLAMP, Math.min(CLAMP, rows / cols));
    const cv = document.createElement("canvas");
    if (a >= 1) { cv.height = BASE; cv.width = Math.round(BASE / a); }
    else { cv.width = BASE; cv.height = Math.round(BASE * a); }
    layerCanvases[L] = cv;

    const wrap = document.createElement("figure");
    wrap.className = "layer-fig";
    wrap.appendChild(cv);

    const editable = L < arch.length - 1;      // hidden layer; its output width = arch[L]
    const name = L === 0 ? "input→h1" : (editable ? `h${L}→h${L + 1}` : `h${L}→out`);
    const cap = document.createElement("figcaption");
    cap.innerHTML = `<span class="lname">${name}</span>` +
      `<span class="ldim"><span class="d-in">${dims[L]}</span>×<span class="d-out${editable ? " edit" : ""}">${dims[L + 1]}</span></span>`;
    wrap.appendChild(cap);
    inSpans[L] = cap.querySelector(".d-in");
    const outSpan = cap.querySelector(".d-out");

    if (editable) {
      const range = document.createElement("input");
      range.type = "range"; range.min = MIN_W; range.max = MAX_W; range.value = arch[L];
      range.className = "layer-slider"; range.title = "drag to resize this layer";
      range.style.width = cv.width + "px";   // match the canvas so the figure isn't widened past it
      range.oninput = () => {
        arch[L] = +range.value;
        outSpan.textContent = range.value;                              // this matrix's output …
        if (inSpans[L + 1]) inSpans[L + 1].textContent = range.value;   // … is the next matrix's input
      };
      range.onchange = () => { if (!morphing) morphTo({ type: "resize" }); };
      wrap.appendChild(range);

      if (L > 0) {   // first hidden layer is anchored to the input — not removable
        const rm = document.createElement("button");
        rm.className = "layer-rm"; rm.innerHTML = SVG_X; rm.title = "remove this layer";
        rm.onclick = () => { if (morphing) return; arch.splice(L, 1); morphTo({ type: "remove", at: L }); };
        wrap.appendChild(rm);
      }
    }
    els.layers.appendChild(wrap);
  }
}

// Per-step: redraw the weight-field canvases only (DOM + controls left intact).
function drawHeatmaps(params) {
  for (let L = 0; L < arch.length; L++) {
    const cv = layerCanvases[L];
    if (!cv) continue;
    const t = params[`Dense_${L}.kernel`];
    let scale = 0;
    for (let i = 0; i < t.data.length; i++) scale = Math.max(scale, Math.abs(t.data[i]));
    drawHeatmap(cv, t, scale || 1);
  }
}

// ---------- accuracy curve ----------
function drawAccuracy() {
  const cv = els.accCanvas, ctx = cv.getContext("2d");
  const W = cv.width, Hh = cv.height, pad = 4;
  ctx.clearRect(0, 0, W, Hh);
  // gridline at 0.9
  ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1;
  for (const y of [0.25, 0.5, 0.75, 1.0]) {
    const py = Hh - pad - y * (Hh - 2 * pad);
    ctx.beginPath(); ctx.moveTo(pad, py); ctx.lineTo(W - pad, py); ctx.stroke();
  }
  if (accHistory.length < 2) return;
  const n = accHistory.length;
  ctx.strokeStyle = "#6c5cff"; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad + (i / (n - 1)) * (W - 2 * pad);
    const y = Hh - pad - accHistory[i] * (Hh - 2 * pad);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
  // glow dot
  const lx = W - pad, ly = Hh - pad - accHistory[n - 1] * (Hh - 2 * pad);
  ctx.fillStyle = "#6c5cff"; ctx.beginPath(); ctx.arc(lx, ly, 3, 0, 7); ctx.fill();
}

// ---------- MNIST sample predictions ----------
function renderDigits(params) {
  const xs = { data: mnist.x.data.subarray(0, DIGIT_SAMPLES * INPUT_DIM), shape: [DIGIT_SAMPLES, INPUT_DIM] };
  const logits = tasknetForward(params, xs, arch);
  const pred = argmaxRows(logits);
  const grid = els.digits;
  if (grid.children.length !== DIGIT_SAMPLES) {
    grid.innerHTML = "";
    for (let s = 0; s < DIGIT_SAMPLES; s++) {
      const c = document.createElement("div"); c.className = "digit";
      const cv = document.createElement("canvas"); cv.width = 28; cv.height = 28;
      const lab = document.createElement("span"); lab.className = "plabel";
      c.appendChild(cv); c.appendChild(lab); grid.appendChild(c);
      // draw the raw digit once
      const ctx = cv.getContext("2d"); const img = ctx.createImageData(28, 28);
      for (let p = 0; p < 784; p++) { const val = mnist.raw.data[s * 784 + p]; const o = p * 4; img.data[o] = img.data[o + 1] = img.data[o + 2] = val; img.data[o + 3] = 255; }
      ctx.putImageData(img, 0, 0);
    }
  }
  for (let s = 0; s < DIGIT_SAMPLES; s++) {
    const truth = mnist.labels.data[s] | 0;
    const c = grid.children[s], lab = c.querySelector(".plabel");
    lab.textContent = pred[s];
    c.classList.toggle("correct", pred[s] === truth);
    c.classList.toggle("wrong", pred[s] !== truth);
  }
}

function computeAccuracy(params) {
  const n = Math.min(ACC_N, mnist.labels.shape[0]);
  const xs = { data: mnist.x.data.subarray(0, n * INPUT_DIM), shape: [n, INPUT_DIM] };
  const pred = argmaxRows(tasknetForward(params, xs, arch));
  let correct = 0;
  for (let i = 0; i < n; i++) if (pred[i] === (mnist.labels.data[i] | 0)) correct++;
  return correct / n;
}

// ---------- arch rebuild / morph ----------
let morphing = false;

// Full reset (boot + "Reset weights" button): fresh random weights, counters cleared.
function rebuild() {
  const params = randInit(arch);
  const hp = initHiddenPosenc(arch);
  engine.setArch([...arch], params, hp, hp);
  steps = 0; accHistory = [];
  buildLayerField();
  refresh(params);
  syncStatus();
}

// Architecture edit: carry surviving weights + hidden states over; only new
// slots are re-initialized. Steps and the accuracy curve keep running.
async function morphTo(op) {
  if (morphing) return;
  morphing = true;
  try {
    const oldParams = await engine.readWeights();
    const oldHidden = await engine.readHidden();
    const map = archEditMap(engine.arch, arch, op);
    const { params, hidden, posenc } = carryOver(oldParams, oldHidden, arch, map, randInit);
    engine.setArch([...arch], params, hidden, posenc);
    buildLayerField();
    await refresh(params);
    syncStatus();
  } finally {
    morphing = false;
  }
}

async function refresh(paramsMaybe) {
  const params = paramsMaybe || await engine.readWeights();
  drawHeatmaps(params);
  const acc = computeAccuracy(params);
  accHistory.push(acc);
  if (accHistory.length > 400) accHistory.shift();
  drawAccuracy();
  renderDigits(params);
  els.acc.textContent = (acc * 100).toFixed(1) + "%";
}

function syncStatus() {
  els.steps.textContent = steps;
  els.ms.textContent = stepRate ? stepRate.toPrecision(3) + "/s" : "—";
  els.play.textContent = playing ? "Pause" : "Play";
  els.play.classList.toggle("on", playing);
}

// ---------- main loop ----------
async function loop() {
  if (playing && !morphing) {
    const interval = targetRate === Infinity ? 0 : 1000 / targetRate;   // ms between steps ("max" -> 0)
    if (!lastStepAt || performance.now() - lastStepAt >= interval) {
      const t = performance.now();
      if (lastStepAt) stepRate = 1000 / (t - lastStepAt);   // start-to-start period -> actual steps/sec
      lastStepAt = t;
      engine.step(1);                          // exactly one rule step...
      steps++;
      await refresh();                         // ...rendered every step (readWeights forces completion)
      syncStatus();
    }
  } else {
    lastStepAt = 0;                            // reset so the rate is measured fresh on resume
  }
  requestAnimationFrame(loop);
}

// ---------- boot ----------
async function boot() {
  els.layers = $("#layers"); els.accCanvas = $("#acc-curve"); els.digits = $("#digits");
  els.acc = $("#acc-val"); els.steps = $("#step-val"); els.ms = $("#ms-val");
  els.play = $("#play");

  if (!navigator.gpu) { $("#demo").classList.add("nogpu"); $("#nogpu-msg").hidden = false; return; }
  try {
    device = await createDevice();
  } catch (e) { $("#demo").classList.add("nogpu"); $("#nogpu-msg").hidden = false; $("#nogpu-msg").textContent = "Couldn't initialize WebGPU: " + e.message; return; }

  ruleBundle = await loadBundle("assets/rule");
  const mb = await loadBundle("assets/mnist");
  mnist = { x: mb.x, labels: mb.labels, raw: mb.raw };
  engine = new GPUEngine(device, ruleBundle);
  engine.prop = 0.8;   // stochastic 80%-of-weights per step — matches how the rule was trained

  rebuild();
  syncStatus();
  loop();

  $("#play").onclick = () => { playing = !playing; syncStatus(); };
  $("#step").onclick = async () => { if (morphing) return; engine.step(1); steps++; await refresh(); syncStatus(); };
  $("#reset").onclick = rebuild;
  $("#speed").oninput = (e) => {
    const v = +e.target.value, mx = +e.target.max;
    if (v >= mx) { targetRate = Infinity; $("#speed-val").textContent = "max"; }
    else { targetRate = v / 2; $("#speed-val").textContent = targetRate.toFixed(1) + "/s"; }   // v/2: 0.5 -> 10 steps/s
  };
}

boot();
