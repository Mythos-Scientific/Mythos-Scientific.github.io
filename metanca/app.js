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

const $ = (s) => document.querySelector(s);
const els = {};
let engine, mnist, ruleBundle, device;
let arch = [50, 30, OUTPUT_DIM];   // hidden... + output; input 784 implicit
let playing = false, steps = 0, speed = 8;  // speed 1..16 (higher = more steps/sec)
let accHistory = [];
let lastStepMs = 0;

// ---------- weight colormap: teal (neg) <- dark substrate -> coral (pos) ----------
const SUBSTRATE = [13, 16, 23];
const TEAL = [31, 158, 143];
const CORAL = [232, 98, 58];
function colormap(v, scale) {
  let t = Math.max(-1, Math.min(1, v / scale));
  const dir = t >= 0 ? CORAL : TEAL;
  const m = Math.pow(Math.abs(t), 0.8);
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
  dc.imageSmoothingEnabled = false;
  dc.clearRect(0, 0, canvas.width, canvas.height);
  // fit preserving aspect, centered
  const s = Math.min(canvas.width / outD, canvas.height / inD);
  const w = outD * s, h = inD * s;
  dc.drawImage(off, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
}

function renderHeatmaps(params) {
  els.layers.innerHTML = "";
  const dims = [INPUT_DIM, ...arch];
  for (let L = 0; L < arch.length; L++) {
    const t = params[`Dense_${L}.kernel`];
    let scale = 0;
    for (let i = 0; i < t.data.length; i++) scale = Math.max(scale, Math.abs(t.data[i]));
    scale = scale || 1;
    const wrap = document.createElement("figure");
    wrap.className = "layer-fig";
    const cv = document.createElement("canvas");
    const tall = t.shape[0] > 120;
    cv.width = 150; cv.height = tall ? 220 : 150;
    wrap.appendChild(cv);
    const cap = document.createElement("figcaption");
    cap.innerHTML = `<span class="lname">${L === 0 ? "input→h1" : (L === arch.length - 1 ? `h${L}→out` : `h${L}→h${L + 1}`)}</span><span class="ldim">${dims[L]}×${dims[L + 1]}</span>`;
    wrap.appendChild(cap);
    els.layers.appendChild(wrap);
    drawHeatmap(cv, t, scale);
    if (L < arch.length - 1) {
      const arrow = document.createElement("div");
      arrow.className = "flow"; arrow.textContent = "→";
      els.layers.appendChild(arrow);
    }
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
    await refresh(params);
    syncStatus();
  } finally {
    morphing = false;
  }
}

async function refresh(paramsMaybe) {
  const params = paramsMaybe || await engine.readWeights();
  renderHeatmaps(params);
  const acc = computeAccuracy(params);
  accHistory.push(acc);
  if (accHistory.length > 400) accHistory.shift();
  drawAccuracy();
  renderDigits(params);
  els.acc.textContent = (acc * 100).toFixed(1) + "%";
}

function syncStatus() {
  els.steps.textContent = steps;
  els.ms.textContent = lastStepMs ? lastStepMs.toFixed(1) + " ms/step" : "—";
  els.play.textContent = playing ? "Pause" : "Play";
  els.play.classList.toggle("on", playing);
}

// ---------- main loop ----------
let frameAcc = 0;
async function loop() {
  if (playing && !morphing) {
    frameAcc++;
    const stride = Math.max(1, 17 - speed);   // speed 16 -> every frame; speed 1 -> every 16 frames
    if (frameAcc >= stride) {
      frameAcc = 0;
      const t0 = performance.now();
      engine.step(1);                          // exactly one rule step...
      steps++;
      await refresh();                         // ...rendered every step (readWeights forces completion)
      lastStepMs = performance.now() - t0;
      syncStatus();
    }
  }
  requestAnimationFrame(loop);
}

// ---------- architecture editor ----------
function renderArchEditor() {
  const box = els.archEditor; box.innerHTML = "";
  const hidden = arch.slice(0, -1);
  const fixedIn = document.createElement("div"); fixedIn.className = "arch-node fixed"; fixedIn.innerHTML = `<b>784</b><small>input</small>`; box.appendChild(fixedIn);
  hidden.forEach((w, i) => {
    const node = document.createElement("div"); node.className = "arch-node";
    node.innerHTML = `<b>${w}</b><small>hidden ${i + 1}</small>`;
    const range = document.createElement("input"); range.type = "range"; range.min = MIN_W; range.max = MAX_W; range.value = w;
    range.oninput = () => { arch[i] = +range.value; node.querySelector("b").textContent = range.value; };
    range.onchange = () => morphTo({ type: "resize" });
    node.appendChild(range);
    if (hidden.length > 1) {
      const rm = document.createElement("button"); rm.className = "rm"; rm.textContent = "×"; rm.title = "remove layer";
      rm.onclick = () => { arch.splice(i, 1); renderArchEditor(); morphTo({ type: "remove", at: i }); };
      node.appendChild(rm);
    }
    box.appendChild(node);
  });
  if (hidden.length < MAX_LAYERS) {
    const add = document.createElement("button"); add.className = "arch-add"; add.textContent = "+ layer";
    add.onclick = () => {
      const at = arch.length - 1;              // new layer sits just before the output layer
      arch.splice(at, 0, 32); renderArchEditor(); morphTo({ type: "add", at });
    };
    box.appendChild(add);
  }
  const fixedOut = document.createElement("div"); fixedOut.className = "arch-node fixed"; fixedOut.innerHTML = `<b>10</b><small>digits</small>`; box.appendChild(fixedOut);
}

// ---------- boot ----------
async function boot() {
  els.layers = $("#layers"); els.accCanvas = $("#acc-curve"); els.digits = $("#digits");
  els.acc = $("#acc-val"); els.steps = $("#step-val"); els.ms = $("#ms-val");
  els.play = $("#play"); els.archEditor = $("#arch-editor");

  if (!navigator.gpu) { $("#demo").classList.add("nogpu"); $("#nogpu-msg").hidden = false; return; }
  try {
    device = await createDevice();
  } catch (e) { $("#demo").classList.add("nogpu"); $("#nogpu-msg").hidden = false; $("#nogpu-msg").textContent = "Couldn't initialize WebGPU: " + e.message; return; }

  ruleBundle = await loadBundle("assets/rule");
  const mb = await loadBundle("assets/mnist");
  mnist = { x: mb.x, labels: mb.labels, raw: mb.raw };
  engine = new GPUEngine(device, ruleBundle);
  engine.prop = 0.8;   // stochastic 80%-of-weights per step — matches how the rule was trained

  renderArchEditor();
  rebuild();
  syncStatus();
  loop();

  $("#play").onclick = () => { playing = !playing; syncStatus(); };
  $("#step").onclick = async () => { if (morphing) return; engine.step(1); steps++; await refresh(); syncStatus(); };
  $("#reset").onclick = rebuild;
  $("#speed").oninput = (e) => { speed = +e.target.value; $("#speed-val").textContent = "lvl " + speed; };
}

boot();
