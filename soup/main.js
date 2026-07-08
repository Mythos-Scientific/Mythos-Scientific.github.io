// WebGPU replicator soup — milestone 1 pipeline: sim (VM) -> color (LSH) -> render, live on a canvas.
// Includes an in-browser self-test that runs the real WGSL VM on the GPU and diffs it against the validated
// JS mirror (js_vm.js). Auto-sizing/perf tuning is milestone 5; this uses a fixed grid.
import { runVM } from "./js_vm.js";

const STRIDE = 129, TAPE = 128;
const $ = (id) => document.getElementById(id);
const setStatus = (m, err = false) => { const s = $("status"); s.textContent = m; s.style.color = err ? "#f66" : "#5f6f7e"; };

async function loadWGSL(name) { return await (await fetch(new URL(name, import.meta.url))).text(); }

// Compressed length via the browser's CompressionStream (deflate). Deflate is a close proxy for the CL
// paper's brotli — the HOE *trajectory* (0 -> a few bits) is the same; absolute values run a touch lower.
async function deflateLen(bytes) {
  const cs = new CompressionStream("deflate-raw");
  const w = cs.writable.getWriter(); w.write(bytes); w.close();
  const r = cs.readable.getReader(); let n = 0;
  for (;;) { const { done, value } = await r.read(); if (done) break; n += value.length; }
  return n;
}

function randomCells(N) {
  const c = new Uint32Array(N * STRIDE);
  let s = ((Math.random() * 4294967296) >>> 0) || 1;                   // xorshift32 (crypto caps at 64KB/call)
  const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return s >>> 0; };
  for (let cell = 0; cell < N; cell++) {
    const b = cell * STRIDE;
    for (let i = 0; i < TAPE; i++) c[b + i] = rnd() & 0xff;            // random tape bytes
    c[b + TAPE] = (14 << 24);                                          // pc=0 h0=0 h1=0 hs=14
  }
  return c;
}

async function main() {
  if (!navigator.gpu) { setStatus("WebGPU not available in this browser. Try Chrome/Edge (stable) or Firefox/Safari (recent).", true); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { setStatus("No WebGPU adapter (no compatible GPU).", true); return; }
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => setStatus("GPU error: " + e.error.message, true));

  const [simSrc, colorSrc, renderSrc] = await Promise.all([loadWGSL("sim.wgsl"), loadWGSL("color.wgsl"), loadWGSL("render.wgsl")]);

  const canvas = $("cv");
  const ctx = canvas.getContext("webgpu");
  const fmt = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: fmt, alphaMode: "opaque" });

  const simMod = device.createShaderModule({ code: simSrc });
  const colorMod = device.createShaderModule({ code: colorSrc });
  const renderMod = device.createShaderModule({ code: renderSrc });
  const simPipe = device.createComputePipeline({ layout: "auto", compute: { module: simMod, entryPoint: "main" } });
  const colorPipe = device.createComputePipeline({ layout: "auto", compute: { module: colorMod, entryPoint: "main" } });
  const renderPipe = device.createRenderPipeline({ layout: "auto", vertex: { module: renderMod, entryPoint: "vs" }, fragment: { module: renderMod, entryPoint: "fs", targets: [{ format: fmt }] }, primitive: { topology: "triangle-list" } });

  const EPOCHS_PER = 2;

  // ---- auto-size the grid to this GPU: benchmark throughput, pick the biggest grid that still evolves
  //      fast enough (smaller grid = more epochs/sec on the same GPU, so it reaches the ~5k-epoch takeoff sooner)
  setStatus("benchmarking your GPU…");
  async function benchThroughput() {
    const Nb = 128 * 128, K = 24;
    const buf = device.createBuffer({ size: Nb * STRIDE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const u = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const eb = device.createBuffer({ size: Nb * 4, usage: GPUBufferUsage.STORAGE });
    device.queue.writeBuffer(u, 0, new Uint32Array([Nb, 128, 128, EPOCHS_PER, 256, 1, 0, 1, 0, 0, 0, 0]));  // grid_local=1, rest off
    device.queue.writeBuffer(buf, 0, randomCells(Nb));
    const bind = device.createBindGroup({ layout: simPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }, { binding: 1, resource: { buffer: u } }, { binding: 2, resource: { buffer: eb } }] });
    const run = (k) => { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(simPipe); p.setBindGroup(0, bind); for (let s = 0; s < k; s++) p.dispatchWorkgroups(Math.ceil(Nb / 64)); p.end(); device.queue.submit([e.finish()]); };
    run(4); await device.queue.onSubmittedWorkDone();                 // warm up
    const s0 = performance.now(); run(K); await device.queue.onSubmittedWorkDone();
    const ms = performance.now() - s0;
    buf.destroy(); u.destroy(); eb.destroy();
    return (Nb * K * EPOCHS_PER) / Math.max(ms, 0.1);                 // cell-epochs / ms
  }
  const cellEpochsPerSec = (await benchThroughput()) * 1000;
  const TARGET_EPS = 250;                                             // bias toward a bigger grid (more reliable takeoff) vs pure speed
  const maxSideBySpeed = Math.sqrt(cellEpochsPerSec / TARGET_EPS);
  const memN = Math.floor(device.limits.maxStorageBufferBindingSize / (STRIDE * 4));
  let SIDE = 96;
  for (const s of [96, 128, 160, 192, 224, 256]) { if (s <= maxSideBySpeed && s * s <= memN) SIDE = s; }
  const GW = SIDE, GH = SIDE, N = GW * GH;
  canvas.width = GW; canvas.height = GH;   // internal resolution = grid; CSS controls the display size

  const cellBuf = device.createBuffer({ size: N * STRIDE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const hoeRead = device.createBuffer({ size: N * STRIDE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const simU = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const colU = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const energyBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const energyRead = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let STEPS = 8;    // sim dispatches per frame — auto-tuned each frame to run as fast as stays smooth
  let mrate = 120;  // mutation "temperature" (parts-per-65536 per cell per frame); 0 = pure races
  let seed = 1;
  let gridLocal = 1, extOps = 0, economy = 0, regen = 400;   // feature toggles (default = the plain emergent demo)
  const writeSimParams = () => device.queue.writeBuffer(simU, 0, new Uint32Array([N, GW, GH, EPOCHS_PER, 256, seed, mrate, gridLocal, extOps, economy, regen, 0]));
  const fillEnergy = () => device.queue.writeBuffer(energyBuf, 0, new Uint32Array(N).fill(40000));
  writeSimParams();
  fillEnergy();
  device.queue.writeBuffer(colU, 0, new Uint32Array([N, GW, GH, 0]));

  const tex = device.createTexture({ size: [GW, GH], format: "rgba8unorm", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
  const sampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
  const colorState = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });  // per-cell smoothed color (EMA)

  const simBind = device.createBindGroup({ layout: simPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: cellBuf } }, { binding: 1, resource: { buffer: simU } }, { binding: 2, resource: { buffer: energyBuf } }] });
  const colBind = device.createBindGroup({ layout: colorPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: cellBuf } }, { binding: 1, resource: { buffer: colU } }, { binding: 2, resource: tex.createView() }, { binding: 3, resource: { buffer: colorState } }] });
  const renBind = device.createBindGroup({ layout: renderPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: sampler }] });

  const reset = () => { device.queue.writeBuffer(cellBuf, 0, randomCells(N)); fillEnergy(); };
  reset();

  // Read a sample of the soup back to the CPU each tick and compute the metrics we track in the CUDA runs:
  // HOE (h0 - compressed bpb), diversity (unique genomes), kin (spatial relatedness), copiers (executed),
  // and n_alive (economy only). Also caches the whole grid for click-to-inspect.
  const SAMP = Math.min(2048, N), sstride = Math.max(1, Math.floor(N / SAMP));
  const lastData = new Uint32Array(N * STRIDE);
  let lastValid = false;
  async function measureStats() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(cellBuf, 0, hoeRead, 0, N * STRIDE * 4);
    device.queue.submit([enc.finish()]);
    await hoeRead.mapAsync(GPUMapMode.READ);
    lastData.set(new Uint32Array(hoeRead.getMappedRange())); lastValid = true;
    hoeRead.unmap();
    const bytes = new Uint8Array(SAMP * TAPE);
    for (let s = 0; s < SAMP; s++) { const b = s * sstride * STRIDE; for (let i = 0; i < TAPE; i++) bytes[s * TAPE + i] = lastData[b + i] & 0xff; }
    const hist = new Float64Array(256); for (let i = 0; i < bytes.length; i++) hist[bytes[i]]++;
    let h0 = 0; for (let b = 0; b < 256; b++) if (hist[b]) { const p = hist[b] / bytes.length; h0 -= p * Math.log2(p); }
    const hoe = h0 - (await deflateLen(bytes)) * 8 / bytes.length;
    const seen = new Set();                                          // diversity: distinct genomes / sample
    for (let s = 0; s < SAMP; s++) { const b = s * sstride * STRIDE; let h = 2166136261; for (let i = 0; i < TAPE; i++) { h ^= (lastData[b + i] & 0xff); h = Math.imul(h, 16777619); } seen.add(h >>> 0); }
    const unique = seen.size / SAMP;
    let kinSum = 0, kinN = 0, rndSum = 0, rndN = 0;                  // kin: E-neighbour byte-match minus random-pair
    for (let s = 0; s < SAMP; s++) {
      const c = s * sstride; if (c % GW === GW - 1) continue;
      const a = c * STRIDE, bb = (c + 1) * STRIDE; let m = 0; for (let i = 0; i < TAPE; i++) if ((lastData[a + i] & 0xff) === (lastData[bb + i] & 0xff)) m++;
      kinSum += m / TAPE; kinN++;
      const r1 = (Math.floor(Math.random() * N)) * STRIDE, r2 = (Math.floor(Math.random() * N)) * STRIDE; let m2 = 0; for (let i = 0; i < TAPE; i++) if ((lastData[r1 + i] & 0xff) === (lastData[r2 + i] & 0xff)) m2++;
      rndSum += m2 / TAPE; rndN++;
    }
    const kin = (kinN ? kinSum / kinN : 0) - (rndN ? rndSum / rndN : 0);
    // functional copiers: EXECUTE a sample and check it copies a real fraction of itself into a neighbour
    // (like vm_trace) — NOT just "contains a FAR byte", which saturates at ~100% and means nothing.
    let faith = 0, lossy = 0, part = 0, csamp = Math.min(256, N), cstr = Math.max(1, Math.floor(N / csamp));
    for (let s = 0; s < csamp; s++) {
      const b = s * cstr * STRIDE, g = new Uint8Array(TAPE); for (let i = 0; i < TAPE; i++) g[i] = lastData[b + i] & 0xff;
      let act = 0; for (let i = 0; i < TAPE; i++) if (ACTIVE_REPL.has(g[i])) act++;
      if (act < 3) continue;
      const nbr = runVM(g, 40); let best = 0;
      for (let d = 0; d < 4; d++) { let m = 0; for (let i = 0; i < TAPE; i++) if (ACTIVE_REPL.has(g[i]) && nbr[d][i] === g[i]) m++; if (m / act > best) best = m / act; }
      if (best >= 0.9) faith++; else if (best >= 0.6) lossy++; else if (best >= 0.25) part++;   // same thresholds as the click-inspect verdict
    }
    const faithful = faith / csamp, lossyC = lossy / csamp, partialC = part / csamp;
    let energyPct = 1;                                              // average battery level (drops as busy copiers drain it)
    if (economy) {
      const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(energyBuf, 0, energyRead, 0, N * 4); device.queue.submit([e2.finish()]);
      await energyRead.mapAsync(GPUMapMode.READ);
      const ed = new Uint32Array(energyRead.getMappedRange()); let sum = 0; for (let i = 0; i < N; i++) sum += ed[i]; energyRead.unmap();
      energyPct = sum / N / 40000;
    }
    return { hoe, unique, kin, faithful, lossyC, partialC, energyPct };
  }

  let paused = false, frames = 0, t0 = performance.now(), totalEpochs = 0, lastT = performance.now();
  function frame() {
    if (!paused) {
      const now = performance.now(), fdelta = now - lastT; lastT = now;
      if (fdelta < 22 && STEPS < 64) STEPS++;                        // headroom -> go faster
      else if (fdelta > 34 && STEPS > 1) STEPS--;                    // janky -> ease off (keeps it smooth)
      seed = (seed + 1) >>> 0;
      device.queue.writeBuffer(simU, 20, new Uint32Array([seed]));   // vary the mutation RNG each frame
      const enc = device.createCommandEncoder();
      const sp = enc.beginComputePass();
      sp.setPipeline(simPipe); sp.setBindGroup(0, simBind);
      for (let s = 0; s < STEPS; s++) sp.dispatchWorkgroups(Math.ceil(N / 64));
      sp.end();
      totalEpochs += STEPS * EPOCHS_PER;
      const cp = enc.beginComputePass();
      cp.setPipeline(colorPipe); cp.setBindGroup(0, colBind);
      cp.dispatchWorkgroups(Math.ceil(GW / 8), Math.ceil(GH / 8));
      cp.end();
      const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
      rp.setPipeline(renderPipe); rp.setBindGroup(0, renBind); rp.draw(3); rp.end();
      device.queue.submit([enc.finish()]);
      frames++;
      const dt = now - t0;
      if (dt > 500) { $("fps").textContent = `${(frames / dt * 1000).toFixed(0)} fps · ${GW}×${GH} · speed ${STEPS}× · ~${(totalEpochs / 1000).toFixed(1)}k epochs`; frames = 0; t0 = now; }
    } else { lastT = performance.now(); }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  setStatus(`running — auto-sized to ${GW}×${GH} = ${(N / 1e3).toFixed(0)}k cells (${adapter.info?.description || "GPU"})`);

  $("reset").onclick = () => { reset(); totalEpochs = 0; sel = null; drawMarker(); };
  $("pause").onclick = () => { paused = !paused; $("pause").textContent = paused ? "play" : "pause"; };
  $("selftest").onclick = () => selfTest(device, simMod);
  const mut = $("mut"); if (mut) mut.oninput = (e) => { mrate = +e.target.value; if ($("mutval")) $("mutval").textContent = mrate; writeSimParams(); };
  if ($("mutval")) $("mutval").textContent = mrate;
  const bindToggle = (id, set) => { const el = $(id); if (el) el.onchange = (e) => { set(e.target.checked ? 1 : 0); writeSimParams(); }; };
  bindToggle("t_grid", (v) => gridLocal = v);
  bindToggle("t_ext", (v) => extOps = v);
  bindToggle("t_econ", (v) => { economy = v; fillEnergy(); });

  const overlay = $("overlay"), octx = overlay ? overlay.getContext("2d") : null;
  let sel = null;
  function drawMarker() {                              // highlight the selected cell (box + crosshair)
    if (!octx) return;
    if (overlay.width !== overlay.clientWidth) overlay.width = overlay.clientWidth;
    if (overlay.height !== overlay.clientHeight) overlay.height = overlay.clientHeight;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (!sel) return;
    const cw = overlay.width / GW, ch = overlay.height / GH, px = (sel.cx + 0.5) * cw, py = (sel.cy + 0.5) * ch;
    octx.beginPath(); octx.moveTo(px, 0); octx.lineTo(px, overlay.height); octx.moveTo(0, py); octx.lineTo(overlay.width, py);
    octx.strokeStyle = "rgba(0,0,0,0.5)"; octx.lineWidth = 3; octx.stroke();     // dark backing for contrast
    octx.strokeStyle = "rgba(255,255,255,0.85)"; octx.lineWidth = 1; octx.stroke();
    const bs = Math.max(cw, ch, 12);
    octx.lineWidth = 3; octx.strokeStyle = "rgba(0,0,0,0.85)"; octx.strokeRect(px - bs / 2, py - bs / 2, bs, bs);
    octx.lineWidth = 1.5; octx.strokeStyle = "#fff"; octx.strokeRect(px - bs / 2, py - bs / 2, bs, bs);
  }
  addEventListener("resize", drawMarker);
  canvas.addEventListener("click", (ev) => {          // tap a cell -> mark it + disassemble its program
    if (!lastValid) return;
    const r = canvas.getBoundingClientRect();
    const cx = Math.floor((ev.clientX - r.left) / r.width * GW), cy = Math.floor((ev.clientY - r.top) / r.height * GH);
    if (cx < 0 || cy < 0 || cx >= GW || cy >= GH) return;
    const b = (cy * GW + cx) * STRIDE, g = new Uint8Array(TAPE);
    for (let i = 0; i < TAPE; i++) g[i] = lastData[b + i] & 0xff;
    sel = { cx, cy }; drawMarker();
    showInspect(cx, cy, g, extOps);
  });

  const spark = $("spark"), sctx = spark ? spark.getContext("2d") : null, hoeHist = [];
  (async function statsLoop() {
    for (;;) {
      if (!paused) {
        try {
          const m = await measureStats();
          const el = $("hoe"); el.textContent = m.hoe.toFixed(2); el.style.color = m.hoe > 1.5 ? "#f5b642" : m.hoe > 0.5 ? "#c99a4e" : "#5f6f7e";
          setTxt("m_div", (m.unique * 100).toFixed(0) + "%");
          setTxt("m_kin", m.kin.toFixed(3));
          setTxt("m_faith", (m.faithful * 100).toFixed(0) + "%");
          setTxt("m_lossy", (m.lossyC * 100).toFixed(0) + "%");
          setTxt("m_partial", (m.partialC * 100).toFixed(0) + "%");
          setTxt("m_energy", economy ? (m.energyPct * 100).toFixed(0) + "%" : "—");
          hoeHist.push(m.hoe); if (hoeHist.length > 160) hoeHist.shift();
          drawSpark(sctx, spark, hoeHist);
        } catch (e) { /* transient map contention — skip this tick */ }
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
  })();
}

function setTxt(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }

function drawSpark(ctx, cv, hist) {
  if (!ctx) return;
  const W = cv.width, H = cv.height, maxv = 3.2;
  ctx.fillStyle = "#0a0f15"; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#1a2430"; ctx.lineWidth = 1; ctx.beginPath();
  for (const y of [1, 2, 3]) { const yy = H - (y / maxv) * H; ctx.moveTo(0, yy); ctx.lineTo(W, yy); } ctx.stroke();
  if (hist.length < 2) return;
  ctx.strokeStyle = "#f5b642"; ctx.lineWidth = 1.5; ctx.beginPath();
  for (let i = 0; i < hist.length; i++) { const xx = i / (hist.length - 1) * W, yy = H - Math.min(hist[i], maxv) / maxv * H; i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); }
  ctx.stroke();
}

// disassemble a genome to its active-op string + a copier verdict (mirrors vm_trace).
const OPNAMES = { 0x83: "h0+", 0x84: "h0-", 0x85: "h1+", 0x86: "h1-", 0x87: "B+", 0x88: "B-", 0x89: "C01", 0x8A: "C10", 0x03: "[", 0x04: "]", 0x9A: "FAR", 0x4C: "S+", 0x4D: "S-", 0x50: "SCAT", 0x51: "MATE", 0x52: "GATH" };
const ACTIVE_REPL = new Set([0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x03, 0x04, 0x9A, 0x4C, 0x4D]);  // ops the VM (runVM) executes
function showInspect(cx, cy, g, extOps) {
  const active = new Set(ACTIVE_REPL); if (extOps) { active.add(0x50); active.add(0x51); active.add(0x52); }  // ext ops are live only when enabled
  const ops = []; for (let i = 0; i < g.length; i++) if (active.has(g[i])) ops.push(OPNAMES[g[i]]);
  const nbr = runVM(g, 60);                             // runVM models the 13 REPL_MIN ops only
  let act = 0; for (let i = 0; i < g.length; i++) if (ACTIVE_REPL.has(g[i])) act++;
  let best = 0; for (let d = 0; d < 4; d++) { let m = 0; for (let i = 0; i < 128; i++) if (ACTIVE_REPL.has(g[i]) && nbr[d][i] === g[i]) m++; best = Math.max(best, act ? m / act : 0); }
  const far = g.includes(0x9A);
  const verdict = best >= 0.9 ? "faithful self-copier" : best >= 0.6 ? "copier (lossy)" : best >= 0.25 ? "partial copier" : far ? "writes to neighbour, ~no transfer" : "non-spreader";
  const hasExt = extOps && (g.includes(0x50) || g.includes(0x51) || g.includes(0x52));
  const note = hasExt ? ` <span style="color:#c96">(verdict is REPL_MIN-based; MATE/SCAT/GATH not modeled)</span>` : "";
  const panel = document.getElementById("inspect");
  if (panel) { panel.style.color = ""; panel.innerHTML = `<b>cell (${cx},${cy})</b> — <span style="color:#f5b642">${verdict}</span> · ${act} active ops · copy fidelity ${(best * 100).toFixed(0)}%${note}<br><span style="color:#7a8391">${ops.join(" ") || "(no active ops — inert)"}</span>`; }
}

// In-browser check: run the real WGSL VM on the GPU and diff vs the JS mirror on random genomes.
async function selfTest(device, simMod) {
  const btn = document.getElementById("selftest"), panel = document.getElementById("inspect");
  const say = (msg, err) => { setStatus(msg, err); if (panel) { panel.innerHTML = msg; panel.style.color = err ? "#f66" : "#f5b642"; } };
  if (btn) btn.textContent = "testing…";
  say("self-test: running the real WGSL VM on your GPU…");
  try {
  const GW = 5, GH = 5, N = 25, CENTER = 12, NBR = [13, 11, 17, 7]; // E,W,S,N
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module: simMod, entryPoint: "main" } });
  const buf = device.createBuffer({ size: N * STRIDE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const uni = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const eb = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE });
  const rd = device.createBuffer({ size: N * STRIDE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const EP = 60;
  device.queue.writeBuffer(uni, 0, new Uint32Array([N, GW, GH, EP, 256, 0, 0, 1, 0, 0, 0, 0]));  // grid_local=1, no mutation/econ
  const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }, { binding: 1, resource: { buffer: uni } }, { binding: 2, resource: { buffer: eb } }] });

  let pass = 0, fail = 0;
  for (let g = 0; g < 12; g++) {
    const genome = new Uint8Array(TAPE); crypto.getRandomValues(genome);
    const cells = new Uint32Array(N * STRIDE);
    for (let c = 0; c < N; c++) { const b = c * STRIDE; for (let i = 0; i < TAPE; i++) cells[b + i] = 0xEE; cells[b + TAPE] = (14 << 24); }
    for (let i = 0; i < TAPE; i++) cells[CENTER * STRIDE + i] = genome[i];
    device.queue.writeBuffer(buf, 0, cells);
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bind); p.dispatchWorkgroups(1); p.end();
    enc.copyBufferToBuffer(buf, 0, rd, 0, N * STRIDE * 4);
    device.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(rd.getMappedRange().slice(0)); rd.unmap();
    const gpu = NBR.map((nc) => { const b = nc * STRIDE; const t = new Uint8Array(TAPE); for (let i = 0; i < TAPE; i++) t[i] = data[b + i] & 0xff; return t; });
    const ref = runVM(genome, EP);
    let ok = true;
    for (let d = 0; d < 4; d++) for (let i = 0; i < TAPE; i++) if (gpu[d][i] !== ref[d][i]) ok = false;
    ok ? pass++ : fail++;
  }
  say(`self-test: WGSL VM vs JS mirror — ${pass}/12 genomes match${fail ? ` (${fail} FAIL)` : " ✓ the real shader is bit-exact on your GPU"}`, fail > 0);
  } catch (e) {
    say("self-test error: " + (e && e.message ? e.message : e), true);
  } finally {
    if (btn) btn.textContent = "self-test VM";
  }
}

main().catch((e) => setStatus("fatal: " + e.message, true));
