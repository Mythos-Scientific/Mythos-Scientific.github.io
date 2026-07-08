// js_vm.js — faithful JS mirror of sim.wgsl (the REPL_MIN VM), written line-for-line to correspond to the
// shader. Used as: (1) the validation reference (checked against Python vm_trace headless), (2) the CPU
// fallback for browsers without WebGPU, (3) the in-browser self-test reference for the real WGSL kernel.
// runVM(genome[128], epochs) runs ONE cell against 4 direction-neighbour tapes (init 0xEE = SENT, matching
// vm_trace) and returns [nbr0,nbr1,nbr2,nbr3]; FAR writes nbr[hs&3][h1] = tape[h0] — identical to the shader's
// per-cell FAR (the shader additionally maps hs&3 -> a physical grid neighbour, which is plain arithmetic).
export function runVM(genome, epochs = 60, cap = 256) {
  const bc = Uint8Array.from(genome);
  const SENT = 0xEE;
  const nbr = [0, 1, 2, 3].map(() => new Uint8Array(128).fill(SENT));
  let h0 = 0, h1 = 0, hs = 14;
  for (let ep = 0; ep < epochs; ep++) {
    let pc = 0, halted = false, turn = 0;
    while (!halted && turn < cap) {
      turn++;
      const op = bc[pc];
      if (op === 0x03) {                              // [
        if (bc[h0] === 0) {
          let d = 1, p = pc + 1;
          while (p < 128 && d > 0) { if (bc[p] === 0x03) d++; else if (bc[p] === 0x04) { d--; if (d === 0) break; } p++; }
          if (d !== 0) halted = true; else pc = p;
        }
      } else if (op === 0x04) {                       // ]
        if (bc[h0] !== 0) {
          let d = 1, p = pc - 1;
          while (p >= 0 && d > 0) { if (bc[p] === 0x04) d++; else if (bc[p] === 0x03) { d--; if (d === 0) break; } p--; }
          if (d !== 0) halted = true; else pc = p;
        }
      }
      else if (op === 0x83) h0 = (h0 + 1) & 127;
      else if (op === 0x84) h0 = (h0 + 127) & 127;
      else if (op === 0x85) h1 = (h1 + 1) & 127;
      else if (op === 0x86) h1 = (h1 + 127) & 127;
      else if (op === 0x87) bc[h0] = (bc[h0] + 1) & 255;
      else if (op === 0x88) bc[h0] = (bc[h0] + 255) & 255;
      else if (op === 0x89) bc[h1] = bc[h0];
      else if (op === 0x8A) bc[h0] = bc[h1];
      else if (op === 0x9A) nbr[hs & 3][h1] = bc[h0];  // FAR
      else if (op === 0x4C) hs = (hs + 1) & 255;
      else if (op === 0x4D) hs = (hs + 255) & 255;
      // else NOP
      if (!halted) { pc++; if (pc >= 128) halted = true; }
    }
  }
  return nbr;
}
