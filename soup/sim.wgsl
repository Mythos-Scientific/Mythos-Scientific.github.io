// Replicator-soup VM (WebGPU). 1 invocation = 1 cell. Faithful port of src/substrate.cu's execution model
// (private working copy + dirty merge-store, substrate.cu:2469/2699) with opt-in features via Params:
//   grid_local : 1 = 2-D grid-neighbour copy reach (diverse lineages); 0 = 1-D flat reach (monoculture)
//   ext_ops    : 1 = enable MATE/SCATTER/GATHER ops (0x50/51/52) on top of the 13-op REPL_MIN core
//   economy    : 1 = each op costs energy from a cost table; a broke cell goes dormant, regen refills it
//   mrate      : point-mutation rate (parts-per-65536 per cell per frame); 0 = pure
// Cell layout: STRIDE=129 u32/cell. [0..128) tape (1 byte/u32). [128] scalars pc|h0<<8|h1<<16|hs<<24.
// Energy lives in a SEPARATE buffer (ebuf), so the tape layout / coloring / readback are unchanged.

struct Params { n: u32, gw: u32, gh: u32, epochs: u32, cap: u32, seed: u32, mrate: u32, grid_local: u32, ext_ops: u32, economy: u32, regen: u32, };
@group(0) @binding(0) var<storage, read_write> cells: array<u32>;
@group(0) @binding(1) var<uniform> P: Params;
@group(0) @binding(2) var<storage, read_write> ebuf: array<u32>;

const STRIDE: u32 = 129u;
const TAPE: u32 = 128u;
const E_CAP: u32 = 40000u;

fn hash(x0: u32) -> u32 { var x = x0; x = x ^ (x >> 16u); x = x * 0x7feb352du; x = x ^ (x >> 15u); x = x * 0x846ca68bu; x = x ^ (x >> 16u); return x; }

fn nbrBase(cell: u32, x: i32, y: i32, dir: u32) -> u32 {   // 2-D grid neighbour, or 1-D flat neighbour
  if (P.grid_local == 1u) {
    var nx = x; var ny = y;
    if (dir == 0u) { nx = nx + 1; } else if (dir == 1u) { nx = nx - 1; } else if (dir == 2u) { ny = ny + 1; } else { ny = ny - 1; }
    nx = (nx + i32(P.gw)) % i32(P.gw); ny = (ny + i32(P.gh)) % i32(P.gh);
    return (u32(ny) * P.gw + u32(nx)) * STRIDE;
  }
  let fwd = (dir == 0u) || (dir == 2u);
  return ((cell + select(P.n - 1u, 1u, fwd)) % P.n) * STRIDE;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = gid.x;
  if (cell >= P.n) { return; }
  let base = cell * STRIDE;
  let x = i32(cell % P.gw);
  let y = i32(cell / P.gw);

  var energy = ebuf[cell];
  if (P.economy == 1u && energy == 0u) { ebuf[cell] = min(energy + P.regen, E_CAP); return; }  // dormant

  var s = cells[base + TAPE];
  var pc: u32 = s & 0xffu; var h0: u32 = (s >> 8u) & 0xffu; var h1: u32 = (s >> 16u) & 0xffu; var hs: u32 = (s >> 24u) & 0xffu;

  var t: array<u32, 128>;
  var dirty: array<u32, 4>;
  for (var i: u32 = 0u; i < 128u; i = i + 1u) { t[i] = cells[base + i] & 0xffu; }
  dirty[0] = 0u; dirty[1] = 0u; dirty[2] = 0u; dirty[3] = 0u;

  for (var ep: u32 = 0u; ep < P.epochs; ep = ep + 1u) {
    pc = 0u; var halted = false; var turn: u32 = 0u;
    loop {
      if (halted || turn >= P.cap) { break; }
      turn = turn + 1u;
      let op = t[pc];
      var ecost: u32 = 1u;
      if (op == 0x03u) { if (t[h0] == 0u) { var d: i32 = 1; var p: i32 = i32(pc) + 1; loop { if (p >= 128 || d <= 0) { break; } let k = t[u32(p)]; if (k == 0x03u) { d = d + 1; } else if (k == 0x04u) { d = d - 1; if (d == 0) { break; } } p = p + 1; } if (d != 0) { halted = true; } else { pc = u32(p); } } }
      else if (op == 0x04u) { if (t[h0] != 0u) { var d: i32 = 1; var p: i32 = i32(pc) - 1; loop { if (p < 0 || d <= 0) { break; } let k = t[u32(p)]; if (k == 0x04u) { d = d + 1; } else if (k == 0x03u) { d = d - 1; if (d == 0) { break; } } p = p - 1; } if (d != 0) { halted = true; } else { pc = u32(p); } } }
      else if (op == 0x83u) { h0 = (h0 + 1u) & 127u; }
      else if (op == 0x84u) { h0 = (h0 + 127u) & 127u; }
      else if (op == 0x85u) { h1 = (h1 + 1u) & 127u; }
      else if (op == 0x86u) { h1 = (h1 + 127u) & 127u; }
      else if (op == 0x87u) { t[h0] = (t[h0] + 1u) & 0xffu; dirty[h0 >> 5u] = dirty[h0 >> 5u] | (1u << (h0 & 31u)); }
      else if (op == 0x88u) { t[h0] = (t[h0] + 255u) & 0xffu; dirty[h0 >> 5u] = dirty[h0 >> 5u] | (1u << (h0 & 31u)); }
      else if (op == 0x89u) { t[h1] = t[h0]; dirty[h1 >> 5u] = dirty[h1 >> 5u] | (1u << (h1 & 31u)); }
      else if (op == 0x8au) { t[h0] = t[h1]; dirty[h0 >> 5u] = dirty[h0 >> 5u] | (1u << (h0 & 31u)); }
      else if (op == 0x9au) { let nb = nbrBase(cell, x, y, hs & 3u); cells[nb + h1] = t[h0]; ecost = 6u; }      // FAR (writer dst=h1)
      else if (op == 0x4cu) { hs = (hs + 1u) & 255u; }
      else if (op == 0x4du) { hs = (hs + 255u) & 255u; }
      else if (P.ext_ops == 1u && op == 0x50u) { let nb = nbrBase(cell, x, y, hs & 3u); cells[nb + h0] = t[h0]; ecost = 3u; }   // SCATTER (position-locked dst=h0)
      else if (P.ext_ops == 1u && op == 0x51u) { let nb = ((cell + 1u) % P.n) * STRIDE; cells[nb + h1] = t[h0]; ecost = 2u; }    // MATE (flat neighbour, cheap)
      else if (P.ext_ops == 1u && op == 0x52u) { let nb = nbrBase(cell, x, y, hs & 3u); t[h0] = cells[nb + h0] & 0xffu; dirty[h0 >> 5u] = dirty[h0 >> 5u] | (1u << (h0 & 31u)); ecost = 2u; }  // GATHER (read neighbour)
      // else: NOP

      if (P.economy == 1u) { if (energy < ecost) { halted = true; } else { energy = energy - ecost; } }
      if (!halted) { pc = pc + 1u; if (pc >= 128u) { halted = true; } }
    }
  }
  if (P.mrate > 0u) { let r = hash(cell * 2654435761u + P.seed); if ((r & 0xffffu) < P.mrate) { let r2 = hash(r); let pos = r2 & 127u; t[pos] = (r2 >> 8u) & 0xffu; dirty[pos >> 5u] = dirty[pos >> 5u] | (1u << (pos & 31u)); } }
  for (var i: u32 = 0u; i < 128u; i = i + 1u) { if (((dirty[i >> 5u] >> (i & 31u)) & 1u) == 1u) { cells[base + i] = t[i]; } }
  cells[base + TAPE] = (pc & 0xffu) | ((h0 & 0xffu) << 8u) | ((h1 & 0xffu) << 16u) | ((hs & 0xffu) << 24u);
  if (P.economy == 1u) { ebuf[cell] = min(energy + P.regen, E_CAP); }
}
