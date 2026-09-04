# K-quant dot products, ggml-structured, with Q8_K (production since 3.5) and Q8_0-grain (`--q8-fine`) activations

Source read: ggml `ggml-quants.c` (`quantize_row_q8_K_ref`, `nearest_int`) and
`ggml-cpu/arch/x86/quants.c` (`ggml_vec_dot_q4_K_q8_K`, `q5_K`, `q6_K`, AVX2 branches) in the
`models/llama.cpp` checkout. Engine files: `crates/core/src/kernels/dot.rs` + `avx2.rs` (the production
K-quant kernels, Q8_0-grain activations), `q8.rs` (activation row with the derived per-block data), `q8k.rs` +
`kdot.rs` (ggml's Q8_K row and kernels, opt-in), `matvec.rs` (row partition, `ActVec`, fused and batched
forms), `pool.rs` (persistent workers). Numbers: `docs/data/membw.txt`, `docs/data/kernels_bench.txt`.

**Outcome in one paragraph.** ggml's structure (exact integer inner loops, 8 f32 lanes accumulated across the
row, one reduction per row, mins folded through activation sums) is what makes the kernels memory-bound; ggml's
Q8_K activation format (one scale per 256 values) is not required for that and turned out 2.5 x noisier than
the Q8_0 rows Phase 2 used, beyond the frozen rule-5 ceilings at layer 0 (`docs/payload-vs-doc.md` finding 37).
Phase 3 therefore kept Q8_0 activations (per-32 f16 scales) under the same lane structure, with one f32
factor per sub-block instead of one per super-block, and left the Q8_K row and kernels in the tree as an
opt-in; both are benchmarked side by side.

**Phase 3.5 decision (2026-09-04).** Rule 5 was amended: the frozen ceilings are per activation format, each
from that format's own 64-layer measurement (`tests/fixtures/q8k_ceilings.json`, `q8_ceilings.json`;
`tests/real_layers.rs`, `AQUEDUCT_ACT=q8k|q8_0`). Measured over all 64 layers with Q8_K activations
(`docs/data/phase3_5_parity_q8k_0_63.log`), the Q8_K path's last-position logits sit 0.362 / 0.281 / 0.357
(capital / fib / sentence) from the f32 reference, argmax 3/3, top-10 10 / 10 / 9, inside the bf16-vs-GGUF
noise floor of 0.653 / 0.516 / 0.679 (finding 35), and 0.349 / 0.226 / 0.392 from llama.cpp, which quantises
to Q8_K itself (argmax 3/3, top-10 9 / 10 / 9). That meets the promotion criterion, so **Q8_K is the
production activation for K-quant weights** and the Q8_0-grain kernels are the `--q8-fine` opt-in
(`matvec::set_q8_fine`); Q8_0 / Q4_0 weights keep Q8_0 rows. Per layer, Q8_K's relative error is 2 to 4 x
Q8_0's (worst 5.3e-2 vs 1.6e-2, layer 63 `sentence`), and on `sentence` the argmax matches the reference at
37 of the 39 prompt positions (39/39 with Q8_0; both flips are inside the noise floor, `tests/phase3_e2e.rs`).

## Why the Phase 2 kernels were compute-bound

Phase 2b paired K-quant weights with Q8_0 activations (32-value blocks, f16 scale) and, per 32-element
sub-block, did an exact integer dot, one horizontal reduction to a scalar, and a scalar f32 expression
`d_x * ((d * sc) * isum - (dmin * m) * xsum)`. That is 8 horizontal reductions and 8 serial f32 chains per
256 weights; the kernels reached a fifth of memory bandwidth (`docs/data/kernels_bench.txt`, Phase 2b
section). ggml's layout of the same arithmetic does one reduction per row.

## Q8_K activations (`q8k.rs`)

`block_q8_K` = 256 values, one **f32** scale `d`, 256 int8 codes, and `bsums[16]`, the int16 sums of each
16-code group. Port of `quantize_row_q8_K_ref`, bit-exact against a Python port of the same function
(`tools/fixtures/gen_q8k.py`; gguf-py has no Q8_K):

```
max    = the signed element with the largest |x|   (first on ties: strict > scan)
iscale = -127 / max                                  (f32)
q      = min(127, nearest_int(iscale * x))            nearest_int = round half to even (f32::round_ties_even)
d      = 1 / iscale                                   (f32; carries the sign of -max)
bsums[g] = sum(q[16g .. 16g+16])
```

An all-zero block stores `d = 0`. Blocks whose largest value is subnormal overflow `iscale` in ggml too and
are outside the contract (as for Q8_0). The engine also keeps `q8s[s] = bsums[2s] + bsums[2s+1]` (the 8
sub-block sums), derived once per activation so the min term below is one `madd`; it is not part of the ggml
layout. The AVX2 quantiser takes the abs-max with vector max, the sign from the first element whose |x| equals
it (`cmp` + `movemask` + trailing zeros), converts with `cvtps_epi32` (round to nearest even, the same
rounding as `nearest_int`), clamps with `min_epi32(127)` and packs; it is bit-identical to the scalar port.

## The integer math

For a super-block of 256 weights with per-sub-block 6-bit scales `sc[s]` and mins `m[s]` (Q4_K, Q5_K:
`w = d * sc[s] * q - dmin * m[s]`) or int8 group scales `sc[g]` (Q6_K: `w = d * sc[g] * (q - 32)`), and
activations `x = d_x * q_x`:

```
Q4_K, Q5_K:  sum_j w_j x_j = d_x * d    * sum_s sc[s] * sum_{j in s} q_j q_x,j          (main, exact i32)
                           - d_x * dmin * sum_s m[s]  * sum_{j in s} q_x,j              (mins, exact i32, via q8s)
Q6_K:        sum_j w_j x_j = d_x * d * ( sum_g sc[g] * sum_{j in g} q_j q_x,j  -  32 * sum_g sc[g] * bsums[g] )
```

Every inner sum is integer and exact; the f32 scales are applied **once per super-block**, and the row is
reduced **once at the end**. Bounds (no saturation anywhere): `maddubs` pairs are at most
`2 * 15 * 127 = 3810` (Q4_K), `2 * 31 * 127 = 7874` (Q5_K), `2 * 63 * 127 = 16002` (Q6_K), all below
32767; `madd` with a 6-bit scale gives at most `2 * 63 * 3810` per lane per sub-block, so a super-block's
lane stays below 2^24 for Q4_K/Q5_K (exact f32 conversion) and below 2^25 for Q6_K with hostile int8 scales
(`cvtepi32_ps` rounds to nearest, identically in both paths).

## Lane structure and summation order (both paths, bit-identical)

The vector kernel processes a super-block in four 64-element chunks. `maddubs(q, q8)` gives 16 i16 lanes of
element pairs, `madd(scale, .)` gives 8 i32 lanes of element quads, so lane `i` of the running `sumi`
accumulates the elements whose position inside each 32-element sub-block is `4i .. 4i+4`, over all
sub-blocks of the super-block, each multiplied by its sub-block scale as an integer. The min term uses
`madd(mins, q8s)`: 4 i32 lanes, lane `i` holding sub-blocks `2i` and `2i+1`. Q6_K folds the -32 offset with
`slli_epi32(madd(bsums, scales), 5)`: 8 lanes, lane `i` holding groups `2i`, `2i+1`, subtracted from `sumi`.

Per super-block: `acc[i] += (d_x * d) * f32(sumi[i])` for the 8 lanes and
`accm[i] += (-d_x * dmin) * f32(prod[i])` for the 4 min lanes (multiply then add, no FMA). At the end of the
row: `((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7))` plus `((m0+m2)+(m1+m3))`. `kdot.rs` computes the same lanes with
scalar integer loops, so `tests/q8k.rs` asserts the two paths agree bit for bit (1263 rows including hostile
scales; NaN scale patterns count as equal).

### Accuracy consequence (finding)

Because the lanes (and the main/min accumulators) are partial sums over elements of random sign, they are
individually much larger than the cancelled result, and each is rounded when scaled to f32. The error is
therefore bounded by `k * eps * sum_j (|d sc q_j| + |dmin m|) |x_j|` (the term magnitudes), not by
`sqrt(n) * eps * max|term|` as the Phase 2 kernels were: on the fixture rows the sum of term magnitudes is
40 to 90,000 times the result, and one hostile Q5_K row lands 28 ulp of its result away from the exact value
while sitting at 1.4 % of that budget. ggml has the same property. The whole-layer effect is gated by the
frozen Q8 ceilings (`tests/fixtures/q8_ceilings.json`, 2x the Phase 2b per-layer relative errors) in
`tests/real_layers.rs`.

## The Q8_0-grain variant (`--q8-fine`; production in Phase 3): the same lane structure (`dot.rs`, `avx2.rs`)

With a scale `d_x[s]` per 32-element sub-block the integer sums cannot be accumulated across sub-blocks, so the
kernel scales each sub-block's exact 8-lane sum `p32` (`madd(maddubs(q, q8), ones)`: lane `i` = positions
`4i..4i+4`) by one f32 factor before adding it to the row accumulator:

```
Q4_K, Q5_K:  acc[i]  += f32(p32[i]) * (d_x[s] * (d * sc[s]))          8 lanes, per sub-block s
             accm[s] += ((m[s] as f32) * -dmin) * (d_x[s] * sum q_x)  8 lanes, lane = sub-block index
Q6_K:        p32[0] -= 32 * sum(q_x[0..16]),  p32[4] -= 32 * sum(q_x[16..32])   (exact, `off6`)
             acc[i]  += f32(p32[i]) * (d_x[s] * (d * sc[g]))          g = the 16-group of lane i
```

`Q8Row` carries what this needs, derived once per quantisation: `d32` (the f16 scales as f32), `dxs = d32 *
sum(q)` (exact: 11 x 12 significant bits) and `off6`. Per super-block the vector code builds the 8 factors
`dsc8 = dx8 * (d * sc8)` once (the scales unpacked with the same vector routine) and broadcasts lane `s` with
`permutevar8x32`; the Q6_K factors come from a lane pair `[sc[2s] x4, sc[2s+1] x4]`. Cost against the Q8_K
form: one `cvt`, one `mul`, one `add` and one broadcast per sub-block instead of a shuffle and an integer
`madd` with the scale, about 10 to 20 % more instructions per super-block; the all-threads throughput is
unchanged because the kernels are memory-bound there. The scalar reference (`dot.rs`) computes the same lanes
in the same order, so `tests/avx2.rs` still asserts bit-identity, and `tests/kernels.rs` checks the fixture
rows against the f64 reference with the term-magnitude budget above.

## Phase 3.5: factor tables and fixed shifts (kept); two rows per pass (measured slower, not shipped)

The Q8_0-grain kernels' per-sub-block tail was `cvt + permutevar8x32 + mul + add` (two permutes and the
`off6` subtract for Q6_K) and Q5_K's high bits went through a variable-shift jump table. Phase 3.5 changes
the instruction stream, not the arithmetic:

- **Factor tables.** `k45_header` (Q4_K / Q5_K) and `q6_factors` (Q6_K) compute the sub-block factors once
  per super-block into a stack array (8 f32; 16 for Q6_K: `d_x[s] * (d * sc[2s])` and `d_x[s] * (d *
  sc[2s+1])`, the int8 scales de-interleaved with one byte shuffle). The sub-block loop reads them back with
  `vbroadcastss`, a load-port uop (Q6_K: a pair joined with `blend_ps`), instead of `permutevar8x32` on
  port 5.
- **Q5_K high bits** are consumed bit by bit: `(qh & 1) << 4`, then `qh >>= 1` in 16-bit lanes (the bit that
  spills from a high byte into a low byte's bit 7 is never read). No variable shift; the Q8_K Q5_K kernel
  does the same.
- **Q6_K unpack** with two masks (`0x0F`, `0x30`) instead of five, shared by both activation forms
  (`q6_codes`): `(qh << 4) & 0x30`, `(qh << 2) & 0x30`, `qh & 0x30`, `(qh >> 2) & 0x30`.

The operations per row are the same in the same order, so the AVX2 kernels and the scalar `dot.rs` /
`kdot.rs` still give the same bits (`tests/avx2.rs`, `tests/q8k.rs`), and the frozen ceilings do not move.

**Two rows per pass** was implemented for both activation forms (`dot2_q{4,5,6}_k_q8`, `dot2_*_q8k`: two
weight rows against one activation, the activation and per-block loads shared, the two rows' accumulator
chains interleaved, `matvec` / `matvec2` walking row pairs) and measured against the single-row kernels with
an interleaved in-process A/B (the two variants alternated 21 times per line, medians compared;
`docs/data/two_row_ab.log`): the two-row form was slower on 51 of 54 lines, ratios 0.65 to 1.16 and mostly
0.85 to 0.95, worst for Q6_K (0.65 to 0.93), at one thread as much as at six. The likely cause is register
pressure (two rows of unpacked codes, two accumulator sets and the masks exceed the 16 ymm registers, and
the Q8_K kernels have no accumulator-latency problem to solve in the first place). It was removed again;
rows stay one per pass. The same A/B gave the only burst measurement of the 3.5 kernels on this box: with
the clock up for a fraction of a second, the single-row Q8_K kernels read 27 to 34 GB/s at 6 threads on
17408 x 5120 (no cache help), at or above the membw of the same minutes (27 to 32 GB/s): they are
memory-bound when the clock is available. The sustained numbers (`docs/data/bench_rounds.log`, protocol
`tools/bench_rounds.ps1`: plugged in, 5 minutes idle, then rounds of `membw` followed by the kernels at both
shapes, every round filed with the thermal zone, clock ratio and other load next to it) are the box's
throttled state under a runaway service (`docs/payload-vs-doc.md` finding 47), not the kernels.

## Other choices

- **Scales unpack** (Q4_K/Q5_K): the 12 packed bytes become the 16 i16 lanes `sc[0..8], m[0..8]` with 10
  vector ops (mask, shift, byte shuffle, blend, 32-bit interleave, zero-extend) instead of ggml's scalar
  `utmp` masks; `avx2.rs` has a unit test against `get_scale_min_k4`.
- **Scale broadcast**: ggml's 256-byte `get_scale_shuffle_k4` table (`shuffle_epi8` per sub-block).
- **Software prefetch** ~1 KB ahead, one prefetch per 64 bytes of weight row: a single thread streaming from
  DRAM was latency-bound at 3.3 GB/s (Q4_K) against 6.6 GB/s from cache; with it the single thread runs at
  its compute speed and six threads reach 70 to 90 % of the measured memory bandwidth.
- **Q8_0 and Q4_0 weights** keep the Phase 2b kernels (4.5 % and 1.3 % of the file), plus the same prefetch.
- **Thread pool**: one job at a time, contiguous row ranges per participant, spin for 60 us then block; the
  caller is participant 0. Row `r` is always computed by one participant with the same kernel, so any thread
  count gives the same bits (`tests/q8k.rs`, `tests/layers.rs`).
- **Fused gate+up** (`matvec2`) and the batched `matmul` (prefill) reuse `one_row`, so their results are the
  bits `matvec` produces for each row and token.
