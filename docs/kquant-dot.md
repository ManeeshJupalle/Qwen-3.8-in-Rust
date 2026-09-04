# K-quant dot products with Q8_K activations (Phase 3.1)

Source read: ggml `ggml-quants.c` (`quantize_row_q8_K_ref`, `nearest_int`) and
`ggml-cpu/arch/x86/quants.c` (`ggml_vec_dot_q4_K_q8_K`, `q5_K`, `q6_K`, AVX2 branches) in the
`models/llama.cpp` checkout. Engine files: `crates/core/src/kernels/q8k.rs` (activation row),
`kdot.rs` (scalar reference), `avx2.rs` (vector kernels), `matvec.rs` (row partition, `ActVec`, fused and
batched forms), `pool.rs` (persistent workers). Numbers: `docs/data/membw.txt`, `docs/data/kernels_bench.txt`.

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

## Other choices

- **Scales unpack** (Q4_K/Q5_K): the 12 packed bytes become the 16 i16 lanes `sc[0..8], m[0..8]` with 10
  vector ops (mask, shift, byte shuffle, blend, 32-bit interleave, zero-extend) instead of ggml's scalar
  `utmp` masks; `avx2.rs` has a unit test against `get_scale_min_k4`.
- **Scale broadcast**: ggml's 256-byte `get_scale_shuffle_k4` table (`shuffle_epi8` per sub-block).
- **Software prefetch** ~1 KB ahead, one prefetch per 64 bytes of weight row: a single thread streaming from
  DRAM was latency-bound at 3.3 GB/s (Q4_K) against 6.6 GB/s from cache; with it the single thread runs at
  its compute speed and six threads reach 70 to 90 % of the measured memory bandwidth.
- **Q8_0 and Q4_0** keep the Phase 2b kernels with Q8_0 activations (4.5 % and 1.3 % of the file), plus the
  same prefetch.
- **Thread pool**: one job at a time, contiguous row ranges per participant, spin for 60 us then block; the
  caller is participant 0. Row `r` is always computed by one participant with the same kernel, so any thread
  count gives the same bits (`tests/q8k.rs`, `tests/layers.rs`).
- **Fused gate+up** (`matvec2`) and the batched `matmul` (prefill) reuse `one_row`, so their results are the
  bits `matvec` produces for each row and token.
