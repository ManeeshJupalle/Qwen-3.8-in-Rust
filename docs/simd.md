# SIMD kernels (Phase 2b): AVX2 paths that are bit-identical to the scalar kernels

`crates/core/src/kernels/avx2.rs` holds AVX2 (+F16C) versions of the kernels that dominate decode time;
`kernels/simd.rs` selects them at runtime. The scalar kernels are unchanged in meaning and remain the reference
that every fixture test checks; the AVX2 path is checked against the scalar path, not against the fixtures.

## Contract

Every AVX2 kernel produces the same bits as its scalar counterpart, on every input the scalar kernel accepts.
The engine therefore emits identical token ids on a CPU with AVX2 and on one without, with the scalar path
forced (`AQUEDUCT_SCALAR=1`), and for any thread count; `tests/avx2.rs` asserts exact equality (max diff 0) on
every kernel fixture and on 1000 random / hostile rows per kernel.

## How each kernel reaches bit-identity

| kernel | scalar order (unchanged) | AVX2 realisation |
|---|---|---|
| `dot_q8_0_q8`, `dot_q4_0_q8` | per 32-block: exact integer `sum(q_w * q_x)` (i32), then `acc += (d_w * d_x) * isum as f32`, blocks left to right | `maddubs` on `\|w\|` and `x * sign(w)` (pairs at most 2 * 128 * 127 = 32512 < 32767, no saturation), `madd` with ones, horizontal add: the same exact integer; the f32 part is the same scalar expression |
| `dot_q4_k_q8`, `dot_q5_k_q8` | per 32 sub-block: exact `isum`, exact `xsum = sum(q_x)`, then `acc += d_x * ((d * sc) * isum - (dmin * m) * xsum)` | nibbles are unsigned, so `maddubs(w, x)` directly; `xsum` via `maddubs(1, x)`; the fifth bit of Q5_K is extracted with a 16-bit shift and mask that never crosses a byte |
| `dot_q6_k_q8` | per 32 values: two exact 16-element sums with int8 scales, `acc += d_x * ((d * s0) * isum0 + (d * s1) * isum1)` | `madd` output lanes 0..4 are elements 0..16 and lanes 4..8 elements 16..32, reduced separately |
| `q8_quantize_row` | block max (order-free), `d = amax / 127`, `id = 1 / d`, `q = round_half_away(x * id)` cast `as i8`, `f16(d)` via `half` | vector abs/max, the same scalar `d`/`id`/`f16`, `round_ps` to even with ties (`\|t - trunc(t)\| == 0.5`) replaced by `trunc(t) + copysign(1, t)`, clamp to [-128, 127], NaN to 0 (the `as i8` rules), pack with the llama.cpp permutation |
| `rmsnorm` (`inv_rms`) | **order redefined in 2b**: eight interleaved lanes (`l` takes elements `l, l+8, ...`), reduced `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))`, then the `n % 8` tail left to right | that order is one `add(mul)` per 8 elements plus the same reduction; `(x * inv) * w` is elementwise |
| `softmax` | **order redefined in 2b**: max (order-free), scalar libm `exp` per element, the same 8-lane sum, elementwise divide | vector max, scalar `exp`, 8-lane sum, `div_ps` |

No FMA anywhere: the scalar Rust never contracts `a * b + c`, so the vector code uses separate `mul`/`add`
too. `_mm_cvtph_ps` and `half::f16::to_f32` are both exact, so f16 scales agree.

Changing the `inv_rms` and softmax summation order moved those scalar results by a few ulps; the Phase 2a
fixture tests (budgets of `k * sqrt(width) * eps * max|x|`) and the tiny oracle (60/60 greedy) still pass, and
`rmsnorm_gated` (DeltaNet) shares `inv_rms`, so the whole engine has one definition of the sum.

## What stays scalar

`dot_f32`, `dot_f32_q8` (F32 weights: `ssm_alpha`, `ssm_beta`, 48 rows per DeltaNet layer), `rmsnorm_gated`
(needs `silu` = `exp` per element), `rope`, `causal_conv1d_step`, `deltanet_step`, `silu`/`swiglu`. They are
either a small share of decode time or bounded by `exp`. The DeltaNet recurrence (`dk x dv` per head per token,
48 heads) is the largest of these and is a candidate for a later vector pass; it is not memory-bound.

## Dispatch

`simd::use_avx2()` caches `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c")` (false when
`AQUEDUCT_SCALAR` is set or off x86_64) in an atomic; `dot_q8`, `Q8Row::quantize`, `rmsnorm` and `softmax`
consult it. `simd::force_scalar(bool)` flips it for the AVX2-vs-scalar tests and for `aqueduct bench kernels`,
which runs every dot kernel on a random `17408 x 5120` matrix (the `ffn_gate` shape, larger than L3) in both
paths at one thread and at all threads and reports weight GB/s (`docs/data/kernels_bench.txt`).

## Phase 3 changes (`docs/kquant-dot.md`)

Bit-identity between the scalar and AVX2 paths is no longer a contract (Phase 3 rule 2); it is a property the
tests still check, and every kernel still has it. The K-quant dot kernels were rebuilt with ggml's structure
(exact integer inner loops, 8-lane f32 accumulation, one reduction per row, software prefetch) for Q8_0-grain
activations (`dot.rs` scalar reference, `avx2.rs`), plus ggml's Q8_K activation row and kernels as an opt-in
(`q8k.rs`, `kdot.rs`). `matvec` runs on the persistent pool (`pool.rs`), `matvec2` fuses the MLP's gate and up,
`matmul` is the batched prefill form. `bench kernels` reports both activation forms and `bench membw` the
ceiling. The scalar `inv_rms` / softmax orders and the Q8_0 / Q4_0 kernels are unchanged apart from prefetch.
