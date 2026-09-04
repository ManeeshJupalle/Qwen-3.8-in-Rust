//! Kernels: scalar reference implementations (Phase 2a) and, on x86_64 with AVX2 + F16C, AVX2 versions of the
//! hot ones (Phase 2b `avx2.rs`, selected at runtime by `simd.rs`). Phase 3 adds the Q8_K activation row
//! (`q8k.rs`) and the K-quant x Q8_K dot kernels (`kdot.rs` scalar, `avx2.rs` vector; `docs/kquant-dot.md`),
//! the persistent thread pool (`pool.rs`) and the batched matmul. Every kernel documents its summation order;
//! all arithmetic is f32 with f32 accumulators (no f64, no fast-math, no FMA). Parallelism only in `matvec` /
//! `matmul` (over rows) and in the layers (over heads), always producing bit-identical results for any thread
//! count; the AVX2 kernels happen to be bit-identical to the scalar ones as well (a property the tests check,
//! not a contract: Phase 3 rule 2).
//!
//! Each kernel has a Python generator in `tools/fixtures/gen_kernels.py`, a fixture directory in
//! `tests/fixtures/kernels/<name>/`, and a test in `crates/core/tests/kernels.rs`.

pub mod act;
pub mod avx2;
pub mod conv;
pub mod deltanet;
pub mod dot;
pub mod kdot;
pub mod matvec;
pub mod pool;
pub mod q8;
pub mod q8k;
pub mod rmsnorm;
pub mod rope;
pub mod simd;
pub mod softmax;

/// f32 machine epsilon (2^-23), used by the derived error budgets.
pub const F32_EPS: f32 = f32::EPSILON;
