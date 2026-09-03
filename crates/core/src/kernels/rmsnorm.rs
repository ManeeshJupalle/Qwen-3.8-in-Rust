//! RMSNorm as in `Qwen3_5RMSNorm` (zero-centered; the GGUF weight already includes the +1) and the
//! DeltaNet `Qwen3_5RMSNormGated` (weight stored as-is, gate applied after the weight).
//!
//! Summation order: the mean of squares is accumulated in eight interleaved lanes (lane `l` takes elements
//! `l, l+8, l+16, ...` left to right) reduced as `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))`, then the tail
//! (`n % 8` elements) is added left to right. This is exactly what the AVX2 version does, so both paths are
//! bit-identical. `x * rsqrt(mean + eps)` is computed as `x * (1 / sqrt(mean + eps))`; then `* w`
//! (then `* silu(gate)`).

use super::act::silu;

/// Zero-centered RMSNorm with a GGUF-layout weight: `y = (x * rsqrt(mean(x^2) + eps)) * w`.
/// AVX2 when available (bit-identical), else scalar.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if super::simd::use_avx2() {
        // SAFETY: `use_avx2` is only true when the CPU reports AVX2 and F16C.
        unsafe { super::avx2::rmsnorm(x, w, eps, y) };
        return;
    }
    rmsnorm_scalar(x, w, eps, y)
}

/// The scalar reference.
pub fn rmsnorm_scalar(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) {
    let n = x.len();
    assert_eq!(w.len(), n, "rmsnorm: weight length");
    assert_eq!(y.len(), n, "rmsnorm: output length");
    let inv = inv_rms(x, eps);
    for i in 0..n {
        y[i] = (x[i] * inv) * w[i];
    }
}

/// Gated RMSNorm of the DeltaNet output (per head): `y = (w * (x * rsqrt(mean(x^2) + eps))) * silu(gate)`.
pub fn rmsnorm_gated(x: &[f32], gate: &[f32], w: &[f32], eps: f32, y: &mut [f32]) {
    let n = x.len();
    assert_eq!(w.len(), n, "rmsnorm_gated: weight length");
    assert_eq!(gate.len(), n, "rmsnorm_gated: gate length");
    assert_eq!(y.len(), n, "rmsnorm_gated: output length");
    let inv = inv_rms(x, eps);
    for i in 0..n {
        y[i] = (w[i] * (x[i] * inv)) * silu(gate[i]);
    }
}

/// `1 / sqrt(mean(x^2) + eps)` with the 8-lane summation order documented above (scalar reference).
#[inline]
pub fn inv_rms(x: &[f32], eps: f32) -> f32 {
    let mean = sum_squares_8lane(x) / x.len() as f32;
    1.0 / (mean + eps).sqrt()
}

/// Sum of squares: eight interleaved lanes, reduced `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))`, then the tail.
pub fn sum_squares_8lane(x: &[f32]) -> f32 {
    let n8 = x.len() / 8 * 8;
    let mut l = [0f32; 8];
    let mut i = 0;
    while i < n8 {
        for (k, lane) in l.iter_mut().enumerate() {
            *lane += x[i + k] * x[i + k];
        }
        i += 8;
    }
    let mut ss = reduce8(&l);
    for &v in &x[n8..] {
        ss += v * v;
    }
    ss
}

/// The lane reduction shared with the AVX2 kernels.
#[inline]
pub fn reduce8(l: &[f32; 8]) -> f32 {
    ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]))
}
