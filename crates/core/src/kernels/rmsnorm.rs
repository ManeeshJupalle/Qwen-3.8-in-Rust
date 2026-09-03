//! RMSNorm as in `Qwen3_5RMSNorm` (zero-centered; the GGUF weight already includes the +1) and the
//! DeltaNet `Qwen3_5RMSNormGated` (weight stored as-is, gate applied after the weight).
//!
//! Summation order: the mean of squares is accumulated left to right in f32 over the row.
//! `x * rsqrt(mean + eps)` is computed as `x * (1 / sqrt(mean + eps))`; then `* w` (then `* silu(gate)`).

use super::act::silu;

/// Zero-centered RMSNorm with a GGUF-layout weight: `y = (x * rsqrt(mean(x^2) + eps)) * w`.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) {
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

/// `1 / sqrt(mean(x^2) + eps)`, mean accumulated left to right.
#[inline]
pub fn inv_rms(x: &[f32], eps: f32) -> f32 {
    let mut ss = 0f32;
    for &v in x {
        ss += v * v;
    }
    let mean = ss / x.len() as f32;
    1.0 / (mean + eps).sqrt()
}
