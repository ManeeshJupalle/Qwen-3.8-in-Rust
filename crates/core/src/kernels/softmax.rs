//! Row softmax with max subtraction, and the causal-masked variant used by attention.
//!
//! Order: `m = max(x)` (-inf allowed), `e_i = exp(x_i - m)` (scalar libm `exp` on every path), `s = sum e_i`
//! in eight interleaved lanes reduced as `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))` plus a left-to-right tail
//! (the order the AVX2 version uses, so both paths are bit-identical), `y_i = e_i / s`. Masked entries
//! (columns > row in the causal form) are exactly 0 and never enter the sum, which equals HF's
//! `softmax(scores + finfo.min)` because `exp(finfo.min - m)` underflows to 0.

/// Softmax over `x` into `y` (same length). An all `-inf` row yields NaN like torch.
/// AVX2 when available (bit-identical), else scalar.
pub fn softmax(x: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if super::simd::use_avx2() {
        // SAFETY: `use_avx2` is only true when the CPU reports AVX2 and F16C.
        unsafe { super::avx2::softmax(x, y) };
        return;
    }
    softmax_scalar(x, y)
}

/// The scalar reference.
pub fn softmax_scalar(x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    let mut m = f32::NEG_INFINITY;
    for &v in x {
        if v > m {
            m = v;
        }
    }
    for i in 0..x.len() {
        y[i] = (x[i] - m).exp();
    }
    let n8 = y.len() / 8 * 8;
    let mut l = [0f32; 8];
    let mut i = 0;
    while i < n8 {
        for (k, lane) in l.iter_mut().enumerate() {
            *lane += y[i + k];
        }
        i += 8;
    }
    let mut s = super::rmsnorm::reduce8(&l);
    for &e in &y[n8..] {
        s += e;
    }
    for v in y.iter_mut() {
        *v /= s;
    }
}

/// Causal softmax for query row `row` over `scores[0..=row]` of a row of length `len`: `y[j] = 0` for `j > row`.
pub fn softmax_causal_row(scores: &[f32], row: usize, y: &mut [f32]) {
    let len = scores.len();
    assert_eq!(y.len(), len);
    assert!(row < len);
    softmax(&scores[..=row], &mut y[..=row]);
    for v in y[row + 1..].iter_mut() {
        *v = 0.0;
    }
}
