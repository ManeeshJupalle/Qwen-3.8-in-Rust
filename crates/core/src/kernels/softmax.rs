//! Row softmax with max subtraction, and the causal-masked variant used by attention.
//!
//! Order: `m = max(x)` (left to right; -inf allowed), `e_i = exp(x_i - m)`, `s = sum e_i` left to right,
//! `y_i = e_i / s`. Masked entries (columns > row in the causal form) are exactly 0 and never enter the sum,
//! which equals HF's `softmax(scores + finfo.min)` because `exp(finfo.min - m)` underflows to 0.

/// Softmax over `x` into `y` (same length). An all `-inf` row yields NaN like torch.
pub fn softmax(x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    let mut m = f32::NEG_INFINITY;
    for &v in x {
        if v > m {
            m = v;
        }
    }
    let mut s = 0f32;
    for i in 0..x.len() {
        let e = (x[i] - m).exp();
        y[i] = e;
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
