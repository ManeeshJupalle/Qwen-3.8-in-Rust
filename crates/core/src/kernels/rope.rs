//! Rotary position embedding as applied by `apply_rotary_pos_emb` in modeling_qwen3_5.py (see docs/rope.md):
//! partial rotary on the first `rope_dim` dims of each head, rotate-half pairing `(i, i + rope_dim/2)`.
//! For text-only input the mRoPE streams are identical, so this is plain RoPE.
//!
//! Tables follow torch: `inv_freq[i] = 1 / theta^(2i / rope_dim)` (f32), `angle = pos as f32 * inv_freq[i]`
//! (one f32 multiply), `cos`/`sin` in f32. `cos[i + half] == cos[i]` (torch concatenates freqs twice).
//! Each rotated output is a 2-term expression `a*c - b*s` / `b*c + a*s`, evaluated exactly in that order.

/// `1 / theta^(2i / rope_dim)` for `i` in `0..rope_dim/2`, computed in f32 like torch.
pub fn inv_freq(rope_dim: usize, theta: f32) -> Vec<f32> {
    (0..rope_dim / 2).map(|i| 1.0 / theta.powf((2 * i) as f32 / rope_dim as f32)).collect()
}

/// cos/sin rows of length `rope_dim` for one position.
pub fn cos_sin(pos: u32, inv_freq: &[f32], cos: &mut [f32], sin: &mut [f32]) {
    let half = inv_freq.len();
    assert_eq!(cos.len(), 2 * half);
    assert_eq!(sin.len(), 2 * half);
    let p = pos as f32;
    for i in 0..half {
        let angle = p * inv_freq[i];
        let (s, c) = (angle.sin(), angle.cos());
        cos[i] = c;
        cos[i + half] = c;
        sin[i] = s;
        sin[i + half] = s;
    }
}

/// Rotate the first `rope_dim` dims of every head of `x` (`heads * head_dim`, head-major) in place.
pub fn apply_rope(x: &mut [f32], head_dim: usize, rope_dim: usize, cos: &[f32], sin: &[f32]) {
    assert!(rope_dim <= head_dim && rope_dim.is_multiple_of(2));
    assert_eq!(x.len() % head_dim, 0);
    assert_eq!(cos.len(), rope_dim);
    assert_eq!(sin.len(), rope_dim);
    let half = rope_dim / 2;
    for h in x.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let a = h[i];
            let b = h[i + half];
            h[i] = a * cos[i] - b * sin[i];
            h[i + half] = b * cos[i + half] + a * sin[i + half];
        }
    }
}
