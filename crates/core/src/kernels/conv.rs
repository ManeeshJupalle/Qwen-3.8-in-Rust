//! Causal depthwise conv1d before the DeltaNet projections (kernel width from the GGUF, 4 for this model),
//! incremental form with a carried window, as `causal_conv1d_update` in modeling_qwen3_5.py (lines 200-218):
//! `window = [state[c][0..K-1], x_new[c]]`, `out[c] = sum_{j=0..K} w[c][j] * window[j]` (left to right; the
//! newest sample multiplies `w[c][K-1]`), `state[c] <- window[1..]`, then SiLU if requested.
//! Prefill = T sequential steps from a zero state (`causal_conv1d_fn`, lines 220-240, zero left padding).

use super::act::silu;

/// One step for all channels. `state` is `channels * (kernel-1)` (per channel, oldest first),
/// `w` is `channels * kernel`, `x_new` and `out` are `channels`.
pub fn causal_conv1d_step(state: &mut [f32], w: &[f32], kernel: usize, x_new: &[f32], out: &mut [f32], act_silu: bool) {
    let c = x_new.len();
    assert!(kernel >= 1);
    assert_eq!(state.len(), c * (kernel - 1), "conv step: state length");
    assert_eq!(w.len(), c * kernel, "conv step: weight length");
    assert_eq!(out.len(), c, "conv step: output length");
    for ch in 0..c {
        let st = &mut state[ch * (kernel - 1)..(ch + 1) * (kernel - 1)];
        let wc = &w[ch * kernel..(ch + 1) * kernel];
        let mut acc = 0f32;
        for j in 0..kernel - 1 {
            acc += wc[j] * st[j];
        }
        acc += wc[kernel - 1] * x_new[ch];
        for j in 0..kernel.saturating_sub(2) {
            st[j] = st[j + 1];
        }
        if kernel >= 2 {
            st[kernel - 2] = x_new[ch];
        }
        out[ch] = if act_silu { silu(acc) } else { acc };
    }
}

/// Prefill over `t` steps: `x` is `channels x t` (row-major, channel-major), output the same shape.
/// The state is updated in place (start from zeros for a fresh sequence).
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_prefill(state: &mut [f32], w: &[f32], kernel: usize, x: &[f32], channels: usize, t: usize, out: &mut [f32], act_silu: bool) {
    assert_eq!(x.len(), channels * t);
    assert_eq!(out.len(), channels * t);
    let mut xin = vec![0f32; channels];
    let mut o = vec![0f32; channels];
    for step in 0..t {
        for ch in 0..channels {
            xin[ch] = x[ch * t + step];
        }
        causal_conv1d_step(state, w, kernel, &xin, &mut o, act_silu);
        for ch in 0..channels {
            out[ch * t + step] = o[ch];
        }
    }
}
