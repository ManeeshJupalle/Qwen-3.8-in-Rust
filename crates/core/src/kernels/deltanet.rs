//! One token of the gated delta rule for one head, in the exact order of `torch_recurrent_gated_delta_rule`
//! (modeling_qwen3_5.py lines 331-386; see docs/deltanet.md):
//!
//! ```text
//! 1. S *= exp(g)                        (decay)
//! 2. kv_mem[v] = sum_k S[k][v] * k[k]   (read; sum over k left to right)
//! 3. delta = (v - kv_mem) * beta        (delta write, rank one)
//! 4. S[k][v] += k[k] * delta[v]
//! 5. out[v] = sum_k S[k][v] * q[k]      (read; sum over k left to right)
//! ```
//!
//! The caller L2-normalises q and k (`l2norm`, eps 1e-6) and scales q by `1/sqrt(dk)` before the step
//! (lines 344-353), and computes `g`/`beta` with `gate_g`/`gate_beta` (module lines 492-494).
//! `S` is row-major `dk x dv` (`S[k * dv + v]`). All f32.

use super::act::{sigmoid, softplus};

/// `x * rsqrt(sum(x^2) + eps)` in place (sum left to right).
pub fn l2norm(x: &mut [f32], eps: f32) {
    let mut ss = 0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / (ss + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `g = A * softplus(a + dt_bias)` with `A = ssm_a` as stored in the GGUF (`-exp(A_log)`).
#[inline]
pub fn gate_g(a: f32, dt_bias: f32, big_a: f32) -> f32 {
    big_a * softplus(a + dt_bias)
}

/// `beta = sigmoid(b)`.
#[inline]
pub fn gate_beta(b: f32) -> f32 {
    sigmoid(b)
}

/// One head, one token. `q`, `k`: `dk` (already normalised/scaled); `v`, `out`: `dv`; `s`: `dk * dv`.
pub fn deltanet_step(q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32, s: &mut [f32], out: &mut [f32]) {
    let dk = q.len();
    let dv = v.len();
    assert_eq!(k.len(), dk);
    assert_eq!(out.len(), dv);
    assert_eq!(s.len(), dk * dv);
    let decay = g.exp();
    for x in s.iter_mut() {
        *x *= decay;
    }
    // kv_mem[v] = sum_k S[k][v] * k[k]
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for kk in 0..dk {
        let row = &s[kk * dv..(kk + 1) * dv];
        let kv = k[kk];
        for vv in 0..dv {
            out[vv] += row[vv] * kv;
        }
    }
    // delta = (v - kv_mem) * beta ; S += outer(k, delta)
    for vv in 0..dv {
        out[vv] = (v[vv] - out[vv]) * beta;
    }
    for kk in 0..dk {
        let row = &mut s[kk * dv..(kk + 1) * dv];
        let kv = k[kk];
        for vv in 0..dv {
            row[vv] += kv * out[vv];
        }
    }
    // out[v] = sum_k S[k][v] * q[k]
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for kk in 0..dk {
        let row = &s[kk * dv..(kk + 1) * dv];
        let qk = q[kk];
        for vv in 0..dv {
            out[vv] += row[vv] * qk;
        }
    }
}

/// The whole per-head token step as the module does it: normalise, scale, gates, recurrence.
/// `q`/`k` are copied and normalised internally; `a`, `b` are the projected gate pre-activations.
#[allow(clippy::too_many_arguments)]
pub fn deltanet_head_token(q: &[f32], k: &[f32], v: &[f32], a: f32, b: f32, big_a: f32, dt_bias: f32, s: &mut [f32], out: &mut [f32]) {
    let dk = q.len();
    let mut qn = q.to_vec();
    let mut kn = k.to_vec();
    l2norm(&mut qn, 1e-6);
    l2norm(&mut kn, 1e-6);
    let scale = 1.0 / (dk as f32).sqrt();
    for x in qn.iter_mut() {
        *x *= scale;
    }
    let g = gate_g(a, dt_bias, big_a);
    let beta = gate_beta(b);
    deltanet_step(&qn, &kn, v, g, beta, s, out);
}
