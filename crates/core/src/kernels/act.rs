//! Activations: SiLU (`x * sigmoid(x)`, ACT2FN["silu"], also what config.json calls "swish") and the
//! elementwise SwiGLU product `silu(gate) * up` used by the MLP. No summation.

/// `x * sigmoid(x)` computed as `x / (1 + exp(-x))` (torch's CPU kernel form). For x = -1000 this gives
/// `-1000 / inf = -0.0`, matching torch.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu_slice(x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len());
    for (o, &v) in y.iter_mut().zip(x) {
        *o = silu(v);
    }
}

/// `y = silu(gate) * up`, elementwise.
pub fn swiglu(gate: &[f32], up: &[f32], y: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(gate.len(), y.len());
    for i in 0..gate.len() {
        y[i] = silu(gate[i]) * up[i];
    }
}

/// `softplus(x) = log1p(exp(x))`, with torch's threshold: returns `x` when `x > 20`.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}
