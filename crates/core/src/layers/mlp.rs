//! `Qwen3_5MLP` (modeling_qwen3_5.py lines 707-722): `y = down(silu(gate(x)) * up(x))`.

use super::{Linear, Result, TensorSource};
use crate::kernels::act::swiglu;

pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl Mlp {
    /// Load `blk.N.ffn_{gate,up,down}.weight` (numpy shapes `[inter, hidden]`, `[inter, hidden]`, `[hidden, inter]`).
    pub fn load(src: &dyn TensorSource, prefix: &str, hidden: usize, inter: usize) -> Result<Mlp> {
        Ok(Mlp {
            gate: Linear::load(src, &format!("{prefix}ffn_gate.weight"), inter, hidden)?,
            up: Linear::load(src, &format!("{prefix}ffn_up.weight"), inter, hidden)?,
            down: Linear::load(src, &format!("{prefix}ffn_down.weight"), hidden, inter)?,
        })
    }

    pub fn forward(&self, x: &[f32], y: &mut [f32], threads: usize) {
        let inter = self.gate.out_features();
        let mut g = vec![0f32; inter];
        let mut u = vec![0f32; inter];
        self.gate.forward(x, &mut g, threads);
        self.up.forward(x, &mut u, threads);
        let mut h = vec![0f32; inter];
        swiglu(&g, &u, &mut h);
        self.down.forward(&h, y, threads);
    }
}
