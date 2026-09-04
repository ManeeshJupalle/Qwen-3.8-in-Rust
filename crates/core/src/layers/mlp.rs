//! `Qwen3_5MLP` (modeling_qwen3_5.py lines 707-722): `y = down(silu(gate(x)) * up(x))`.

use super::{Linear, Result, TensorSource};
use crate::kernels::act::swiglu;
use crate::kernels::matvec::{acts_for, matmul, matvec2, ActVec};

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

    /// Gate and up are one fused pass over `x` (quantised once); then SwiGLU and down.
    pub fn forward(&self, x: &[f32], y: &mut [f32], threads: usize) {
        let inter = self.gate.out_features();
        let mut g = vec![0f32; inter];
        let mut u = vec![0f32; inter];
        matvec2(&self.gate.w, &self.up.w, &ActVec::new(x), &mut g, &mut u, threads);
        let mut h = vec![0f32; inter];
        swiglu(&g, &u, &mut h);
        self.down.forward(&h, y, threads);
    }
}

impl Mlp {
    /// `t` rows in (`x` is `t * hidden`), `t` rows out: gate, up and down as batched matmuls (each weight read
    /// once for the whole batch), SwiGLU per row. Row `i` is the arithmetic of `forward` on row `i`.
    pub fn forward_batch(&self, x: &[f32], t: usize, y: &mut [f32], threads: usize) {
        let hidden = self.gate.in_features();
        let inter = self.gate.out_features();
        assert_eq!(x.len(), t * hidden);
        assert_eq!(y.len(), t * hidden);
        let acts: Vec<ActVec<'_>> = x.chunks_exact(hidden).map(ActVec::new).collect();
        let mut g = vec![0f32; t * inter];
        let mut u = vec![0f32; t * inter];
        matmul(&self.gate.w, &acts_for(&acts, self.gate.w.ggml_type), &mut g, threads);
        matmul(&self.up.w, &acts_for(&acts, self.up.w.ggml_type), &mut u, threads);
        let mut h = vec![0f32; t * inter];
        for i in 0..t {
            swiglu(&g[i * inter..(i + 1) * inter], &u[i * inter..(i + 1) * inter], &mut h[i * inter..(i + 1) * inter]);
        }
        let hacts: Vec<ActVec<'_>> = h.chunks_exact(inter).map(ActVec::new).collect();
        matmul(&self.down.w, &acts_for(&hacts, self.down.w.ggml_type), y, threads);
    }
}
