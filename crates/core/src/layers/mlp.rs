//! `Qwen3_5MLP` (modeling_qwen3_5.py lines 707-722): `y = down(silu(gate(x)) * up(x))`.

use super::{Linear, Result, TensorSource};
use crate::kernels::act::swiglu;
use crate::kernels::matvec::{acts_for, matmul, matvec, matvec2, ActBuf, ActVec};

/// Preallocated per-token buffers for `Mlp::forward_in` (Phase 3.6): the gate and up outputs, the SwiGLU
/// result, and the quantised form of that result for the down projection.
pub struct MlpScratch {
    pub g: Vec<f32>,
    pub u: Vec<f32>,
    pub h: Vec<f32>,
    pub act: ActBuf,
}

impl MlpScratch {
    pub fn new(inter: usize) -> MlpScratch {
        MlpScratch { g: vec![0f32; inter], u: vec![0f32; inter], h: vec![0f32; inter], act: ActBuf::with_capacity(inter) }
    }
}

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
        let mut sc = MlpScratch::new(self.gate.out_features());
        let mut act = ActBuf::with_capacity(x.len());
        act.fill(x, &self.input_types());
        self.forward_in(x, &act, &mut sc, y, threads);
    }

    /// `forward` with caller-owned buffers: no allocation. `act` must already be filled for `x` with the
    /// gate and up weight types (the decoder layer fills it once for the whole layer).
    pub fn forward_in(&self, x: &[f32], act: &ActBuf, sc: &mut MlpScratch, y: &mut [f32], threads: usize) {
        let inter = self.gate.out_features();
        let (gt, ut, dt) = (self.gate.w.ggml_type, self.up.w.ggml_type, self.down.w.ggml_type);
        matvec2(&self.gate.w, &self.up.w, act.act_for(gt, x), act.act_for(ut, x), &mut sc.g[..inter], &mut sc.u[..inter], threads);
        {
            crate::prof_scope!(crate::prof::Stage::Swiglu);
            swiglu(&sc.g[..inter], &sc.u[..inter], &mut sc.h[..inter]);
        }
        sc.act.fill(&sc.h[..inter], &[dt]);
        matvec(&self.down.w, sc.act.act_for(dt, &sc.h[..inter]), y, threads);
    }

    /// The weight types whose activation forms `forward_in` needs `act` filled for.
    pub fn input_types(&self) -> [crate::GgmlType; 2] {
        [self.gate.w.ggml_type, self.up.w.ggml_type]
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
