//! `Qwen3_5GatedDeltaNet` (modeling_qwen3_5.py lines 387-548) in GGUF head order; op order per docs/deltanet.md:
//! projections -> causal conv + SiLU -> split q/k/v -> gates -> per V head: L2-norm q/k, scale q, delta step ->
//! gated RMSNorm with z -> out projection. V head `p` uses K head `p % n_k` (converter re-tiling).
//! Parallel over V heads; each head owns its state and output slice. `forward_prefill` (Phase 3.3) runs the
//! projections as batched matmuls over the prompt and the conv / recurrence sequentially per token, so it is
//! the same arithmetic as `t` calls of `forward_token`.

use super::{Linear, Result, TensorSource};
use crate::kernels::conv::causal_conv1d_step;
use crate::kernels::deltanet::{deltanet_step, gate_beta, gate_g, l2norm};
use crate::kernels::matvec::{acts_for, matmul, matvec, row_range, ActBuf, ActVec};
use crate::kernels::pool::{self, SharedMut};
use crate::kernels::rmsnorm::rmsnorm_gated;
use crate::GgmlType;

/// Preallocated per-token buffers for `forward_token_in` (Phase 3.6). The per-head temporaries (`qn`, `kn`,
/// `o`) get one slot per head rather than one per participant, so every head's scratch is disjoint whatever
/// the thread count and the head loop can run on the shared pool.
pub struct DeltaScratch {
    pub mixed: Vec<f32>,
    pub z: Vec<f32>,
    pub ga: Vec<f32>,
    pub gb: Vec<f32>,
    pub conved: Vec<f32>,
    pub heads: Vec<f32>,
    pub qn: Vec<f32>,
    pub kn: Vec<f32>,
    pub o: Vec<f32>,
    pub act: ActBuf,
}

impl DeltaScratch {
    pub fn new(conv_dim: usize, n_v: usize, dk: usize, dv: usize) -> DeltaScratch {
        let vd = n_v * dv;
        DeltaScratch {
            mixed: vec![0f32; conv_dim],
            z: vec![0f32; vd],
            ga: vec![0f32; n_v],
            gb: vec![0f32; n_v],
            conved: vec![0f32; conv_dim],
            heads: vec![0f32; vd],
            qn: vec![0f32; n_v * dk],
            kn: vec![0f32; n_v * dk],
            o: vec![0f32; vd],
            act: ActBuf::with_capacity(vd),
        }
    }
}

pub struct GatedDeltaNet {
    pub hidden: usize,
    pub n_k: usize,
    pub n_v: usize,
    pub dk: usize,
    pub dv: usize,
    pub kernel: usize,
    pub eps: f32,
    pub qkv: Linear,
    pub gate_z: Linear,
    pub alpha: Linear,
    pub beta: Linear,
    /// `ssm_a` (`-exp(A_log)`), one per V head.
    pub a: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// `ssm_conv1d.weight`, `conv_dim * kernel`, channel-major.
    pub conv_w: Vec<f32>,
    /// `ssm_norm.weight`, `dv`.
    pub norm_w: Vec<f32>,
    pub out: Linear,
}

/// Carried state: conv window (`conv_dim * (kernel-1)`, per channel oldest first) and recurrent matrices
/// (`n_v * dk * dv`, row-major per head).
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaState {
    pub conv: Vec<f32>,
    pub rec: Vec<f32>,
}

impl GatedDeltaNet {
    #[allow(clippy::too_many_arguments)]
    pub fn load(src: &dyn TensorSource, prefix: &str, hidden: usize, n_k: usize, n_v: usize, dk: usize, dv: usize, kernel: usize, eps: f32) -> Result<GatedDeltaNet> {
        let kd = n_k * dk;
        let vd = n_v * dv;
        let conv_dim = 2 * kd + vd;
        Ok(GatedDeltaNet {
            hidden,
            n_k,
            n_v,
            dk,
            dv,
            kernel,
            eps,
            qkv: Linear::load(src, &format!("{prefix}attn_qkv.weight"), conv_dim, hidden)?,
            gate_z: Linear::load(src, &format!("{prefix}attn_gate.weight"), vd, hidden)?,
            alpha: Linear::load(src, &format!("{prefix}ssm_alpha.weight"), n_v, hidden)?,
            beta: Linear::load(src, &format!("{prefix}ssm_beta.weight"), n_v, hidden)?,
            a: src.vec(&format!("{prefix}ssm_a"), n_v)?,
            dt_bias: src.vec(&format!("{prefix}ssm_dt.bias"), n_v)?,
            conv_w: src.vec(&format!("{prefix}ssm_conv1d.weight"), conv_dim * kernel)?,
            norm_w: src.vec(&format!("{prefix}ssm_norm.weight"), dv)?,
            out: Linear::load(src, &format!("{prefix}ssm_out.weight"), hidden, vd)?,
        })
    }

    pub fn conv_dim(&self) -> usize {
        2 * self.n_k * self.dk + self.n_v * self.dv
    }

    pub fn new_state(&self) -> DeltaState {
        DeltaState { conv: vec![0f32; self.conv_dim() * (self.kernel - 1)], rec: vec![0f32; self.n_v * self.dk * self.dv] }
    }

    /// The weight types `forward_token_in` needs its `act` filled for.
    pub fn input_types(&self) -> [GgmlType; 4] {
        [self.qkv.w.ggml_type, self.gate_z.w.ggml_type, self.alpha.w.ggml_type, self.beta.w.ggml_type]
    }

    /// One token: `x` (hidden) -> `y` (hidden), updating `state`.
    pub fn forward_token(&self, x: &[f32], state: &mut DeltaState, y: &mut [f32], threads: usize) {
        let mut sc = DeltaScratch::new(self.conv_dim(), self.n_v, self.dk, self.dv);
        let mut act = ActBuf::with_capacity(x.len());
        act.fill(x, &self.input_types());
        self.forward_token_in(x, &act, state, &mut sc, y, threads);
    }

    /// `forward_token` with caller-owned buffers: no allocation. `act` must already be filled for `x` with
    /// `input_types()`. Bit for bit what `forward_token` computes: every head reads and writes the same
    /// values in the same order, only from preallocated slots instead of fresh ones.
    pub fn forward_token_in(&self, x: &[f32], act: &ActBuf, state: &mut DeltaState, sc: &mut DeltaScratch, y: &mut [f32], threads: usize) {
        let (n_k, n_v, dk, dv) = (self.n_k, self.n_v, self.dk, self.dv);
        let kd = n_k * dk;
        let vd = n_v * dv;
        let conv_dim = self.conv_dim();
        // one quantisation of `x` (in `act`), shared by the four projections
        matvec(&self.qkv.w, act.act_for(self.qkv.w.ggml_type, x), &mut sc.mixed[..conv_dim], threads);
        matvec(&self.gate_z.w, act.act_for(self.gate_z.w.ggml_type, x), &mut sc.z[..vd], threads);
        matvec(&self.alpha.w, act.act_for(self.alpha.w.ggml_type, x), &mut sc.ga[..n_v], threads);
        matvec(&self.beta.w, act.act_for(self.beta.w.ggml_type, x), &mut sc.gb[..n_v], threads);
        {
            crate::prof_scope!(crate::prof::Stage::Conv);
            causal_conv1d_step(&mut state.conv, &self.conv_w, self.kernel, &sc.mixed[..conv_dim], &mut sc.conved[..conv_dim], true);
        }
        {
            // disjoint field borrows, so the head loop can read `conved` / `z` / gates while writing its own slots
            let DeltaScratch { conved, z, ga, gb, heads, qn, kn, o, .. } = &mut *sc;
            let conved: &[f32] = &conved[..conv_dim];
            let (q_all, rest) = conved.split_at(kd);
            let (k_all, v_all) = rest.split_at(kd);
            let (z, ga, gb): (&[f32], &[f32], &[f32]) = (&z[..vd], &ga[..n_v], &gb[..n_v]);
            let heads_p = SharedMut::new(&mut heads[..vd]);
            let qn_p = SharedMut::new(&mut qn[..n_v * dk]);
            let kn_p = SharedMut::new(&mut kn[..n_v * dk]);
            let o_p = SharedMut::new(&mut o[..vd]);
            let rec_p = SharedMut::new(&mut state.rec);
            let scale = 1.0 / (dk as f32).sqrt();
            crate::prof_scope!(crate::prof::Stage::DeltaNetRec);
            pool::global().run(threads.max(1).min(n_v), &|tid, n| {
                let (p0, p1) = row_range(n_v, tid, n);
                for p in p0..p1 {
                    let kh = p % n_k;
                    // SAFETY: head `p` is handled by exactly one participant, and every range here is that
                    // head's own slot (its q/k/output temporaries, its recurrent state, its output columns).
                    let (qnh, knh, oh, rec, out) = unsafe {
                        (qn_p.slice(p * dk, dk), kn_p.slice(p * dk, dk), o_p.slice(p * dv, dv), rec_p.slice(p * dk * dv, dk * dv), heads_p.slice(p * dv, dv))
                    };
                    qnh.copy_from_slice(&q_all[kh * dk..(kh + 1) * dk]);
                    knh.copy_from_slice(&k_all[kh * dk..(kh + 1) * dk]);
                    l2norm(qnh, 1e-6);
                    l2norm(knh, 1e-6);
                    for v in qnh.iter_mut() {
                        *v *= scale;
                    }
                    let g = gate_g(ga[p], self.dt_bias[p], self.a[p]);
                    let beta = gate_beta(gb[p]);
                    deltanet_step(qnh, knh, &v_all[p * dv..(p + 1) * dv], g, beta, rec, oh);
                    rmsnorm_gated(oh, &z[p * dv..(p + 1) * dv], &self.norm_w, self.eps, out);
                }
            });
        }
        let ot = self.out.w.ggml_type;
        sc.act.fill(&sc.heads[..vd], &[ot]);
        matvec(&self.out.w, sc.act.act_for(ot, &sc.heads[..vd]), y, threads);
    }

    /// Prefill `t` tokens (`x` is `t * hidden`, `y` is `t * hidden`): the four input projections and the output
    /// projection as batched matmuls (weights read once), the conv window and the recurrence stepped token by
    /// token per head (parallel over heads). Row `i` equals `forward_token` on row `i` bit for bit.
    pub fn forward_prefill(&self, x: &[f32], t: usize, state: &mut DeltaState, y: &mut [f32], threads: usize) {
        let (n_k, n_v, dk, dv) = (self.n_k, self.n_v, self.dk, self.dv);
        let h = self.hidden;
        let kd = n_k * dk;
        let vd = n_v * dv;
        let conv_dim = self.conv_dim();
        assert_eq!(x.len(), t * h);
        assert_eq!(y.len(), t * h);
        let acts: Vec<ActVec<'_>> = x.chunks_exact(h).map(ActVec::new).collect();
        let mut mixed = vec![0f32; t * conv_dim];
        matmul(&self.qkv.w, &acts_for(&acts, self.qkv.w.ggml_type), &mut mixed, threads);
        let mut z = vec![0f32; t * vd];
        matmul(&self.gate_z.w, &acts_for(&acts, self.gate_z.w.ggml_type), &mut z, threads);
        let mut a = vec![0f32; t * n_v];
        matmul(&self.alpha.w, &acts_for(&acts, self.alpha.w.ggml_type), &mut a, threads);
        let mut b = vec![0f32; t * n_v];
        matmul(&self.beta.w, &acts_for(&acts, self.beta.w.ggml_type), &mut b, threads);
        let mut conved = vec![0f32; t * conv_dim];
        for i in 0..t {
            causal_conv1d_step(&mut state.conv, &self.conv_w, self.kernel, &mixed[i * conv_dim..(i + 1) * conv_dim], &mut conved[i * conv_dim..(i + 1) * conv_dim], true);
        }
        let mut heads_out = vec![0f32; t * vd];
        let scale = 1.0 / (dk as f32).sqrt();
        let out_p = SharedMut::new(&mut heads_out);
        let rec_p = SharedMut::new(&mut state.rec);
        pool::global().run(threads.max(1).min(n_v), &|tid, n| {
            let (p0, p1) = row_range(n_v, tid, n);
            let mut o = vec![0f32; dv];
            for p in p0..p1 {
                let kh = p % n_k;
                // SAFETY: head `p` is handled by exactly one participant; its state and output columns are disjoint.
                let rec = unsafe { rec_p.slice(p * dk * dv, dk * dv) };
                for i in 0..t {
                    let base = i * conv_dim;
                    let mut qn = conved[base + kh * dk..base + (kh + 1) * dk].to_vec();
                    let mut kn = conved[base + kd + kh * dk..base + kd + (kh + 1) * dk].to_vec();
                    l2norm(&mut qn, 1e-6);
                    l2norm(&mut kn, 1e-6);
                    for v in qn.iter_mut() {
                        *v *= scale;
                    }
                    let g = gate_g(a[i * n_v + p], self.dt_bias[p], self.a[p]);
                    let beta = gate_beta(b[i * n_v + p]);
                    let v = &conved[base + 2 * kd + p * dv..base + 2 * kd + (p + 1) * dv];
                    deltanet_step(&qn, &kn, v, g, beta, rec, &mut o);
                    // SAFETY: as above.
                    let out = unsafe { out_p.slice(i * vd + p * dv, dv) };
                    rmsnorm_gated(&o, &z[i * vd + p * dv..i * vd + (p + 1) * dv], &self.norm_w, self.eps, out);
                }
            }
        });
        let hacts: Vec<ActVec<'_>> = heads_out.chunks_exact(vd).map(ActVec::new).collect();
        matmul(&self.out.w, &acts_for(&hacts, self.out.w.ggml_type), y, threads);
    }
}
