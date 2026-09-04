//! `Qwen3_5GatedDeltaNet` (modeling_qwen3_5.py lines 387-548) in GGUF head order; op order per docs/deltanet.md:
//! projections -> causal conv + SiLU -> split q/k/v -> gates -> per V head: L2-norm q/k, scale q, delta step ->
//! gated RMSNorm with z -> out projection. V head `p` uses K head `p % n_k` (converter re-tiling).
//! Parallel over V heads with `std::thread::scope`; each head owns its state and output slice.

use super::{Linear, Result, TensorSource};
use crate::kernels::conv::causal_conv1d_step;
use crate::kernels::deltanet::{deltanet_step, gate_beta, gate_g, l2norm};
use crate::kernels::matvec::ActVec;
use crate::kernels::rmsnorm::rmsnorm_gated;

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

    /// One token: `x` (hidden) -> `y` (hidden), updating `state`.
    pub fn forward_token(&self, x: &[f32], state: &mut DeltaState, y: &mut [f32], threads: usize) {
        let (n_k, n_v, dk, dv) = (self.n_k, self.n_v, self.dk, self.dv);
        let kd = n_k * dk;
        let vd = n_v * dv;
        let conv_dim = self.conv_dim();
        // one quantisation of `x`, shared by the four projections
        let xa = ActVec::new(x);
        let mut mixed = vec![0f32; conv_dim];
        self.qkv.forward_act(&xa, &mut mixed, threads);
        let mut z = vec![0f32; vd];
        self.gate_z.forward_act(&xa, &mut z, threads);
        let mut a = vec![0f32; n_v];
        self.alpha.forward_act(&xa, &mut a, threads);
        let mut b = vec![0f32; n_v];
        self.beta.forward_act(&xa, &mut b, threads);
        let mut conved = vec![0f32; conv_dim];
        causal_conv1d_step(&mut state.conv, &self.conv_w, self.kernel, &mixed, &mut conved, true);
        let (q_all, rest) = conved.split_at(kd);
        let (k_all, v_all) = rest.split_at(kd);

        let mut heads_out = vec![0f32; vd];
        let scale = 1.0 / (dk as f32).sqrt();
        let run_head = |p: usize, rec: &mut [f32], out: &mut [f32]| {
            let kh = p % n_k;
            let mut qn = q_all[kh * dk..(kh + 1) * dk].to_vec();
            let mut kn = k_all[kh * dk..(kh + 1) * dk].to_vec();
            l2norm(&mut qn, 1e-6);
            l2norm(&mut kn, 1e-6);
            for v in qn.iter_mut() {
                *v *= scale;
            }
            let g = gate_g(a[p], self.dt_bias[p], self.a[p]);
            let beta = gate_beta(b[p]);
            let mut o = vec![0f32; dv];
            deltanet_step(&qn, &kn, &v_all[p * dv..(p + 1) * dv], g, beta, rec, &mut o);
            rmsnorm_gated(&o, &z[p * dv..(p + 1) * dv], &self.norm_w, self.eps, out);
        };
        let threads = threads.max(1).min(n_v);
        if threads == 1 {
            for p in 0..n_v {
                let (rec, out) = (&mut state.rec[p * dk * dv..(p + 1) * dk * dv], &mut heads_out[p * dv..(p + 1) * dv]);
                run_head(p, rec, out);
            }
        } else {
            let chunk = n_v.div_ceil(threads);
            std::thread::scope(|s| {
                for (ci, (recs, outs)) in state.rec.chunks_mut(chunk * dk * dv).zip(heads_out.chunks_mut(chunk * dv)).enumerate() {
                    let run_head = &run_head;
                    s.spawn(move || {
                        for (i, (rec, out)) in recs.chunks_mut(dk * dv).zip(outs.chunks_mut(dv)).enumerate() {
                            run_head(ci * chunk + i, rec, out);
                        }
                    });
                }
            });
        }
        self.out.forward(&heads_out, y, threads);
    }

    /// Prefill: `t` tokens (`x` is `t * hidden`) as sequential steps; `y` is `t * hidden`.
    pub fn forward_prefill(&self, x: &[f32], t: usize, state: &mut DeltaState, y: &mut [f32], threads: usize) {
        let h = self.hidden;
        assert_eq!(x.len(), t * h);
        assert_eq!(y.len(), t * h);
        for i in 0..t {
            let (xi, yi) = (&x[i * h..(i + 1) * h], &mut y[i * h..(i + 1) * h]);
            self.forward_token(xi, state, yi, threads);
        }
    }
}
