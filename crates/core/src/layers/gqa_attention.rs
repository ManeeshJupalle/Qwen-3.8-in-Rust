//! `Qwen3_5Attention` (modeling_qwen3_5.py lines 632-706), single-token form with a plain KV cache; prefill is
//! T sequential tokens (mathematically the causal prefill). Per head: `[q | gate]` from `attn_q`, q/k RMSNorm
//! (weights with the +1 folded in), partial RoPE on the first `rope_dim` dims, scores over the cache scaled by
//! `head_dim^-0.5`, softmax, weighted V sum, `* sigmoid(gate)`, then `attn_output` projection.
//! Parallel over query heads with `std::thread::scope`.

use super::{Linear, Result, TensorSource};
use crate::kernels::act::sigmoid;
use crate::kernels::rmsnorm::rmsnorm;
use crate::kernels::rope::{apply_rope, cos_sin, inv_freq};
use crate::kernels::softmax::softmax;

pub struct GqaAttention {
    pub hidden: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub eps: f32,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub inv_freq: Vec<f32>,
}

/// Post-RoPE keys and values, `len` positions, each `n_head_kv * head_dim`.
#[derive(Debug, Clone, PartialEq)]
pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
    pub stride: usize,
}

impl GqaAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn load(src: &dyn TensorSource, prefix: &str, hidden: usize, n_head: usize, n_head_kv: usize, head_dim: usize, rope_dim: usize, theta: f32, eps: f32) -> Result<GqaAttention> {
        Ok(GqaAttention {
            hidden,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            eps,
            q: Linear::load(src, &format!("{prefix}attn_q.weight"), 2 * n_head * head_dim, hidden)?,
            k: Linear::load(src, &format!("{prefix}attn_k.weight"), n_head_kv * head_dim, hidden)?,
            v: Linear::load(src, &format!("{prefix}attn_v.weight"), n_head_kv * head_dim, hidden)?,
            o: Linear::load(src, &format!("{prefix}attn_output.weight"), hidden, n_head * head_dim)?,
            q_norm: src.vec(&format!("{prefix}attn_q_norm.weight"), head_dim)?,
            k_norm: src.vec(&format!("{prefix}attn_k_norm.weight"), head_dim)?,
            inv_freq: inv_freq(rope_dim, theta),
        })
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache { k: Vec::new(), v: Vec::new(), len: 0, stride: self.n_head_kv * self.head_dim }
    }

    /// One token at absolute position `pos` (the cache must hold exactly the previous positions).
    pub fn forward_token(&self, x: &[f32], pos: u32, cache: &mut KvCache, y: &mut [f32], threads: usize) {
        let (nh, nkv, hd, rd) = (self.n_head, self.n_head_kv, self.head_dim, self.rope_dim);
        let group = nh / nkv;
        let mut qg = vec![0f32; 2 * nh * hd];
        self.q.forward(x, &mut qg, threads);
        let mut k = vec![0f32; nkv * hd];
        self.k.forward(x, &mut k, threads);
        let mut v = vec![0f32; nkv * hd];
        self.v.forward(x, &mut v, threads);

        let mut cos = vec![0f32; rd];
        let mut sin = vec![0f32; rd];
        cos_sin(pos, &self.inv_freq, &mut cos, &mut sin);

        // q (normed, roped) and gates, per head
        let mut q = vec![0f32; nh * hd];
        let mut gate = vec![0f32; nh * hd];
        let mut tmp = vec![0f32; hd];
        for h in 0..nh {
            let src = &qg[h * 2 * hd..(h + 1) * 2 * hd];
            rmsnorm(&src[..hd], &self.q_norm, self.eps, &mut tmp);
            apply_rope(&mut tmp, hd, rd, &cos, &sin);
            q[h * hd..(h + 1) * hd].copy_from_slice(&tmp);
            gate[h * hd..(h + 1) * hd].copy_from_slice(&src[hd..]);
        }
        for h in 0..nkv {
            let kh = &mut k[h * hd..(h + 1) * hd];
            rmsnorm(kh, &self.k_norm, self.eps, &mut tmp);
            apply_rope(&mut tmp, hd, rd, &cos, &sin);
            kh.copy_from_slice(&tmp);
        }
        cache.k.extend_from_slice(&k);
        cache.v.extend_from_slice(&v);
        cache.len += 1;
        let len = cache.len;
        let scaling = (hd as f32).powf(-0.5);

        let mut attn = vec![0f32; nh * hd];
        let run_head = |h: usize, out: &mut [f32]| {
            let kvh = h / group;
            let qh = &q[h * hd..(h + 1) * hd];
            let mut scores = vec![0f32; len];
            for (t, sc) in scores.iter_mut().enumerate() {
                let kt = &cache.k[t * cache.stride + kvh * hd..t * cache.stride + (kvh + 1) * hd];
                let mut s = 0f32;
                for d in 0..hd {
                    s += qh[d] * kt[d];
                }
                *sc = s * scaling;
            }
            let mut p = vec![0f32; len];
            softmax(&scores, &mut p);
            for o in out.iter_mut() {
                *o = 0.0;
            }
            for (t, &pt) in p.iter().enumerate() {
                let vt = &cache.v[t * cache.stride + kvh * hd..t * cache.stride + (kvh + 1) * hd];
                for d in 0..hd {
                    out[d] += pt * vt[d];
                }
            }
            for d in 0..hd {
                out[d] *= sigmoid(gate[h * hd + d]);
            }
        };
        let threads = threads.max(1).min(nh);
        if threads == 1 {
            for h in 0..nh {
                run_head(h, &mut attn[h * hd..(h + 1) * hd]);
            }
        } else {
            let chunk = nh.div_ceil(threads);
            std::thread::scope(|s| {
                for (ci, outs) in attn.chunks_mut(chunk * hd).enumerate() {
                    let run_head = &run_head;
                    s.spawn(move || {
                        for (i, out) in outs.chunks_mut(hd).enumerate() {
                            run_head(ci * chunk + i, out);
                        }
                    });
                }
            });
        }
        self.o.forward(&attn, y, threads);
    }

    /// Prefill `t` tokens starting at `start_pos` as sequential steps.
    pub fn forward_prefill(&self, x: &[f32], t: usize, start_pos: u32, cache: &mut KvCache, y: &mut [f32], threads: usize) {
        let h = self.hidden;
        assert_eq!(x.len(), t * h);
        assert_eq!(y.len(), t * h);
        for i in 0..t {
            self.forward_token(&x[i * h..(i + 1) * h], start_pos + i as u32, cache, &mut y[i * h..(i + 1) * h], threads);
        }
    }
}
