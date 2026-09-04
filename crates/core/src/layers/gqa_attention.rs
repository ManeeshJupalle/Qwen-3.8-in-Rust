//! `Qwen3_5Attention` (modeling_qwen3_5.py lines 632-706), single-token form with a plain KV cache; prefill is
//! T sequential tokens (mathematically the causal prefill). Per head: `[q | gate]` from `attn_q`, q/k RMSNorm
//! (weights with the +1 folded in), partial RoPE on the first `rope_dim` dims, scores over the cache scaled by
//! `head_dim^-0.5`, softmax, weighted V sum, `* sigmoid(gate)`, then `attn_output` projection.
//! Parallel over query heads. `forward_prefill` (Phase 3.3): projections as batched matmuls over the prompt,
//! then causal attention of every prompt position in one pass, the same arithmetic as `t` sequential steps.

use super::{Linear, Result, TensorSource};
use crate::kernels::act::sigmoid;
use crate::kernels::matvec::{acts_for, matmul, row_range, ActVec};
use crate::kernels::pool::{self, SharedMut};
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
        let xa = ActVec::new(x);
        let mut qg = vec![0f32; 2 * nh * hd];
        self.q.forward_act(&xa, &mut qg, threads);
        let mut k = vec![0f32; nkv * hd];
        self.k.forward_act(&xa, &mut k, threads);
        let mut v = vec![0f32; nkv * hd];
        self.v.forward_act(&xa, &mut v, threads);

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

    /// Prefill `t` tokens starting at `start_pos`: q/k/v and the output projection as batched matmuls, the KV
    /// cache filled for all `t` positions, then every query attends causally over the cache in one pass
    /// (parallel over heads). Row `i` equals `forward_token` on row `i` bit for bit.
    pub fn forward_prefill(&self, x: &[f32], t: usize, start_pos: u32, cache: &mut KvCache, y: &mut [f32], threads: usize) {
        let (nh, nkv, hd, rd) = (self.n_head, self.n_head_kv, self.head_dim, self.rope_dim);
        let h = self.hidden;
        let group = nh / nkv;
        assert_eq!(x.len(), t * h);
        assert_eq!(y.len(), t * h);
        let acts: Vec<ActVec<'_>> = x.chunks_exact(h).map(ActVec::new).collect();
        let mut qg = vec![0f32; t * 2 * nh * hd];
        matmul(&self.q.w, &acts_for(&acts, self.q.w.ggml_type), &mut qg, threads);
        let mut k = vec![0f32; t * nkv * hd];
        matmul(&self.k.w, &acts_for(&acts, self.k.w.ggml_type), &mut k, threads);
        let mut v = vec![0f32; t * nkv * hd];
        matmul(&self.v.w, &acts_for(&acts, self.v.w.ggml_type), &mut v, threads);

        let base = cache.len;
        let mut q_all = vec![0f32; t * nh * hd];
        let mut gate_all = vec![0f32; t * nh * hd];
        let mut cos = vec![0f32; rd];
        let mut sin = vec![0f32; rd];
        let mut tmp = vec![0f32; hd];
        for i in 0..t {
            cos_sin(start_pos + i as u32, &self.inv_freq, &mut cos, &mut sin);
            for hh in 0..nh {
                let src = &qg[i * 2 * nh * hd + hh * 2 * hd..i * 2 * nh * hd + (hh + 1) * 2 * hd];
                rmsnorm(&src[..hd], &self.q_norm, self.eps, &mut tmp);
                apply_rope(&mut tmp, hd, rd, &cos, &sin);
                q_all[i * nh * hd + hh * hd..i * nh * hd + (hh + 1) * hd].copy_from_slice(&tmp);
                gate_all[i * nh * hd + hh * hd..i * nh * hd + (hh + 1) * hd].copy_from_slice(&src[hd..]);
            }
            let krow = &mut k[i * nkv * hd..(i + 1) * nkv * hd];
            for hh in 0..nkv {
                let kh = &mut krow[hh * hd..(hh + 1) * hd];
                rmsnorm(kh, &self.k_norm, self.eps, &mut tmp);
                apply_rope(&mut tmp, hd, rd, &cos, &sin);
                kh.copy_from_slice(&tmp);
            }
            cache.k.extend_from_slice(krow);
            cache.v.extend_from_slice(&v[i * nkv * hd..(i + 1) * nkv * hd]);
            cache.len += 1;
        }
        let scaling = (hd as f32).powf(-0.5);
        let mut attn = vec![0f32; t * nh * hd];
        let attn_p = SharedMut::new(&mut attn);
        let cache_ref: &KvCache = cache;
        pool::global().run(threads.max(1).min(nh), &|tid, n| {
            let (h0, h1) = row_range(nh, tid, n);
            for hh in h0..h1 {
                let kvh = hh / group;
                for i in 0..t {
                    let len = base + i + 1;
                    let qh = &q_all[i * nh * hd + hh * hd..i * nh * hd + (hh + 1) * hd];
                    let mut scores = vec![0f32; len];
                    for (tt, sc) in scores.iter_mut().enumerate() {
                        let kt = &cache_ref.k[tt * cache_ref.stride + kvh * hd..tt * cache_ref.stride + (kvh + 1) * hd];
                        let mut s = 0f32;
                        for d in 0..hd {
                            s += qh[d] * kt[d];
                        }
                        *sc = s * scaling;
                    }
                    let mut p = vec![0f32; len];
                    softmax(&scores, &mut p);
                    // SAFETY: head `hh` is handled by exactly one participant; its columns of every row are disjoint.
                    let out = unsafe { attn_p.slice(i * nh * hd + hh * hd, hd) };
                    for o in out.iter_mut() {
                        *o = 0.0;
                    }
                    for (tt, &pt) in p.iter().enumerate() {
                        let vt = &cache_ref.v[tt * cache_ref.stride + kvh * hd..tt * cache_ref.stride + (kvh + 1) * hd];
                        for d in 0..hd {
                            out[d] += pt * vt[d];
                        }
                    }
                    for d in 0..hd {
                        out[d] *= sigmoid(gate_all[i * nh * hd + hh * hd + d]);
                    }
                }
            }
        });
        let aacts: Vec<ActVec<'_>> = attn.chunks_exact(nh * hd).map(ActVec::new).collect();
        matmul(&self.o.w, &acts_for(&aacts, self.o.w.ggml_type), y, threads);
    }
}
