//! `Qwen3_5Attention` (modeling_qwen3_5.py lines 632-706), single-token form with a plain KV cache; prefill is
//! T sequential tokens (mathematically the causal prefill). Per head: `[q | gate]` from `attn_q`, q/k RMSNorm
//! (weights with the +1 folded in), partial RoPE on the first `rope_dim` dims, scores over the cache scaled by
//! `head_dim^-0.5`, softmax, weighted V sum, `* sigmoid(gate)`, then `attn_output` projection.
//! Parallel over query heads. `forward_prefill` (Phase 3.3): projections as batched matmuls over the prompt,
//! then causal attention of every prompt position in one pass, the same arithmetic as `t` sequential steps.

use super::{Linear, Result, TensorSource};
use crate::kernels::act::sigmoid;
use crate::kernels::matvec::{acts_for, matmul, matvec, row_range, ActBuf, ActVec};
use crate::kernels::pool::{self, SharedMut};
use crate::kernels::rmsnorm::rmsnorm;
use crate::kernels::rope::{apply_rope, cos_sin, inv_freq};
use crate::kernels::softmax::softmax;
use crate::GgmlType;

/// Preallocated per-token buffers for `forward_token_in` (Phase 3.6). `scores` and `probs` get one row of
/// `max_pos` per head, so the head loop is disjoint and allocation-free; `reserve` sizes them for the
/// longest sequence the state will see.
pub struct AttnScratch {
    pub qg: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub q: Vec<f32>,
    pub gate: Vec<f32>,
    pub tmp: Vec<f32>,
    pub attn: Vec<f32>,
    pub scores: Vec<f32>,
    pub probs: Vec<f32>,
    pub max_pos: usize,
    pub act: ActBuf,
    n_head: usize,
}

impl AttnScratch {
    pub fn new(n_head: usize, n_head_kv: usize, head_dim: usize, rope_dim: usize) -> AttnScratch {
        let (nh, nkv, hd) = (n_head, n_head_kv, head_dim);
        AttnScratch {
            qg: vec![0f32; 2 * nh * hd],
            k: vec![0f32; nkv * hd],
            v: vec![0f32; nkv * hd],
            cos: vec![0f32; rope_dim],
            sin: vec![0f32; rope_dim],
            q: vec![0f32; nh * hd],
            gate: vec![0f32; nh * hd],
            tmp: vec![0f32; hd],
            attn: vec![0f32; nh * hd],
            scores: Vec::new(),
            probs: Vec::new(),
            max_pos: 0,
            act: ActBuf::with_capacity(nh * hd),
            n_head: nh,
        }
    }

    /// Size the per-head score buffers for sequences up to `max_pos` positions. Called by
    /// `model::State::reserve`; without it the head loop grows them on first use (one allocation).
    pub fn reserve(&mut self, max_pos: usize) {
        if max_pos <= self.max_pos {
            return;
        }
        self.max_pos = max_pos;
        self.scores = vec![0f32; self.n_head * max_pos];
        self.probs = vec![0f32; self.n_head * max_pos];
    }

    /// Bytes held (capacities), for the memory plan.
    pub fn bytes(&self) -> usize {
        (self.qg.capacity() + self.k.capacity() + self.v.capacity() + self.cos.capacity() + self.sin.capacity() + self.q.capacity() + self.gate.capacity() + self.tmp.capacity() + self.attn.capacity() + self.scores.capacity() + self.probs.capacity()) * 4 + self.act.bytes()
    }
}

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

impl KvCache {
    /// Preallocate for `max_pos` positions so `forward_token` never grows the cache mid-decode.
    pub fn reserve(&mut self, max_pos: usize) {
        let want = max_pos * self.stride;
        self.k.reserve_exact(want.saturating_sub(self.k.len()));
        self.v.reserve_exact(want.saturating_sub(self.v.len()));
    }

    /// Bytes held (capacities), for the memory plan.
    pub fn bytes(&self) -> usize {
        (self.k.capacity() + self.v.capacity()) * 4
    }
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

    /// The weight types `forward_token_in` needs its `act` filled for.
    pub fn input_types(&self) -> [GgmlType; 3] {
        [self.q.w.ggml_type, self.k.w.ggml_type, self.v.w.ggml_type]
    }

    /// One token at absolute position `pos` (the cache must hold exactly the previous positions).
    pub fn forward_token(&self, x: &[f32], pos: u32, cache: &mut KvCache, y: &mut [f32], threads: usize) {
        let mut sc = AttnScratch::new(self.n_head, self.n_head_kv, self.head_dim, self.rope_dim);
        sc.reserve(cache.len + 1);
        let mut act = ActBuf::with_capacity(x.len());
        act.fill(x, &self.input_types());
        self.forward_token_in(x, &act, pos, cache, &mut sc, y, threads);
    }

    /// `forward_token` with caller-owned buffers: no allocation, provided `sc.reserve` covers the cache
    /// length and the cache itself has capacity. Bit for bit what `forward_token` computes.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token_in(&self, x: &[f32], act: &ActBuf, pos: u32, cache: &mut KvCache, sc: &mut AttnScratch, y: &mut [f32], threads: usize) {
        let (nh, nkv, hd, rd) = (self.n_head, self.n_head_kv, self.head_dim, self.rope_dim);
        let group = nh / nkv;
        matvec(&self.q.w, act.act_for(self.q.w.ggml_type, x), &mut sc.qg[..2 * nh * hd], threads);
        matvec(&self.k.w, act.act_for(self.k.w.ggml_type, x), &mut sc.k[..nkv * hd], threads);
        matvec(&self.v.w, act.act_for(self.v.w.ggml_type, x), &mut sc.v[..nkv * hd], threads);

        {
            crate::prof_scope!(crate::prof::Stage::Rope);
            let AttnScratch { qg, k, cos, sin, q, gate, tmp, .. } = &mut *sc;
            cos_sin(pos, &self.inv_freq, &mut cos[..rd], &mut sin[..rd]);
            // q (normed, roped) and gates, per head
            for h in 0..nh {
                let src = &qg[h * 2 * hd..(h + 1) * 2 * hd];
                rmsnorm(&src[..hd], &self.q_norm, self.eps, &mut tmp[..hd]);
                apply_rope(&mut tmp[..hd], hd, rd, &cos[..rd], &sin[..rd]);
                q[h * hd..(h + 1) * hd].copy_from_slice(&tmp[..hd]);
                gate[h * hd..(h + 1) * hd].copy_from_slice(&src[hd..]);
            }
            for h in 0..nkv {
                let kh = &mut k[h * hd..(h + 1) * hd];
                rmsnorm(kh, &self.k_norm, self.eps, &mut tmp[..hd]);
                apply_rope(&mut tmp[..hd], hd, rd, &cos[..rd], &sin[..rd]);
                kh.copy_from_slice(&tmp[..hd]);
            }
        }
        cache.k.extend_from_slice(&sc.k[..nkv * hd]);
        cache.v.extend_from_slice(&sc.v[..nkv * hd]);
        cache.len += 1;
        let len = cache.len;
        sc.reserve(len);
        let scaling = (hd as f32).powf(-0.5);

        {
            let AttnScratch { q, gate, attn, scores, probs, max_pos, .. } = &mut *sc;
            let max_pos = *max_pos;
            let (q, gate): (&[f32], &[f32]) = (&q[..nh * hd], &gate[..nh * hd]);
            let cache_ref: &KvCache = cache;
            let attn_p = SharedMut::new(&mut attn[..nh * hd]);
            let scores_p = SharedMut::new(&mut scores[..nh * max_pos]);
            let probs_p = SharedMut::new(&mut probs[..nh * max_pos]);
            crate::prof_scope!(crate::prof::Stage::Attn);
            pool::global().run(threads.max(1).min(nh), &|tid, n| {
                let (h0, h1) = row_range(nh, tid, n);
                for h in h0..h1 {
                    let kvh = h / group;
                    let qh = &q[h * hd..(h + 1) * hd];
                    // SAFETY: head `h` is handled by exactly one participant; each range is that head's own row.
                    let (s_h, p_h, out) = unsafe { (scores_p.slice(h * max_pos, len), probs_p.slice(h * max_pos, len), attn_p.slice(h * hd, hd)) };
                    for (t, sc) in s_h.iter_mut().enumerate() {
                        let kt = &cache_ref.k[t * cache_ref.stride + kvh * hd..t * cache_ref.stride + (kvh + 1) * hd];
                        let mut s = 0f32;
                        for d in 0..hd {
                            s += qh[d] * kt[d];
                        }
                        *sc = s * scaling;
                    }
                    {
                        crate::prof_scope!(crate::prof::Stage::Softmax);
                        softmax(s_h, p_h);
                    }
                    for o in out.iter_mut() {
                        *o = 0.0;
                    }
                    for (t, &pt) in p_h.iter().enumerate() {
                        let vt = &cache_ref.v[t * cache_ref.stride + kvh * hd..t * cache_ref.stride + (kvh + 1) * hd];
                        for d in 0..hd {
                            out[d] += pt * vt[d];
                        }
                    }
                    for d in 0..hd {
                        out[d] *= sigmoid(gate[h * hd + d]);
                    }
                }
            });
        }
        let ot = self.o.w.ggml_type;
        sc.act.fill(&sc.attn[..nh * hd], &[ot]);
        matvec(&self.o.w, sc.act.act_for(ot, &sc.attn[..nh * hd]), y, threads);
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
