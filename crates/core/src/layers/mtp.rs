//! The MTP draft head (Phase 5.1, `docs/mtp.md`): `blk.{n_layer}` of the GGUF. One row is a token `x` and a
//! target hidden `h` (the post-final-norm vector the lm_head read): `u = eh_proj . [rmsnorm(embed(x), enorm) |
//! rmsnorm(h, hnorm)]`, then one attention decoder block (`DecoderLayer`, the same code as the 16 GQA layers,
//! with its own KV cache: row `j` sits at position `j`), then `h' = rmsnorm(block(u), shared_head_norm)`. The
//! caller runs the shared `output.weight` on `h'` for the draft logits and feeds `h'` back as the next chained
//! row's hidden. The embedding lookup stays with the model (it owns `token_embd`); this module takes the
//! embedding row.

use super::gqa_attention::KvCache;
use super::{BatchScratch, DecoderLayer, Linear, Mixer, MixerState, Result, Scratch, TensorSource};
use crate::config::ModelConfig;
use crate::kernels::matvec::{matmul_fn, matvec, ActBuf};
use crate::kernels::rmsnorm::rmsnorm;
use crate::GgmlType;

pub struct MtpHead {
    pub index: u32,
    pub hidden: usize,
    pub eps: f32,
    /// `nextn.eh_proj.weight`, `hidden x 2 hidden`.
    pub eh_proj: Linear,
    pub enorm: Vec<f32>,
    pub hnorm: Vec<f32>,
    /// `nextn.shared_head_norm.weight`.
    pub head_norm: Vec<f32>,
    pub block: DecoderLayer,
}

/// Every buffer the draft step needs (Phase 5: allocation-free like the decode step): `max_t` rows of the
/// concatenated input, their activation buffers, the block's input / output rows, the post-norm hiddens; the
/// block's per-token and batch scratch; the head's activation buffer and the draft logits.
pub struct MtpScratch {
    pub max_t: usize,
    pub e: Vec<f32>,
    /// The previous row's post-norm hidden, for a chained row.
    pub hprev: Vec<f32>,
    pub cat: Vec<f32>,
    pub acts: Vec<ActBuf>,
    pub u: Vec<f32>,
    pub y: Vec<f32>,
    pub hout: Vec<f32>,
    pub sc: Scratch,
    pub bsc: BatchScratch,
    pub head_act: ActBuf,
    pub logits: Vec<f32>,
}

impl MtpHead {
    pub fn load(src: &dyn TensorSource, cfg: &ModelConfig, index: u32) -> Result<MtpHead> {
        let hidden = cfg.hidden_size as usize;
        let p = format!("blk.{index}.nextn.");
        let block = DecoderLayer::load_kind(src, cfg, index, true)?;
        assert!(matches!(block.mixer, Mixer::Attention(_)));
        Ok(MtpHead {
            index,
            hidden,
            eps: cfg.rms_norm_eps,
            eh_proj: Linear::load(src, &format!("{p}eh_proj.weight"), hidden, 2 * hidden)?,
            enorm: src.vec(&format!("{p}enorm.weight"), hidden)?,
            hnorm: src.vec(&format!("{p}hnorm.weight"), hidden)?,
            head_norm: src.vec(&format!("{p}shared_head_norm.weight"), hidden)?,
            block,
        })
    }

    pub fn new_cache(&self) -> MixerState {
        self.block.new_state()
    }

    pub fn new_scratch(&self, vocab: usize, max_t: usize) -> MtpScratch {
        let h = self.hidden;
        MtpScratch {
            max_t,
            e: vec![0f32; h],
            hprev: vec![0f32; h],
            cat: vec![0f32; max_t * 2 * h],
            acts: (0..max_t).map(|_| ActBuf::with_capacity(2 * h)).collect(),
            u: vec![0f32; max_t * h],
            y: vec![0f32; max_t * h],
            hout: vec![0f32; max_t * h],
            sc: Scratch::for_layers(std::slice::from_ref(&self.block), h),
            bsc: BatchScratch::for_layers(std::slice::from_ref(&self.block), h, max_t),
            head_act: ActBuf::with_capacity(h),
            logits: vec![0f32; vocab],
        }
    }

    pub fn input_type(&self) -> GgmlType {
        self.eh_proj.w.ggml_type
    }

    /// Row `i` of `sc.cat` from an embedding row (written into `sc.e` by `embed`) and a hidden (`h`, or the
    /// scratch's `hprev` when `None`: the chained case): `[rmsnorm(e, enorm) | rmsnorm(h, hnorm)]`, and its
    /// activation buffer filled.
    pub fn set_row_with(&self, sc: &mut MtpScratch, i: usize, h: Option<&[f32]>, embed: &dyn Fn(&mut [f32])) {
        let hd = self.hidden;
        let MtpScratch { e, hprev, cat, acts, .. } = sc;
        embed(e);
        let h: &[f32] = h.unwrap_or(hprev);
        let cat = &mut cat[i * 2 * hd..(i + 1) * 2 * hd];
        rmsnorm(e, &self.enorm, self.eps, &mut cat[..hd]);
        rmsnorm(h, &self.hnorm, self.eps, &mut cat[hd..]);
        acts[i].fill(cat, &[self.eh_proj.w.ggml_type]);
    }

    /// `t` rows already set with `set_row`, at positions `cache.len..`: `eh_proj`, the block (batched), the head
    /// norm; the post-norm hiddens land in `sc.hout[..t * hidden]`. No allocation.
    pub fn rows_in(&self, t: usize, cache: &mut MixerState, sc: &mut MtpScratch, threads: usize) {
        let hd = self.hidden;
        assert!(t <= sc.max_t, "MtpScratch: {t} rows > max_t {}", sc.max_t);
        let et = self.eh_proj.w.ggml_type;
        let MtpScratch { cat, acts, u, y, hout, sc: tok_sc, bsc, .. } = sc;
        let cat: &[f32] = cat;
        matmul_fn(&self.eh_proj.w, t, &|i| acts[i].act_for(et, &cat[i * 2 * hd..(i + 1) * 2 * hd]), &mut u[..t * hd], threads);
        let start = match &*cache {
            MixerState::Attention(c) => c.len as u32,
            _ => unreachable!(),
        };
        if t == 1 {
            self.block.forward_token_in(&u[..hd], start, cache, tok_sc, &mut y[..hd], threads);
        } else {
            self.block.forward_batch_in(&u[..t * hd], t, start, cache, bsc, tok_sc, &mut y[..t * hd], threads);
        }
        for i in 0..t {
            rmsnorm(&y[i * hd..(i + 1) * hd], &self.head_norm, self.eps, &mut hout[i * hd..(i + 1) * hd]);
        }
    }

    /// The draft logits of row `i` of the last `rows_in`: the shared lm_head on `hout[i]`, into `sc.logits`.
    pub fn head_in(&self, lm_head: &crate::kernels::matvec::WeightMat, sc: &mut MtpScratch, i: usize, threads: usize) {
        let hd = self.hidden;
        let t = lm_head.ggml_type;
        let MtpScratch { hout, head_act, logits, .. } = sc;
        let h = &hout[i * hd..(i + 1) * hd];
        head_act.fill(h, &[t]);
        matvec(lm_head, head_act.act_for(t, h), logits, threads);
    }

    /// The MTP's KV cache length: rows consumed so far.
    pub fn rows(cache: &MixerState) -> usize {
        match cache {
            MixerState::Attention(c) => c.len,
            _ => unreachable!(),
        }
    }

    /// Forget rows `len..` of the MTP cache (no allocation).
    pub fn truncate(cache: &mut MixerState, len: usize) {
        match cache {
            MixerState::Attention(c) => c.truncate(len),
            _ => unreachable!(),
        }
    }

    pub fn cache_of(cache: &MixerState) -> &KvCache {
        match cache {
            MixerState::Attention(c) => c,
            _ => unreachable!(),
        }
    }
}

impl MtpScratch {
    /// Size the block's per-head score rows for `max_pos` rows.
    pub fn reserve(&mut self, max_pos: usize) {
        self.sc.reserve(max_pos);
    }

    /// Copy row `i` of `hout` into `hprev` (the next chained row's hidden).
    pub fn take_prev(&mut self, i: usize) {
        let h = self.hprev.len();
        self.hprev.copy_from_slice(&self.hout[i * h..(i + 1) * h]);
    }

    /// Bytes held (capacities), for the memory plan.
    pub fn bytes(&self) -> usize {
        (self.e.capacity() + self.hprev.capacity() + self.cat.capacity() + self.u.capacity() + self.y.capacity() + self.hout.capacity() + self.logits.capacity()) * 4
            + self.acts.iter().map(|a| a.bytes()).sum::<usize>()
            + self.sc.bytes()
            + self.bsc.bytes()
            + self.head_act.bytes()
    }
}
