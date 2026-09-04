//! Phase 3.2: the whole model resident in RAM and the per-token forward pass.
//! `Model::load` reads every tensor of the GGUF into `WeightMat`s in ggml layout (no dequantisation except the
//! small F32 vectors); `forward_token` is embed -> 64 decoder layers (48 DeltaNet with carried state, 16 GQA
//! with a KV cache) -> final norm -> lm_head. Greedy decoding only; no sampling, no chat template, no tiers.

use std::path::Path;
use std::time::Instant;

use crate::arch::{arch_consts, stop_ids, ArchConsts};
use crate::config::ModelConfig;
use crate::gguf::Gguf;
use crate::kernels::matvec::{acts_for, matmul, matvec, ActBuf, ActVec, WeightMat};
use crate::kernels::rmsnorm::rmsnorm;
use crate::layers::{DecoderLayer, LayerError, MixerState, Scratch, TensorSource};
use crate::quant::dequantize_into;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Gguf(#[from] crate::gguf::GgufError),
    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),
    #[error(transparent)]
    Layer(#[from] LayerError),
    #[error(transparent)]
    Arch(#[from] crate::arch::UnknownArch),
}

pub struct Model {
    pub cfg: ModelConfig,
    pub consts: &'static ArchConsts,
    /// `token_embd.weight`, `vocab x hidden` in ggml layout; one row is dequantised per token.
    pub embed: WeightMat,
    pub layers: Vec<DecoderLayer>,
    pub output_norm: Vec<f32>,
    /// `output.weight`, `vocab x hidden`.
    pub lm_head: WeightMat,
    /// GGUF EOS ids plus the architecture's extra stop ids.
    pub stop: Vec<u32>,
    pub threads: usize,
    /// Seconds spent in `load`.
    pub load_secs: f64,
    /// Bytes of weights held (`WeightMat` data plus f32 vectors), for the RSS check.
    pub weight_bytes: u64,
}

/// Carried state of one sequence: per-layer DeltaNet state / KV cache, the next position, and every
/// per-token scratch buffer the decode path uses (Phase 3.6). `Model::forward_token` allocates nothing
/// once `reserve` has covered the sequence length: the buffers here are the whole working set.
pub struct State {
    pub layers: Vec<MixerState>,
    pub pos: u32,
    /// Per-token buffers shared by every layer in turn.
    pub scratch: Scratch,
    /// Residual-stream ping-pong: the layer stack reads `h0` and writes `h1`, then swaps.
    h0: Vec<f32>,
    h1: Vec<f32>,
    /// Final-norm output and its quantised form, for the lm_head.
    lm_normed: Vec<f32>,
    lm_act: ActBuf,
}

impl State {
    /// Preallocate for sequences up to `max_pos` positions: the KV caches and the attention score buffers.
    /// Without it the first token past the current capacity grows them, which is an allocation.
    pub fn reserve(&mut self, max_pos: usize) {
        self.scratch.reserve(max_pos);
        for l in &mut self.layers {
            if let MixerState::Attention(c) = l {
                c.reserve(max_pos);
            }
        }
    }

    /// The residual stream after the last layer of the most recent `forward_token` / `forward_hidden`.
    pub fn hidden(&self) -> &[f32] {
        &self.h0
    }
}

impl Model {
    /// Read every tensor into RAM. `threads` is the pool participation used by every forward.
    pub fn load(path: impl AsRef<Path>, threads: usize) -> Result<Model, ModelError> {
        let t0 = Instant::now();
        let g = Gguf::open(path)?;
        let cfg = ModelConfig::from_gguf(&g)?;
        let consts = arch_consts(&cfg.architecture)?;
        let hidden = cfg.hidden_size as usize;
        let vocab = cfg.vocab_size as usize;
        let src: &dyn TensorSource = &g;
        let embed = src.mat("token_embd.weight", vocab, hidden)?;
        let mut layers = Vec::with_capacity(cfg.n_layer as usize);
        for i in 0..cfg.n_layer {
            layers.push(DecoderLayer::load(src, &cfg, i)?);
        }
        let output_norm = src.vec("output_norm.weight", hidden)?;
        let lm_head = src.mat("output.weight", vocab, hidden)?;
        let stop = stop_ids(&cfg.eos_ids, consts);
        let weight_bytes = g.tensor_bytes_read();
        Ok(Model { cfg, consts, embed, layers, output_norm, lm_head, stop, threads, load_secs: t0.elapsed().as_secs_f64(), weight_bytes })
    }

    pub fn hidden(&self) -> usize {
        self.cfg.hidden_size as usize
    }

    pub fn vocab(&self) -> usize {
        self.cfg.vocab_size as usize
    }

    pub fn new_state(&self) -> State {
        let hidden = self.hidden();
        State {
            layers: self.layers.iter().map(|l| l.new_state()).collect(),
            pos: 0,
            scratch: Scratch::for_layers(&self.layers, hidden),
            h0: vec![0f32; hidden],
            h1: vec![0f32; hidden],
            lm_normed: vec![0f32; hidden],
            lm_act: ActBuf::with_capacity(hidden),
        }
    }

    /// Bytes the carried state occupies at `n_pos` positions (DeltaNet state + conv window + KV cache).
    pub fn state_bytes(&self, n_pos: usize) -> u64 {
        let c = &self.cfg;
        let dn = c.deltanet_layers.len() as u64 * (c.dn_state_elems_per_layer + c.dn_conv_state_elems_per_layer) * 4;
        let kv = c.full_attention_layers.len() as u64 * 2 * (c.n_head_kv as u64 * c.head_dim_k as u64) * 4 * n_pos as u64;
        dn + kv
    }

    /// The embedding row of `id` as f32.
    pub fn embed_token(&self, id: u32) -> Vec<f32> {
        let mut out = vec![0f32; self.hidden()];
        self.embed_token_into(id, &mut out);
        out
    }

    /// The embedding row of `id` into a caller-owned buffer (no allocation).
    pub fn embed_token_into(&self, id: u32, out: &mut [f32]) {
        crate::prof_scope!(crate::prof::Stage::Embed);
        assert!((id as usize) < self.embed.rows, "token id {id} outside the vocabulary");
        dequantize_into(self.embed.ggml_type, self.embed.row(id as usize), out).expect("embedding row");
    }

    /// Embed and run the layer stack, leaving the residual stream in `state.h0`; advances the state.
    /// Allocation-free (Phase 3.6).
    fn run_layers(&self, id: u32, state: &mut State) {
        let State { layers: sts, pos, scratch, h0, h1, .. } = state;
        self.embed_token_into(id, h0);
        for (layer, st) in self.layers.iter().zip(sts.iter_mut()) {
            layer.forward_token_in(h0, *pos, st, scratch, h1, self.threads);
            std::mem::swap(h0, h1);
        }
        *pos += 1;
    }

    /// The residual stream after the last layer for token `id` at the state's position; advances the state.
    /// Allocates the returned vector: the decode loop uses `forward_token`, which does not.
    pub fn forward_hidden(&self, id: u32, state: &mut State) -> Vec<f32> {
        self.run_layers(id, state);
        state.h0.clone()
    }

    /// Final norm and lm_head of a residual-stream vector.
    pub fn logits_of(&self, h: &[f32], logits: &mut [f32]) {
        let mut normed = vec![0f32; self.hidden()];
        let mut act = ActBuf::with_capacity(self.hidden());
        self.logits_with(h, &mut normed, &mut act, logits);
    }

    /// `logits_of` on caller-owned buffers (no allocation).
    fn logits_with(&self, h: &[f32], normed: &mut [f32], act: &mut ActBuf, logits: &mut [f32]) {
        {
            crate::prof_scope!(crate::prof::Stage::Norm);
            rmsnorm(h, &self.output_norm, self.cfg.rms_norm_eps, normed);
        }
        let t = self.lm_head.ggml_type;
        act.fill(normed, &[t]);
        matvec(&self.lm_head, act.act_for(t, normed), logits, self.threads);
    }

    /// Batched prefill (Phase 3.3): the residual stream after the last layer for every token of `ids`
    /// (`t * hidden`), the state advanced by `t`. Row `i` is `forward_hidden` on token `i` bit for bit.
    pub fn prefill(&self, ids: &[u32], state: &mut State) -> Vec<f32> {
        let t = ids.len();
        let hidden = self.hidden();
        let mut h = Vec::with_capacity(t * hidden);
        for &id in ids {
            h.extend(self.embed_token(id));
        }
        let mut y = vec![0f32; t * hidden];
        for (layer, st) in self.layers.iter().zip(state.layers.iter_mut()) {
            layer.forward_prefill(&h, t, state.pos, st, &mut y, self.threads);
            std::mem::swap(&mut h, &mut y);
        }
        state.pos += t as u32;
        h
    }

    /// Final norm and lm_head for `t` residual-stream rows (`t * hidden` in, `t * vocab` out), the head read once.
    pub fn logits_all(&self, hs: &[f32], t: usize) -> Vec<f32> {
        let hidden = self.hidden();
        assert_eq!(hs.len(), t * hidden);
        let mut normed = vec![0f32; t * hidden];
        for i in 0..t {
            rmsnorm(&hs[i * hidden..(i + 1) * hidden], &self.output_norm, self.cfg.rms_norm_eps, &mut normed[i * hidden..(i + 1) * hidden]);
        }
        let acts: Vec<ActVec<'_>> = normed.chunks_exact(hidden).map(ActVec::new).collect();
        let mut logits = vec![0f32; t * self.vocab()];
        matmul(&self.lm_head, &acts_for(&acts, self.lm_head.ggml_type), &mut logits, self.threads);
        logits
    }

    /// Weight bytes one decode token streams: every layer, the norms, the head, and one embedding row.
    pub fn decode_bytes_per_token(&self) -> u64 {
        self.weight_bytes - self.embed.bytes() as u64 + self.embed.row_bytes as u64
    }

    /// One token in, logits out (`vocab` long); the state advances by one position. This is the decode
    /// loop's step and it allocates nothing once `State::reserve` has covered the sequence length
    /// (`tests/decode_alloc.rs` asserts exactly that).
    pub fn forward_token(&self, id: u32, state: &mut State, logits: &mut [f32]) {
        self.run_layers(id, state);
        let State { h0, lm_normed, lm_act, .. } = state;
        self.logits_with(h0, lm_normed, lm_act, logits);
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop.contains(&id)
    }
}

/// Index of the largest logit (first on ties).
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}
