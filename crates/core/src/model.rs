//! Phase 3.2: the model and the per-token forward pass; Phase 4: the memory plan and the tiers.
//! `Model::load_with` computes and checks the memory plan before allocating anything, then reads the
//! always-resident set and the tier-1 layers into sector-aligned arenas with unbuffered reads, builds the
//! tier-2 layers as views into the ring slots, and starts the ring's I/O thread. `forward_token` is embed ->
//! 64 decoder layers (48 DeltaNet with carried state, 16 GQA with a KV cache) -> final norm -> lm_head, each
//! streamed layer awaited from its slot and released after use. Greedy decoding only; no sampling, no chat
//! template.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::arch::{arch_consts, stop_ids, ArchConsts};
use crate::config::ModelConfig;
use crate::gguf::Gguf;
use crate::kernels::matvec::{acts_for, matmul, matvec, ActBuf, ActVec, WeightMat};
use crate::kernels::rmsnorm::rmsnorm;
use crate::layers::{DecoderLayer, LayerError, MixerState, Scratch, TensorSource};
use crate::os::{arena_bytes, AlignedBuf, DirectFile, DEFAULT_CHUNK};
use crate::quant::dequantize_into;
use crate::tier::{ArenaSource, LayerRead, MemoryPlan, PlanError, PlanInput, PlanParams, Ring, StreamCounters, StreamStats};

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
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// How to load: the budget (None = everything resident), what the plan sizes the KV cache for, the ring
/// depth, and the read parameters.
#[derive(Debug, Clone)]
pub struct LoadOpts {
    pub budget: Option<u64>,
    pub max_pos: usize,
    pub n_slots: usize,
    /// Try large pages for the arenas and slots (falls back with a note if the privilege is not held).
    pub large_pages: bool,
    /// Bytes per overlapped read request and requests in flight.
    pub chunk: usize,
    pub qd: usize,
    pub baseline: u64,
    /// Print the plan and load progress to stderr.
    pub verbose: bool,
}

impl Default for LoadOpts {
    fn default() -> Self {
        LoadOpts { budget: None, max_pos: 4096, n_slots: 2, large_pages: true, chunk: DEFAULT_CHUNK, qd: 2, baseline: PlanParams::DEFAULT_BASELINE, verbose: false }
    }
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
    /// Bytes of weights held in RAM: arenas (non-layer, MTP, tier 1), ring slots, and the small resident
    /// copies; for the RSS check.
    pub weight_bytes: u64,
    /// Weight bytes one decode token streams through the CPU: every layer, the head, the final norm, one
    /// embedding row (the same whichever tier a layer is in: a streamed layer is read from its slot).
    decode_bytes: u64,
    pub plan: MemoryPlan,
    /// Arenas that have no view holding them (the MTP block); the others live on in their views.
    _arenas: Vec<Arc<AlignedBuf>>,
    ring: Option<Ring>,
    pub sector: u64,
    /// Did any arena get large pages?
    pub large_pages_used: bool,
    /// Bytes and seconds of the unbuffered loading reads.
    pub load_read_bytes: u64,
    pub load_read_secs: f64,
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

    /// Bytes this state holds (capacities): the carried per-layer state and every scratch buffer. The plan's
    /// `scratch_bytes(max_pos) + deltanet_state_bytes() + kv_bytes_per_pos() * max_pos` (less the logits
    /// vector, which the caller owns) must equal it after `reserve(max_pos)`.
    pub fn bytes(&self) -> usize {
        let per_layer: usize = self
            .layers
            .iter()
            .map(|l| match l {
                MixerState::DeltaNet(d) => d.bytes(),
                MixerState::Attention(c) => c.bytes(),
            })
            .sum();
        per_layer + self.scratch.bytes() + (self.h0.capacity() + self.h1.capacity() + self.lm_normed.capacity()) * 4 + self.lm_act.bytes()
    }
}

/// Read the aligned superset of `span` into a fresh arena and return it with its file start.
fn load_arena(file: &mut DirectFile, span: (u64, u64), large: bool, stats: &mut (u64, f64)) -> Result<(Arc<AlignedBuf>, u64), ModelError> {
    let sector = file.sector as u64;
    let r = LayerRead::new(0, span, sector);
    let buf = AlignedBuf::new(arena_bytes(r.span_bytes, sector) as usize, large)?;
    let t0 = Instant::now();
    let got = file.read_at(r.read_start, &buf, r.read_bytes as usize)?;
    if (got as u64) < r.head as u64 + r.span_bytes {
        return Err(ModelError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, format!("short read at {}: {got} of {} bytes", r.read_start, r.read_bytes))));
    }
    stats.0 += got as u64;
    stats.1 += t0.elapsed().as_secs_f64();
    Ok((Arc::new(buf), r.read_start))
}

impl Model {
    /// Everything resident (no budget); `threads` is the pool participation used by every forward.
    pub fn load(path: impl AsRef<Path>, threads: usize) -> Result<Model, ModelError> {
        Self::load_with(path, threads, &LoadOpts::default())
    }

    /// Load under `opts`: the plan is computed and checked first; if it does not fit the budget nothing is
    /// allocated and the error names the shortfall.
    pub fn load_with(path: impl AsRef<Path>, threads: usize, opts: &LoadOpts) -> Result<Model, ModelError> {
        let path = path.as_ref();
        let t0 = Instant::now();
        let g = Gguf::open(path)?;
        let cfg = ModelConfig::from_gguf(&g)?;
        let consts = arch_consts(&cfg.architecture)?;
        let hidden = cfg.hidden_size as usize;
        let vocab = cfg.vocab_size as usize;
        let mut file = DirectFile::open(path, opts.chunk, opts.qd)?;
        let sector = file.sector as u64;

        // ---- the plan, before a single byte is allocated
        let input = PlanInput::new(&g, &cfg);
        let mut params = PlanParams::new(opts.budget, opts.max_pos as u64, opts.n_slots, sector);
        params.baseline = opts.baseline;
        let plan = MemoryPlan::compute(&input, &params);
        if opts.verbose {
            eprint!("{}", plan.table());
        }
        plan.check()?;

        let mut stats = (0u64, 0f64);
        let mut arenas: Vec<Arc<AlignedBuf>> = Vec::new();
        let large = opts.large_pages;
        let mut any_large = false;

        // ---- always resident: the non-layer tensors (one arena when they are contiguous), the MTP blocks
        let (embed, lm_head, output_norm) = match input.non_layer_region {
            Some(region) => {
                let (buf, file_start) = load_arena(&mut file, region, large, &mut stats)?;
                any_large |= buf.large_pages;
                let src = ArenaSource { g: &g, file_start, buf };
                (src.mat("token_embd.weight", vocab, hidden)?, src.mat("output.weight", vocab, hidden)?, src.vec("output_norm.weight", hidden)?)
            }
            None => {
                let src: &dyn TensorSource = &g;
                (src.mat("token_embd.weight", vocab, hidden)?, src.mat("output.weight", vocab, hidden)?, src.vec("output_norm.weight", hidden)?)
            }
        };
        let mut mtp_arena_bytes = 0u64;
        for &(_, span, start) in &input.mtp {
            let (buf, _) = load_arena(&mut file, (start, start + span), large, &mut stats)?;
            any_large |= buf.large_pages;
            mtp_arena_bytes += arena_bytes(span, sector);
            arenas.push(buf);
        }

        // ---- tier 1: whole-layer arenas, filled once
        let mut layers = Vec::with_capacity(cfg.n_layer as usize);
        let mut small_bytes = 0u64;
        let t_tier1 = Instant::now();
        for i in 0..plan.pinned {
            let (buf, file_start) = load_arena(&mut file, input.layer_span[i as usize], large, &mut stats)?;
            any_large |= buf.large_pages;
            let src = ArenaSource { g: &g, file_start, buf };
            layers.push(DecoderLayer::load(&src, &cfg, i)?);
            small_bytes += input.layer_small[i as usize];
            if opts.verbose && (i + 1) % 16 == 0 {
                eprintln!("  tier 1: {} of {} layers in {:.1} s", i + 1, plan.pinned, t_tier1.elapsed().as_secs_f64());
            }
        }

        // ---- tier 2: views into the ring slots; the I/O thread starts prefetching at once
        let mut ring = None;
        if plan.streamed() > 0 {
            let counters = Arc::new(StreamCounters {
                reads_per_layer: (0..cfg.n_layer).map(|_| Default::default()).collect(),
                consumes_per_layer: (0..cfg.n_layer).map(|_| Default::default()).collect(),
                ..Default::default()
            });
            let r = Ring::start(file, &input.layer_span, plan.pinned, plan.n_slots, plan.slot_bytes, large, counters)?;
            for i in plan.pinned..cfg.n_layer {
                let src = ArenaSource { g: &g, file_start: r.reads[i as usize].read_start, buf: r.slot_buf(r.slot_of(i)) };
                layers.push(DecoderLayer::load(&src, &cfg, i)?);
                small_bytes += input.layer_small[i as usize];
            }
            ring = Some(r);
        }

        let large_pages_used = any_large || ring.as_ref().is_some_and(|r| r.large_pages);
        let stop = stop_ids(&cfg.eos_ids, consts);
        let weight_bytes = input.non_layer_arena_bytes(sector) + mtp_arena_bytes + small_bytes + plan.pinned_bytes + plan.ring_bytes;
        let layer_total: u64 = (0..cfg.n_layer as usize).map(|i| input.layer_bytes(i)).sum();
        let decode_bytes = layer_total + lm_head.bytes() as u64 + (output_norm.len() * 4) as u64 + embed.row_bytes as u64;
        if opts.verbose {
            eprintln!(
                "loaded: tier 1 {} layers, tier 2 {} layers; {:.3} GB read unbuffered in {:.1} s ({:.2} GB/s); sector {sector}; large pages {}",
                plan.pinned,
                plan.streamed(),
                stats.0 as f64 / 1e9,
                stats.1,
                stats.0 as f64 / stats.1.max(1e-9) / 1e9,
                if large_pages_used { "yes" } else { "no" }
            );
        }
        Ok(Model {
            cfg,
            consts,
            embed,
            layers,
            output_norm,
            lm_head,
            stop,
            threads,
            load_secs: t0.elapsed().as_secs_f64(),
            weight_bytes,
            decode_bytes,
            plan,
            _arenas: arenas,
            ring,
            sector,
            large_pages_used,
            load_read_bytes: stats.0,
            load_read_secs: stats.1,
        })
    }

    pub fn hidden(&self) -> usize {
        self.cfg.hidden_size as usize
    }

    pub fn vocab(&self) -> usize {
        self.cfg.vocab_size as usize
    }

    /// Streaming counters, if any layer is streamed.
    pub fn stream_stats(&self) -> Option<StreamStats> {
        self.ring.as_ref().map(|r| r.counters.snapshot())
    }

    /// Bytes one pass (one token) reads from disk: the streamed layers' spans.
    pub fn streamed_bytes_per_pass(&self) -> u64 {
        self.plan.streamed_bytes_per_pass
    }

    /// Stop the I/O thread (the counters are final afterwards). Idempotent; `drop` does it too.
    pub fn shutdown_streaming(&mut self) {
        if let Some(r) = &mut self.ring {
            r.shutdown();
        }
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

    /// Wait for layer `i` if it is streamed (no-op for tier 1).
    #[inline]
    fn acquire(&self, i: usize) {
        if let Some(r) = &self.ring {
            if i as u32 >= r.first {
                r.acquire(i as u32);
            }
        }
    }

    #[inline]
    fn release(&self, i: usize) {
        if let Some(r) = &self.ring {
            if i as u32 >= r.first {
                r.release(i as u32);
            }
        }
    }

    fn consumed(&self) -> u64 {
        self.ring.as_ref().map_or(0, |r| r.counters.consumed_bytes.load(std::sync::atomic::Ordering::Acquire))
    }

    /// Rule 4, per pass: the bytes taken from the ring equal the streamed layers' spans exactly.
    fn check_pass(&self, before: u64) {
        let got = self.consumed() - before;
        assert_eq!(got, self.plan.streamed_bytes_per_pass, "disk bytes this pass {got} != streamed layer bytes {}", self.plan.streamed_bytes_per_pass);
    }

    /// Embed and run the layer stack, leaving the residual stream in `state.h0`; advances the state.
    /// Allocation-free (Phase 3.6), streamed layers awaited and released in order (Phase 4).
    fn run_layers(&self, id: u32, state: &mut State) {
        let State { layers: sts, pos, scratch, h0, h1, .. } = state;
        let before = self.consumed();
        self.embed_token_into(id, h0);
        for (i, (layer, st)) in self.layers.iter().zip(sts.iter_mut()).enumerate() {
            self.acquire(i);
            layer.forward_token_in(h0, *pos, st, scratch, h1, self.threads);
            self.release(i);
            std::mem::swap(h0, h1);
        }
        *pos += 1;
        self.check_pass(before);
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
    /// One pass over the layers: each streamed layer is read from disk once for the whole prompt.
    pub fn prefill(&self, ids: &[u32], state: &mut State) -> Vec<f32> {
        let t = ids.len();
        let hidden = self.hidden();
        let mut h = Vec::with_capacity(t * hidden);
        for &id in ids {
            h.extend(self.embed_token(id));
        }
        let mut y = vec![0f32; t * hidden];
        let before = self.consumed();
        for (i, (layer, st)) in self.layers.iter().zip(state.layers.iter_mut()).enumerate() {
            self.acquire(i);
            layer.forward_prefill(&h, t, state.pos, st, &mut y, self.threads);
            self.release(i);
            std::mem::swap(&mut h, &mut y);
        }
        state.pos += t as u32;
        self.check_pass(before);
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

    /// Weight bytes one decode token streams through the CPU: every layer, the norms, the head, and one
    /// embedding row.
    pub fn decode_bytes_per_token(&self) -> u64 {
        self.decode_bytes
    }

    /// One token in, logits out (`vocab` long); the state advances by one position. This is the decode
    /// loop's step and it allocates nothing once `State::reserve` has covered the sequence length
    /// (`tests/decode_alloc.rs` asserts exactly that, resident and streamed).
    pub fn forward_token(&self, id: u32, state: &mut State, logits: &mut [f32]) {
        self.run_layers(id, state);
        let State { h0, lm_normed, lm_act, .. } = state;
        self.logits_with(h0, lm_normed, lm_act, logits);
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop.contains(&id)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // stop the I/O thread before the slots it writes into go away (the views hold them, the ring too)
        self.shutdown_streaming();
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
