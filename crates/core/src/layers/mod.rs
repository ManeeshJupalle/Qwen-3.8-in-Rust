//! Layers (Phase 2a): gated DeltaNet, GQA attention, MLP, and the decoder-layer composition, built on the
//! scalar kernels. Weights come from a `TensorSource` (a GGUF, or f32 fixture arrays in tests) in GGUF
//! layout: `blk.N.<suffix>` names, V heads re-tiled, `ssm_a = -exp(A_log)`, norms with the +1 folded in.
//! No full-model forward lives here; the tiny-oracle test composes layers itself.

pub mod gated_deltanet;
pub mod gqa_attention;
pub mod mlp;
pub mod mtp;

use crate::config::ModelConfig;
use crate::gguf::{GgmlType, Gguf};
use crate::kernels::matvec::{matvec, Act, ActBuf, WeightMat};
use crate::kernels::rmsnorm::rmsnorm;
use crate::quant::dequantize;

#[derive(Debug, thiserror::Error)]
pub enum LayerError {
    #[error("tensor {0} not available")]
    Missing(String),
    #[error("tensor {name}: expected shape {expected:?}, found {found:?}")]
    Shape { name: String, expected: Vec<usize>, found: Vec<usize> },
    #[error(transparent)]
    Gguf(#[from] crate::gguf::GgufError),
    #[error(transparent)]
    Quant(#[from] crate::quant::QuantError),
}

pub type Result<T> = std::result::Result<T, LayerError>;

/// Where layer weights come from. Shapes are numpy order (`[rows, cols]` for matrices).
pub trait TensorSource {
    /// A 2-D weight as a `WeightMat` with `rows` output features and `cols` input features.
    fn mat(&self, name: &str, rows: usize, cols: usize) -> Result<WeightMat>;
    /// A 1-D (or flattened) f32 tensor of `len` elements.
    fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>>;
}

impl TensorSource for Gguf {
    fn mat(&self, name: &str, rows: usize, cols: usize) -> Result<WeightMat> {
        let t = self.tensor(name).map_err(|_| LayerError::Missing(name.to_string()))?;
        let found: Vec<usize> = t.shape_numpy_order().iter().map(|&d| d as usize).collect();
        if found != vec![rows, cols] {
            return Err(LayerError::Shape { name: name.into(), expected: vec![rows, cols], found });
        }
        let bytes = self.read_raw(name)?;
        Ok(WeightMat::new(t.ggml_type, rows, cols, bytes))
    }

    fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>> {
        let t = self.tensor(name).map_err(|_| LayerError::Missing(name.to_string()))?;
        if t.element_count as usize != len {
            return Err(LayerError::Shape { name: name.into(), expected: vec![len], found: t.shape_numpy_order().iter().map(|&d| d as usize).collect() });
        }
        let bytes = self.read_raw(name)?;
        if t.ggml_type == GgmlType::F32 {
            Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        } else {
            Ok(dequantize(t.ggml_type, &bytes, len)?)
        }
    }
}

/// A projection `y = W x` (`W` is `out x in` in numpy order).
pub struct Linear {
    pub w: WeightMat,
}

impl Linear {
    pub fn load(src: &dyn TensorSource, name: &str, out: usize, inp: usize) -> Result<Linear> {
        Ok(Linear { w: src.mat(name, out, inp)? })
    }
    pub fn out_features(&self) -> usize {
        self.w.rows
    }
    pub fn in_features(&self) -> usize {
        self.w.cols
    }
    /// One vector in, one vector out: `x` is quantised to the form the weight type consumes (Q8_K for
    /// K-quants, Q8_0 for Q8_0 / Q4_0, f32 for F32) once, here.
    pub fn forward(&self, x: &[f32], y: &mut [f32], threads: usize) {
        assert_eq!(x.len(), self.w.cols, "Linear: input length");
        let mut act = ActBuf::with_capacity(x.len());
        act.fill(x, &[self.w.ggml_type]);
        self.forward_act(act.act_for(self.w.ggml_type, x), y, threads);
    }
    /// As `forward` on an activation whose quantised forms are shared with other projections.
    pub fn forward_act(&self, x: Act<'_>, y: &mut [f32], threads: usize) {
        matvec(&self.w, x, y, threads);
    }
}

/// Decoder-layer mixer: DeltaNet for `linear_attention` blocks, GQA attention for `full_attention` blocks.
pub enum Mixer {
    DeltaNet(gated_deltanet::GatedDeltaNet),
    Attention(gqa_attention::GqaAttention),
}

/// Per-layer carried state.
#[derive(Clone)]
pub enum MixerState {
    DeltaNet(gated_deltanet::DeltaState),
    Attention(gqa_attention::KvCache),
}

/// Every per-token buffer a decoder layer needs, preallocated once and reused by all layers in turn
/// (Phase 3.6, finding 52). Sized for the largest layer of the model, so one instance serves the whole
/// stack; held by `model::State`, which is where "preallocated at load" lives.
pub struct Scratch {
    /// The RMSNorm output feeding the mixer, then the MLP.
    pub normed: Vec<f32>,
    /// The mixer output, then the MLP output.
    pub mixed: Vec<f32>,
    /// `x + mixer(...)`, the layer's mid-residual.
    pub resid: Vec<f32>,
    /// The quantised form of `normed`, shared by every projection that reads it.
    pub act: ActBuf,
    pub mlp: mlp::MlpScratch,
    pub dn: Option<gated_deltanet::DeltaScratch>,
    pub at: Option<gqa_attention::AttnScratch>,
}

impl Scratch {
    /// Sized for the largest layer in `layers` (the DeltaNet and attention parts are allocated only if the
    /// stack has such a layer).
    pub fn for_layers(layers: &[DecoderLayer], hidden: usize) -> Scratch {
        let mut inter = 0usize;
        let mut dn: Option<gated_deltanet::DeltaScratch> = None;
        let mut at: Option<gqa_attention::AttnScratch> = None;
        let (mut dn_dims, mut at_dims) = ((0usize, 0usize, 0usize, 0usize), (0usize, 0usize, 0usize, 0usize));
        for l in layers {
            inter = inter.max(l.mlp.gate.out_features());
            match &l.mixer {
                Mixer::DeltaNet(d) => {
                    let c = &mut dn_dims;
                    *c = (c.0.max(d.conv_dim()), c.1.max(d.n_v), c.2.max(d.dk), c.3.max(d.dv));
                }
                Mixer::Attention(a) => {
                    let c = &mut at_dims;
                    *c = (c.0.max(a.n_head), c.1.max(a.n_head_kv), c.2.max(a.head_dim), c.3.max(a.rope_dim));
                }
            }
        }
        if dn_dims.1 > 0 {
            dn = Some(gated_deltanet::DeltaScratch::new(dn_dims.0, dn_dims.1, dn_dims.2, dn_dims.3));
        }
        if at_dims.0 > 0 {
            at = Some(gqa_attention::AttnScratch::new(at_dims.0, at_dims.1, at_dims.2, at_dims.3));
        }
        Scratch {
            normed: vec![0f32; hidden],
            mixed: vec![0f32; hidden],
            resid: vec![0f32; hidden],
            act: ActBuf::with_capacity(hidden),
            mlp: mlp::MlpScratch::new(inter),
            dn,
            at,
        }
    }

    /// Size the attention score buffers for sequences up to `max_pos` positions.
    pub fn reserve(&mut self, max_pos: usize) {
        if let Some(at) = &mut self.at {
            at.reserve(max_pos);
        }
    }

    /// Bytes held (capacities), for the memory plan.
    pub fn bytes(&self) -> usize {
        (self.normed.capacity() + self.mixed.capacity() + self.resid.capacity()) * 4
            + self.act.bytes()
            + self.mlp.bytes()
            + self.dn.as_ref().map_or(0, |d| d.bytes())
            + self.at.as_ref().map_or(0, |a| a.bytes())
    }
}

/// Every buffer the batched forward of `max_t` rows needs (Phase 5: the verification batch of `k + 1` tokens
/// and the MTP re-feed), preallocated once and reused by every layer in turn, like `Scratch` for one token.
/// The per-head attention score rows are taken from the `Scratch` passed alongside.
pub struct BatchScratch {
    pub max_t: usize,
    pub normed: Vec<f32>,
    pub mixed: Vec<f32>,
    pub resid: Vec<f32>,
    pub acts: Vec<ActBuf>,
    pub mlp: mlp::MlpBatch,
    pub dn: Option<gated_deltanet::DeltaBatch>,
    pub at: Option<gqa_attention::AttnBatch>,
}

impl BatchScratch {
    /// Sized for the largest layer in `layers` and `max_t` rows.
    pub fn for_layers(layers: &[DecoderLayer], hidden: usize, max_t: usize) -> BatchScratch {
        let mut inter = 0usize;
        let (mut dn_dims, mut at_dims) = ((0usize, 0usize, 0usize, 0usize), (0usize, 0usize, 0usize, 0usize));
        for l in layers {
            inter = inter.max(l.mlp.gate.out_features());
            match &l.mixer {
                Mixer::DeltaNet(d) => {
                    let c = &mut dn_dims;
                    *c = (c.0.max(d.conv_dim()), c.1.max(d.n_v), c.2.max(d.dk), c.3.max(d.dv));
                }
                Mixer::Attention(a) => {
                    let c = &mut at_dims;
                    *c = (c.0.max(a.n_head), c.1.max(a.n_head_kv), c.2.max(a.head_dim), c.3.max(a.rope_dim));
                }
            }
        }
        BatchScratch {
            max_t,
            normed: vec![0f32; max_t * hidden],
            mixed: vec![0f32; max_t * hidden],
            resid: vec![0f32; max_t * hidden],
            acts: (0..max_t).map(|_| ActBuf::with_capacity(hidden)).collect(),
            mlp: mlp::MlpBatch::new(inter, max_t),
            dn: (dn_dims.1 > 0).then(|| gated_deltanet::DeltaBatch::new(dn_dims.0, dn_dims.1, dn_dims.2, dn_dims.3, max_t)),
            at: (at_dims.0 > 0).then(|| gqa_attention::AttnBatch::new(at_dims.0, at_dims.1, at_dims.2, at_dims.3, max_t)),
        }
    }

    /// Bytes held (capacities), for the memory plan.
    pub fn bytes(&self) -> usize {
        (self.normed.capacity() + self.mixed.capacity() + self.resid.capacity()) * 4
            + self.acts.iter().map(|a| a.bytes()).sum::<usize>()
            + self.mlp.bytes()
            + self.dn.as_ref().map_or(0, |d| d.bytes())
            + self.at.as_ref().map_or(0, |a| a.bytes())
    }
}

/// `h = x + mixer(attn_norm(x)); y = h + mlp(post_attention_norm(h))` (Qwen3_5DecoderLayer lines 757-797).
pub struct DecoderLayer {
    pub index: u32,
    pub attn_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub mixer: Mixer,
    pub mlp: mlp::Mlp,
    pub eps: f32,
}

impl DecoderLayer {
    /// Load layer `index` (`blk.{index}.*`) with the sizes and layer schedule from the GGUF config.
    pub fn load(src: &dyn TensorSource, cfg: &ModelConfig, index: u32) -> Result<DecoderLayer> {
        Self::load_kind(src, cfg, index, cfg.full_attention_layers.contains(&index))
    }

    /// Load `blk.{index}.*` as an attention block or a DeltaNet block regardless of the layer schedule (the MTP
    /// block, `blk.64`, is an attention block that is not in `full_attention_layers`).
    pub fn load_kind(src: &dyn TensorSource, cfg: &ModelConfig, index: u32, attention: bool) -> Result<DecoderLayer> {
        let p = format!("blk.{index}.");
        let hidden = cfg.hidden_size as usize;
        let mixer = if attention {
            Mixer::Attention(gqa_attention::GqaAttention::load(
                src, &p, hidden, cfg.n_head as usize, cfg.n_head_kv as usize, cfg.head_dim_k as usize, cfg.rope_dim as usize, cfg.rope_freq_base, cfg.rms_norm_eps,
            )?)
        } else {
            Mixer::DeltaNet(gated_deltanet::GatedDeltaNet::load(
                src, &p, hidden, cfg.dn_n_k_heads as usize, cfg.dn_n_v_heads as usize, cfg.dn_head_dim_k as usize, cfg.dn_head_dim_v as usize, cfg.dn_conv_kernel as usize, cfg.rms_norm_eps,
            )?)
        };
        Ok(DecoderLayer {
            index,
            attn_norm: src.vec(&format!("{p}attn_norm.weight"), hidden)?,
            post_attention_norm: src.vec(&format!("{p}post_attention_norm.weight"), hidden)?,
            mixer,
            mlp: mlp::Mlp::load(src, &p, hidden, cfg.intermediate_size as usize)?,
            eps: cfg.rms_norm_eps,
        })
    }

    pub fn new_state(&self) -> MixerState {
        match &self.mixer {
            Mixer::DeltaNet(d) => MixerState::DeltaNet(d.new_state()),
            Mixer::Attention(a) => MixerState::Attention(a.new_cache()),
        }
    }

    /// Scratch sized for this layer alone (what `forward_token` allocates per call).
    pub fn scratch(&self) -> Scratch {
        Scratch::for_layers(std::slice::from_ref(self), self.attn_norm.len())
    }

    /// One token at position `pos`. `x` and `y` are `hidden`; `y` may alias nothing.
    pub fn forward_token(&self, x: &[f32], pos: u32, state: &mut MixerState, y: &mut [f32], threads: usize) {
        let mut sc = self.scratch();
        if let MixerState::Attention(c) = &*state {
            sc.reserve(c.len + 1);
        }
        self.forward_token_in(x, pos, state, &mut sc, y, threads);
    }

    /// `forward_token` with caller-owned buffers: no allocation once `sc` is sized (Phase 3.6). Bit for bit
    /// what `forward_token` computes.
    pub fn forward_token_in(&self, x: &[f32], pos: u32, state: &mut MixerState, sc: &mut Scratch, y: &mut [f32], threads: usize) {
        let hidden = x.len();
        {
            crate::prof_scope!(crate::prof::Stage::Norm);
            rmsnorm(x, &self.attn_norm, self.eps, &mut sc.normed[..hidden]);
        }
        match (&self.mixer, state) {
            (Mixer::DeltaNet(d), MixerState::DeltaNet(s)) => {
                let dn = sc.dn.as_mut().expect("Scratch: no DeltaNet buffers");
                sc.act.fill(&sc.normed[..hidden], &d.input_types());
                d.forward_token_in(&sc.normed[..hidden], &sc.act, s, dn, &mut sc.mixed[..hidden], threads);
            }
            (Mixer::Attention(a), MixerState::Attention(c)) => {
                let at = sc.at.as_mut().expect("Scratch: no attention buffers");
                sc.act.fill(&sc.normed[..hidden], &a.input_types());
                a.forward_token_in(&sc.normed[..hidden], &sc.act, pos, c, at, &mut sc.mixed[..hidden], threads);
            }
            _ => panic!("DecoderLayer: state kind does not match mixer kind"),
        }
        {
            crate::prof_scope!(crate::prof::Stage::Residual);
            for ((r, xi), mi) in sc.resid[..hidden].iter_mut().zip(x).zip(&sc.mixed[..hidden]) {
                *r = *xi + *mi;
            }
        }
        {
            crate::prof_scope!(crate::prof::Stage::Norm);
            rmsnorm(&sc.resid[..hidden], &self.post_attention_norm, self.eps, &mut sc.normed[..hidden]);
        }
        sc.act.fill(&sc.normed[..hidden], &self.mlp.input_types());
        self.mlp.forward_in(&sc.normed[..hidden], &sc.act, &mut sc.mlp, &mut sc.mixed[..hidden], threads);
        crate::prof_scope!(crate::prof::Stage::Residual);
        for ((yi, r), mi) in y[..hidden].iter_mut().zip(&sc.resid[..hidden]).zip(&sc.mixed[..hidden]) {
            *yi = *r + *mi;
        }
    }

    /// `forward_prefill` with caller-owned buffers (Phase 5): no allocation once `bsc` is sized for `t` rows and
    /// `sc` / the state are reserved for the positions. Row `i` is `forward_token_in` on row `i`, bit for bit.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch_in(&self, x: &[f32], t: usize, start_pos: u32, state: &mut MixerState, bsc: &mut BatchScratch, sc: &mut Scratch, y: &mut [f32], threads: usize) {
        let hidden = self.attn_norm.len();
        assert_eq!(x.len(), t * hidden);
        assert_eq!(y.len(), t * hidden);
        assert!(t <= bsc.max_t, "BatchScratch: {t} rows > max_t {}", bsc.max_t);
        for i in 0..t {
            rmsnorm(&x[i * hidden..(i + 1) * hidden], &self.attn_norm, self.eps, &mut bsc.normed[i * hidden..(i + 1) * hidden]);
        }
        match (&self.mixer, state) {
            (Mixer::DeltaNet(d), MixerState::DeltaNet(s)) => {
                let BatchScratch { normed, mixed, acts, dn, .. } = &mut *bsc;
                let dnb = dn.as_mut().expect("BatchScratch: no DeltaNet buffers");
                for i in 0..t {
                    acts[i].fill(&normed[i * hidden..(i + 1) * hidden], &d.input_types());
                }
                d.forward_batch_in(&normed[..t * hidden], t, acts, s, dnb, &mut mixed[..t * hidden], threads);
            }
            (Mixer::Attention(a), MixerState::Attention(c)) => {
                let BatchScratch { normed, mixed, acts, at, .. } = &mut *bsc;
                let atb = at.as_mut().expect("BatchScratch: no attention buffers");
                let ssc = sc.at.as_mut().expect("Scratch: no attention buffers");
                for i in 0..t {
                    acts[i].fill(&normed[i * hidden..(i + 1) * hidden], &a.input_types());
                }
                a.forward_batch_in(&normed[..t * hidden], t, acts, start_pos, c, atb, ssc, &mut mixed[..t * hidden], threads);
            }
            _ => panic!("DecoderLayer: state kind does not match mixer kind"),
        }
        for i in 0..t * hidden {
            bsc.resid[i] = x[i] + bsc.mixed[i];
        }
        for i in 0..t {
            rmsnorm(&bsc.resid[i * hidden..(i + 1) * hidden], &self.post_attention_norm, self.eps, &mut bsc.normed[i * hidden..(i + 1) * hidden]);
        }
        let BatchScratch { normed, mixed, acts, mlp, .. } = &mut *bsc;
        for i in 0..t {
            acts[i].fill(&normed[i * hidden..(i + 1) * hidden], &self.mlp.input_types());
        }
        self.mlp.forward_batch_in(&normed[..t * hidden], t, acts, mlp, &mut mixed[..t * hidden], threads);
        for i in 0..t * hidden {
            y[i] = bsc.resid[i] + bsc.mixed[i];
        }
    }

    /// `t` tokens at positions `start_pos..start_pos + t` (`x`, `y` are `t * hidden`): norms per row, the mixer's
    /// batched prefill, the MLP's batched forward. Row `i` is `forward_token` on row `i` (bit for bit).
    pub fn forward_prefill(&self, x: &[f32], t: usize, start_pos: u32, state: &mut MixerState, y: &mut [f32], threads: usize) {
        let hidden = self.attn_norm.len();
        assert_eq!(x.len(), t * hidden);
        assert_eq!(y.len(), t * hidden);
        let mut normed = vec![0f32; t * hidden];
        for i in 0..t {
            rmsnorm(&x[i * hidden..(i + 1) * hidden], &self.attn_norm, self.eps, &mut normed[i * hidden..(i + 1) * hidden]);
        }
        let mut mixed = vec![0f32; t * hidden];
        match (&self.mixer, state) {
            (Mixer::DeltaNet(d), MixerState::DeltaNet(s)) => d.forward_prefill(&normed, t, s, &mut mixed, threads),
            (Mixer::Attention(a), MixerState::Attention(c)) => a.forward_prefill(&normed, t, start_pos, c, &mut mixed, threads),
            _ => panic!("DecoderLayer: state kind does not match mixer kind"),
        }
        let mut h = vec![0f32; t * hidden];
        for i in 0..t * hidden {
            h[i] = x[i] + mixed[i];
        }
        for i in 0..t {
            rmsnorm(&h[i * hidden..(i + 1) * hidden], &self.post_attention_norm, self.eps, &mut normed[i * hidden..(i + 1) * hidden]);
        }
        self.mlp.forward_batch(&normed, t, &mut mixed, threads);
        for i in 0..t * hidden {
            y[i] = h[i] + mixed[i];
        }
    }
}
