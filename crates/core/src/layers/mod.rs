//! Layers (Phase 2a): gated DeltaNet, GQA attention, MLP, and the decoder-layer composition, built on the
//! scalar kernels. Weights come from a `TensorSource` (a GGUF, or f32 fixture arrays in tests) in GGUF
//! layout: `blk.N.<suffix>` names, V heads re-tiled, `ssm_a = -exp(A_log)`, norms with the +1 folded in.
//! No full-model forward lives here; the tiny-oracle test composes layers itself.

pub mod gated_deltanet;
pub mod gqa_attention;
pub mod mlp;

use crate::gguf::{GgmlType, Gguf};
use crate::kernels::matvec::{matvec_f32_in, WeightMat};
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
    /// One vector in, one vector out. Quantised weights quantise `x` to Q8_0 first; F32 weights use f32.
    pub fn forward(&self, x: &[f32], y: &mut [f32], threads: usize) {
        assert_eq!(x.len(), self.w.cols, "Linear: input length");
        matvec_f32_in(&self.w, x, y, threads);
    }
}

/// Decoder-layer mixer: DeltaNet for `linear_attention` blocks, GQA attention for `full_attention` blocks.
pub enum Mixer {
    DeltaNet(gated_deltanet::GatedDeltaNet),
    Attention(gqa_attention::GqaAttention),
}

/// Per-layer carried state.
pub enum MixerState {
    DeltaNet(gated_deltanet::DeltaState),
    Attention(gqa_attention::KvCache),
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
    pub fn new_state(&self) -> MixerState {
        match &self.mixer {
            Mixer::DeltaNet(d) => MixerState::DeltaNet(d.new_state()),
            Mixer::Attention(a) => MixerState::Attention(a.new_cache()),
        }
    }

    /// One token at position `pos`. `x` and `y` are `hidden`; `y` may alias nothing.
    pub fn forward_token(&self, x: &[f32], pos: u32, state: &mut MixerState, y: &mut [f32], threads: usize) {
        let hidden = x.len();
        let mut normed = vec![0f32; hidden];
        let mut mixed = vec![0f32; hidden];
        rmsnorm(x, &self.attn_norm, self.eps, &mut normed);
        match (&self.mixer, state) {
            (Mixer::DeltaNet(d), MixerState::DeltaNet(s)) => d.forward_token(&normed, s, &mut mixed, threads),
            (Mixer::Attention(a), MixerState::Attention(c)) => a.forward_token(&normed, pos, c, &mut mixed, threads),
            _ => panic!("DecoderLayer: state kind does not match mixer kind"),
        }
        let mut h = vec![0f32; hidden];
        for i in 0..hidden {
            h[i] = x[i] + mixed[i];
        }
        rmsnorm(&h, &self.post_attention_norm, self.eps, &mut normed);
        self.mlp.forward(&normed, &mut mixed, threads);
        for i in 0..hidden {
            y[i] = h[i] + mixed[i];
        }
    }
}
