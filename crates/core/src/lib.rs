//! aqueduct-core, Phase 1: read the model. GGUF header/index, config from GGUF metadata,
//! tokenizer wrapper, and scalar dequantisation. No forward pass, no kernels, no tiering.

pub mod arch;
pub mod config;
pub mod gguf;
pub mod kernels;
pub mod layers;
pub mod quant;
pub mod tok;

pub use arch::{arch_consts, stop_ids, ArchConsts};
pub use config::{ConfigError, ModelConfig, MAPPING};
pub use gguf::{layer_of, GgmlType, Gguf, GgufError, LayerSpan, TensorInfo, Value, ValueType};
pub use quant::{dequantize, QuantError};
pub use tok::{SpecialIds, Tok, TokError};
