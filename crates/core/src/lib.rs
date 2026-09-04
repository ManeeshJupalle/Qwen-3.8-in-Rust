//! aqueduct-core: GGUF header/index, config from GGUF metadata, tokenizer wrapper, dequantisation (Phase 1);
//! kernels and layers (Phase 2); the resident model and its per-token forward (Phase 3). No tiering yet.

pub mod arch;
pub mod config;
pub mod gguf;
pub mod kernels;
pub mod layers;
pub mod model;
pub mod quant;
pub mod rss;
pub mod tok;

pub use arch::{arch_consts, stop_ids, ArchConsts};
pub use config::{ConfigError, ModelConfig, MAPPING};
pub use gguf::{layer_of, GgmlType, Gguf, GgufError, LayerSpan, TensorInfo, Value, ValueType};
pub use quant::{dequantize, QuantError};
pub use tok::{SpecialIds, Tok, TokError};
