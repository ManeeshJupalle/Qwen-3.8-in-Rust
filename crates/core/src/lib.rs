//! aqueduct-core: GGUF header/index, config from GGUF metadata, tokenizer wrapper, dequantisation (Phase 1);
//! kernels and layers (Phase 2); the model and its per-token forward (Phase 3); the memory plan, pinned
//! arenas and the streaming ring (Phase 4, `tier.rs` / `os.rs`).

pub mod arch;
pub mod chat;
pub mod config;
pub mod cpu;
pub mod gguf;
pub mod kernels;
pub mod layers;
pub mod model;
pub mod os;
pub mod prof;
pub mod quant;
pub mod rss;
pub mod sample;
pub mod spec;
pub mod tier;
pub mod tok;

pub use arch::{arch_consts, stop_ids, ArchConsts};
pub use config::{ConfigError, ModelConfig, MAPPING};
pub use cpu::{logical_cores, physical_cores};
pub use gguf::{layer_of, GgmlType, Gguf, GgufError, LayerSpan, TensorInfo, Value, ValueType};
pub use quant::{dequantize, dequantize_into, QuantError};
pub use tok::{SpecialIds, Tok, TokError};
