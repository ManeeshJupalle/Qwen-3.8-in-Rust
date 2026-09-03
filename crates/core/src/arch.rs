//! Architecture-implied constants: values the engine needs that the GGUF metadata does not carry
//! (docs/payload-vs-doc.md finding #25). This is not a defaults table. It is an assertion that we know
//! exactly one architecture; any other `general.architecture` is a hard error naming it.
//!
//! Every constant cites the `config.json` / `generation_config.json` path it mirrors;
//! `tests/arch_vs_fixture.rs` checks each one against those fixtures.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// x * sigmoid(x). `silu` and `swish` are the same function; both names appear in config.json.
    Silu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateDtype {
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScaling {
    /// `rope_type: default`: plain RoPE, attention scaling 1.0, no frequency scaling.
    None,
}

/// Constants for one architecture string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchConsts {
    /// `general.architecture` this table entry is for.
    pub name: &'static str,
    /// MLP activation. Mirrors `config.json: text_config.hidden_act = "silu"`.
    pub hidden_act: Activation,
    /// Activation applied to the DeltaNet gate `z` inside the gated RMSNorm.
    /// Mirrors `config.json: text_config.output_gate_type = "swish"` (Qwen3_5RMSNormGated uses ACT2FN["silu"]).
    pub deltanet_output_gate: Activation,
    /// Precision of the DeltaNet recurrent state and conv window.
    /// Mirrors `config.json: text_config.mamba_ssm_dtype = "float32"`.
    pub recurrent_state_dtype: StateDtype,
    /// Multimodal RoPE stream interleave. Mirrors `config.json: text_config.rope_parameters.mrope_interleaved = true`.
    /// For text-only positions all three streams are identical, so this only matters with images/video.
    pub mrope_interleaved: bool,
    /// Mirrors `config.json: text_config.rope_parameters.rope_type = "default"` (no YaRN/linear scaling keys in the GGUF).
    pub rope_scaling: RopeScaling,
    /// Stop ids the GGUF does not list as EOS. Mirrors `generation_config.json: eos_token_id = [248046, 248044]`;
    /// the GGUF names only 248046 (`<|im_end|>`); 248044 is `<|endoftext|>`, also the GGUF bos/pad id.
    pub extra_stop_ids: &'static [u32],
}

const QWEN35: ArchConsts = ArchConsts {
    name: "qwen35",
    hidden_act: Activation::Silu,
    deltanet_output_gate: Activation::Silu,
    recurrent_state_dtype: StateDtype::F32,
    mrope_interleaved: true,
    rope_scaling: RopeScaling::None,
    extra_stop_ids: &[248044],
};

const TABLE: &[ArchConsts] = &[QWEN35];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownArch(pub String);

impl fmt::Display for UnknownArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown architecture {:?}: this engine knows only {:?}", self.0, TABLE.iter().map(|a| a.name).collect::<Vec<_>>())
    }
}

impl std::error::Error for UnknownArch {}

/// Look up the constants for `general.architecture`.
pub fn arch_consts(architecture: &str) -> Result<&'static ArchConsts, UnknownArch> {
    TABLE.iter().find(|a| a.name == architecture).ok_or_else(|| UnknownArch(architecture.to_string()))
}

/// The runtime stop set: GGUF EOS ids plus the architecture's extra stop ids (deduplicated, in that order).
pub fn stop_ids(gguf_eos: &[u32], arch: &ArchConsts) -> Vec<u32> {
    let mut v: Vec<u32> = gguf_eos.to_vec();
    for &id in arch.extra_stop_ids {
        if !v.contains(&id) {
            v.push(id);
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_architecture_is_named_in_the_error() {
        let e = arch_consts("qwen3next").unwrap_err();
        assert!(e.to_string().contains("qwen3next"));
        assert!(arch_consts("qwen35").is_ok());
    }

    #[test]
    fn stop_set_is_union_in_order() {
        let a = arch_consts("qwen35").unwrap();
        assert_eq!(stop_ids(&[248046], a), vec![248046, 248044]);
        assert_eq!(stop_ids(&[248046, 248044], a), vec![248046, 248044]);
    }
}
