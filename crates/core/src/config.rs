//! `ModelConfig`: the model contract, built ONLY from GGUF KV metadata and the tensor index.
//!
//! Every field documents the GGUF key it comes from (`{arch}` is the value of `general.architecture`,
//! `qwen35` for this model) or the derivation. A missing or mistyped key is a hard error naming the key;
//! nothing falls back to a default. `config.json` is never read here; it is only a test fixture that
//! `tests/config_vs_fixture.rs` cross-checks against, via the table in `MAPPING` / `docs/config-mapping.md`.

use crate::gguf::{Gguf, GgufError};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("metadata is inconsistent: {what}: {detail}")]
    Inconsistent { what: String, detail: String },
    #[error("tensor {0} is required to derive the config but is not in the file")]
    MissingTensor(String),
}

pub type Result<T> = std::result::Result<T, ConfigError>;

/// One row of the GGUF -> HF config.json -> field mapping. Kept in code so the test that checks the
/// config against `tests/fixtures/config.json` and `docs/config-mapping.md` use the same table.
#[derive(Debug, Clone, Copy)]
pub struct Mapping {
    /// GGUF key (`{arch}` = `general.architecture`), or `derived` for computed fields.
    pub gguf_key: &'static str,
    /// JSON path inside `config.json`, or `-` when HF has no equivalent.
    pub hf_path: &'static str,
    /// `ModelConfig` field name.
    pub field: &'static str,
    pub note: &'static str,
}

pub const MAPPING: &[Mapping] = &[
    Mapping { gguf_key: "general.architecture", hf_path: "model_type", field: "architecture", note: "qwen35 vs qwen3_5: same architecture, different spelling; compared after removing '_'" },
    Mapping { gguf_key: "general.name", hf_path: "-", field: "name", note: "display only" },
    Mapping { gguf_key: "general.file_type", hf_path: "-", field: "file_type", note: "15 = Q4_K_M recipe label, not the tensor types" },
    Mapping { gguf_key: "general.quantization_version", hf_path: "-", field: "quantization_version", note: "" },
    Mapping { gguf_key: "{arch}.block_count", hf_path: "text_config.num_hidden_layers + text_config.mtp_num_hidden_layers", field: "block_count", note: "65 = 64 + 1 MTP block" },
    Mapping { gguf_key: "{arch}.nextn_predict_layers", hf_path: "text_config.mtp_num_hidden_layers", field: "nextn_predict_layers", note: "" },
    Mapping { gguf_key: "derived: block_count - nextn_predict_layers", hf_path: "text_config.num_hidden_layers", field: "n_layer", note: "" },
    Mapping { gguf_key: "{arch}.context_length", hf_path: "text_config.max_position_embeddings", field: "context_length", note: "" },
    Mapping { gguf_key: "{arch}.embedding_length", hf_path: "text_config.hidden_size", field: "hidden_size", note: "" },
    Mapping { gguf_key: "{arch}.feed_forward_length", hf_path: "text_config.intermediate_size", field: "intermediate_size", note: "" },
    Mapping { gguf_key: "derived: len(tokenizer.ggml.tokens), checked against token_embd.weight shape", hf_path: "text_config.vocab_size", field: "vocab_size", note: "padded vocab" },
    Mapping { gguf_key: "{arch}.attention.head_count", hf_path: "text_config.num_attention_heads", field: "n_head", note: "" },
    Mapping { gguf_key: "{arch}.attention.head_count_kv", hf_path: "text_config.num_key_value_heads", field: "n_head_kv", note: "" },
    Mapping { gguf_key: "{arch}.attention.key_length", hf_path: "text_config.head_dim", field: "head_dim_k", note: "" },
    Mapping { gguf_key: "{arch}.attention.value_length", hf_path: "text_config.head_dim", field: "head_dim_v", note: "" },
    Mapping { gguf_key: "{arch}.attention.layer_norm_rms_epsilon", hf_path: "text_config.rms_norm_eps", field: "rms_norm_eps", note: "f32 in GGUF, f64 in JSON; compared as f32" },
    Mapping { gguf_key: "{arch}.full_attention_interval", hf_path: "text_config.full_attention_interval", field: "full_attention_interval", note: "" },
    Mapping { gguf_key: "{arch}.rope.freq_base", hf_path: "text_config.rope_parameters.rope_theta", field: "rope_freq_base", note: "" },
    Mapping { gguf_key: "{arch}.rope.dimension_count", hf_path: "text_config.head_dim * text_config.partial_rotary_factor", field: "rope_dim", note: "64 = 256 * 0.25" },
    Mapping { gguf_key: "{arch}.rope.dimension_sections", hf_path: "text_config.rope_parameters.mrope_section", field: "rope_sections", note: "GGUF has a 4th entry (0); first 3 compared" },
    Mapping { gguf_key: "{arch}.ssm.conv_kernel", hf_path: "text_config.linear_conv_kernel_dim", field: "dn_conv_kernel", note: "" },
    Mapping { gguf_key: "{arch}.ssm.state_size", hf_path: "text_config.linear_key_head_dim", field: "dn_head_dim_k", note: "Mamba name for the DeltaNet head dim" },
    Mapping { gguf_key: "{arch}.ssm.group_count", hf_path: "text_config.linear_num_key_heads", field: "dn_n_k_heads", note: "" },
    Mapping { gguf_key: "{arch}.ssm.time_step_rank", hf_path: "text_config.linear_num_value_heads", field: "dn_n_v_heads", note: "" },
    Mapping { gguf_key: "{arch}.ssm.inner_size", hf_path: "text_config.linear_num_value_heads * text_config.linear_value_head_dim", field: "dn_inner_size", note: "6144 = 48 * 128" },
    Mapping { gguf_key: "derived: ssm.inner_size / ssm.time_step_rank", hf_path: "text_config.linear_value_head_dim", field: "dn_head_dim_v", note: "" },
    Mapping { gguf_key: "tokenizer.ggml.bos_token_id", hf_path: "text_config.bos_token_id", field: "bos_id", note: "" },
    Mapping { gguf_key: "tokenizer.ggml.eos_token_id", hf_path: "generation_config.json eos_token_id[0]; text_config.eos_token_id differs (finding)", field: "eos_ids", note: "GGUF lists one EOS; generation_config lists two" },
    Mapping { gguf_key: "tokenizer.ggml.padding_token_id", hf_path: "tokenizer_config.json pad_token (id)", field: "pad_id", note: "text_config.pad_token_id is null" },
    Mapping { gguf_key: "tokenizer.ggml.add_bos_token", hf_path: "tokenizer_config.json add_bos_token", field: "add_bos", note: "" },
    Mapping { gguf_key: "tokenizer.ggml.model", hf_path: "tokenizer.json model.type (BPE) / tokenizer_config tokenizer_class", field: "tokenizer_model", note: "gpt2 = byte-level BPE" },
    Mapping { gguf_key: "tokenizer.ggml.pre", hf_path: "-", field: "tokenizer_pre", note: "pre-tokenizer regex id used by llama.cpp" },
    Mapping { gguf_key: "derived: blk.N.attn_q.weight present <-> (N+1) % full_attention_interval == 0", hf_path: "text_config.layer_types", field: "full_attention_layers / deltanet_layers", note: "0-based; error if tensors and interval disagree" },
    Mapping { gguf_key: "derived: attention.head_count / attention.head_count_kv", hf_path: "text_config.num_attention_heads / text_config.num_key_value_heads", field: "gqa_group", note: "" },
    Mapping { gguf_key: "derived: blk.N.attn_q.weight rows == 2 * head_count * key_length", hf_path: "text_config.attn_output_gate", field: "attn_output_gate", note: "from the tensor shape, not a KV key" },
    Mapping { gguf_key: "derived: output.weight present", hf_path: "text_config.tie_word_embeddings (negated)", field: "tie_word_embeddings", note: "" },
    Mapping { gguf_key: "derived: no dedicated MTP embedding/output tensors", hf_path: "text_config.mtp_use_dedicated_embeddings", field: "mtp_dedicated_embeddings", note: "" },
    Mapping { gguf_key: "derived: n_v_heads * head_dim_k * head_dim_v", hf_path: "-", field: "dn_state_elems_per_layer", note: "DeltaNet recurrent state, f32 elements" },
    Mapping { gguf_key: "derived: (2*k_dim + v_dim) * (conv_kernel - 1)", hf_path: "-", field: "dn_conv_state_elems_per_layer", note: "causal conv carry-over" },
    Mapping { gguf_key: "derived: max layer span over layers 0..n_layer", hf_path: "-", field: "ring_slot_bytes", note: "largest streamed layer" },
    Mapping { gguf_key: "derived: non-layer tensors + MTP blocks", hf_path: "-", field: "pinned_bytes", note: "embed, output, output_norm, blk.64" },
];

/// The model contract. See `MAPPING` for provenance of every field.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    // identity
    pub architecture: String,
    pub name: String,
    pub file_type: u32,
    pub quantization_version: u32,
    // sizes
    pub block_count: u32,
    pub nextn_predict_layers: u32,
    pub n_layer: u32,
    pub context_length: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    // full attention
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub rms_norm_eps: f32,
    pub full_attention_interval: u32,
    pub rope_freq_base: f32,
    pub rope_dim: u32,
    pub rope_sections: Vec<i32>,
    // gated DeltaNet (GGUF uses Mamba-style `ssm.*` names)
    pub dn_conv_kernel: u32,
    pub dn_head_dim_k: u32,
    pub dn_n_k_heads: u32,
    pub dn_n_v_heads: u32,
    pub dn_inner_size: u32,
    pub dn_head_dim_v: u32,
    // tokenizer ids
    pub bos_id: u32,
    pub eos_ids: Vec<u32>,
    pub pad_id: u32,
    pub add_bos: bool,
    pub tokenizer_model: String,
    pub tokenizer_pre: String,
    // derived
    pub full_attention_layers: Vec<u32>,
    pub deltanet_layers: Vec<u32>,
    pub mtp_layers: Vec<u32>,
    pub gqa_group: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub attn_output_gate: bool,
    pub tie_word_embeddings: bool,
    pub mtp_dedicated_embeddings: bool,
    pub dn_k_dim: u32,
    pub dn_v_dim: u32,
    pub dn_qkv_dim: u32,
    pub dn_state_elems_per_layer: u64,
    pub dn_conv_state_elems_per_layer: u64,
    pub layer_bytes: Vec<u64>,
    pub largest_streamed_layer: u32,
    pub ring_slot_bytes: u64,
    pub pinned_bytes: u64,
}

fn inconsistent<T>(what: &str, detail: String) -> Result<T> {
    Err(ConfigError::Inconsistent { what: what.to_string(), detail })
}

impl ModelConfig {
    /// Build the contract from an opened GGUF. Reads no tensor data.
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let architecture = g.get_str("general.architecture")?.to_string();
        let k = |suffix: &str| format!("{architecture}.{suffix}");

        let block_count = g.get_u32(&k("block_count"))?;
        let nextn_predict_layers = g.get_u32(&k("nextn_predict_layers"))?;
        if nextn_predict_layers >= block_count {
            return inconsistent("nextn_predict_layers", format!("{nextn_predict_layers} >= block_count {block_count}"));
        }
        let n_layer = block_count - nextn_predict_layers;
        let context_length = g.get_u32(&k("context_length"))?;
        let hidden_size = g.get_u32(&k("embedding_length"))?;
        let intermediate_size = g.get_u32(&k("feed_forward_length"))?;
        let n_head = g.get_u32(&k("attention.head_count"))?;
        let n_head_kv = g.get_u32(&k("attention.head_count_kv"))?;
        let head_dim_k = g.get_u32(&k("attention.key_length"))?;
        let head_dim_v = g.get_u32(&k("attention.value_length"))?;
        let rms_norm_eps = g.get_f32(&k("attention.layer_norm_rms_epsilon"))?;
        let full_attention_interval = g.get_u32(&k("full_attention_interval"))?;
        let rope_freq_base = g.get_f32(&k("rope.freq_base"))?;
        let rope_dim = g.get_u32(&k("rope.dimension_count"))?;
        let rope_sections = g.get_array_i32(&k("rope.dimension_sections"))?;
        let dn_conv_kernel = g.get_u32(&k("ssm.conv_kernel"))?;
        let dn_head_dim_k = g.get_u32(&k("ssm.state_size"))?;
        let dn_n_k_heads = g.get_u32(&k("ssm.group_count"))?;
        let dn_n_v_heads = g.get_u32(&k("ssm.time_step_rank"))?;
        let dn_inner_size = g.get_u32(&k("ssm.inner_size"))?;
        let name = g.get_str("general.name")?.to_string();
        let file_type = g.get_u32("general.file_type")?;
        let quantization_version = g.get_u32("general.quantization_version")?;
        let bos_id = g.get_u32("tokenizer.ggml.bos_token_id")?;
        let eos_id = g.get_u32("tokenizer.ggml.eos_token_id")?;
        let pad_id = g.get_u32("tokenizer.ggml.padding_token_id")?;
        let add_bos = g.get_bool("tokenizer.ggml.add_bos_token")?;
        let tokenizer_model = g.get_str("tokenizer.ggml.model")?.to_string();
        let tokenizer_pre = g.get_str("tokenizer.ggml.pre")?.to_string();
        let (_, tokens) = g.get_array("tokenizer.ggml.tokens")?;
        let vocab_size = tokens.len() as u32;
        let mut eos_ids = vec![eos_id];
        if g.has("tokenizer.ggml.eot_token_id") {
            let eot = g.get_u32("tokenizer.ggml.eot_token_id")?;
            if !eos_ids.contains(&eot) {
                eos_ids.push(eot);
            }
        }

        // ---- checks against the tensor index (no data read) ----
        if n_head_kv == 0 || n_head % n_head_kv != 0 {
            return inconsistent("head_count / head_count_kv", format!("{n_head} / {n_head_kv}"));
        }
        if dn_n_v_heads == 0 || dn_inner_size % dn_n_v_heads != 0 {
            return inconsistent("ssm.inner_size / ssm.time_step_rank", format!("{dn_inner_size} / {dn_n_v_heads}"));
        }
        let dn_head_dim_v = dn_inner_size / dn_n_v_heads;
        if dn_n_k_heads == 0 || dn_n_v_heads % dn_n_k_heads != 0 {
            return inconsistent("ssm.time_step_rank / ssm.group_count", format!("{dn_n_v_heads} / {dn_n_k_heads}"));
        }
        if rope_dim > head_dim_k {
            return inconsistent("rope.dimension_count", format!("{rope_dim} > key_length {head_dim_k}"));
        }
        let sec_sum: i32 = rope_sections.iter().sum();
        if sec_sum as u32 * 2 != rope_dim {
            return inconsistent("rope.dimension_sections", format!("sum {sec_sum} * 2 != rope.dimension_count {rope_dim}"));
        }

        let embd = g.tensor("token_embd.weight").map_err(|_| ConfigError::MissingTensor("token_embd.weight".into()))?;
        if embd.shape.len() != 2 || embd.shape[0] != hidden_size as u64 || embd.shape[1] != vocab_size as u64 {
            return inconsistent(
                "token_embd.weight shape",
                format!("{:?} vs [embedding_length {hidden_size}, len(tokens) {vocab_size}]", embd.shape),
            );
        }
        let tie_word_embeddings = !g.has_tensor("output.weight");
        if !tie_word_embeddings {
            let out = g.tensor("output.weight")?;
            if out.shape != embd.shape {
                return inconsistent("output.weight shape", format!("{:?} vs token_embd {:?}", out.shape, embd.shape));
            }
        }
        if !g.has_tensor("output_norm.weight") {
            return Err(ConfigError::MissingTensor("output_norm.weight".into()));
        }

        let q_dim = n_head * head_dim_k;
        let kv_dim = n_head_kv * head_dim_k;
        let dn_k_dim = dn_n_k_heads * dn_head_dim_k;
        let dn_v_dim = dn_n_v_heads * dn_head_dim_v;
        let dn_qkv_dim = 2 * dn_k_dim + dn_v_dim;

        let mut full_attention_layers = Vec::new();
        let mut deltanet_layers = Vec::new();
        let mut attn_output_gate: Option<bool> = None;
        for i in 0..n_layer {
            let has_attn = g.has_tensor(&format!("blk.{i}.attn_q.weight"));
            let has_ssm = g.has_tensor(&format!("blk.{i}.ssm_a"));
            let by_rule = (i + 1) % full_attention_interval == 0;
            match (has_attn, has_ssm) {
                (true, false) => {
                    if !by_rule {
                        return inconsistent("layer_types", format!("blk.{i} has attn_q but (i+1) % {full_attention_interval} != 0"));
                    }
                    let q = g.tensor(&format!("blk.{i}.attn_q.weight"))?;
                    if q.shape.len() != 2 || q.shape[0] != hidden_size as u64 {
                        return inconsistent("attn_q shape", format!("blk.{i}: {:?}", q.shape));
                    }
                    let gate = if q.shape[1] == 2 * q_dim as u64 {
                        true
                    } else if q.shape[1] == q_dim as u64 {
                        false
                    } else {
                        return inconsistent("attn_q rows", format!("blk.{i}: {} is neither q_dim {q_dim} nor 2*q_dim", q.shape[1]));
                    };
                    match attn_output_gate {
                        None => attn_output_gate = Some(gate),
                        Some(prev) if prev != gate => return inconsistent("attn_output_gate", format!("differs at blk.{i}")),
                        _ => {}
                    }
                    let kt = g.tensor(&format!("blk.{i}.attn_k.weight"))?;
                    if kt.shape != vec![hidden_size as u64, kv_dim as u64] {
                        return inconsistent("attn_k shape", format!("blk.{i}: {:?} vs [{hidden_size}, {kv_dim}]", kt.shape));
                    }
                    full_attention_layers.push(i);
                }
                (false, true) => {
                    if by_rule {
                        return inconsistent("layer_types", format!("blk.{i} has ssm_a but (i+1) % {full_attention_interval} == 0"));
                    }
                    let qkv = g.tensor(&format!("blk.{i}.attn_qkv.weight"))?;
                    if qkv.shape != vec![hidden_size as u64, dn_qkv_dim as u64] {
                        return inconsistent("attn_qkv shape", format!("blk.{i}: {:?} vs [{hidden_size}, {dn_qkv_dim}]", qkv.shape));
                    }
                    let conv = g.tensor(&format!("blk.{i}.ssm_conv1d.weight"))?;
                    if conv.shape != vec![dn_conv_kernel as u64, dn_qkv_dim as u64] {
                        return inconsistent("ssm_conv1d shape", format!("blk.{i}: {:?} vs [{dn_conv_kernel}, {dn_qkv_dim}]", conv.shape));
                    }
                    let a = g.tensor(&format!("blk.{i}.ssm_a"))?;
                    if a.shape != vec![dn_n_v_heads as u64] {
                        return inconsistent("ssm_a shape", format!("blk.{i}: {:?} vs [{dn_n_v_heads}]", a.shape));
                    }
                    deltanet_layers.push(i);
                }
                (true, true) => return inconsistent("layer_types", format!("blk.{i} has both attn_q and ssm_a")),
                (false, false) => return inconsistent("layer_types", format!("blk.{i} has neither attn_q nor ssm_a")),
            }
        }
        let attn_output_gate = match attn_output_gate {
            Some(v) => v,
            None => return inconsistent("layer_types", "no full-attention layer found".into()),
        };
        let mtp_layers: Vec<u32> = (n_layer..block_count).collect();
        for &m in &mtp_layers {
            if !g.has_tensor(&format!("blk.{m}.nextn.eh_proj.weight")) {
                return Err(ConfigError::MissingTensor(format!("blk.{m}.nextn.eh_proj.weight")));
            }
        }
        let mtp_dedicated_embeddings = mtp_layers.iter().any(|m| {
            g.has_tensor(&format!("blk.{m}.nextn.embed_tokens.weight")) || g.has_tensor(&format!("blk.{m}.nextn.shared_head.head.weight"))
        });

        // ---- layout-derived sizes ----
        let mut layer_bytes = Vec::with_capacity(block_count as usize);
        for i in 0..block_count {
            let span = match g.layer_span(i) {
                Some(s) => s,
                None => return Err(ConfigError::MissingTensor(format!("blk.{i}.*"))),
            };
            layer_bytes.push(span.end - span.start);
        }
        // first streamed layer with the maximum span (same tie-break as tools/gguf_index.py)
        let mut largest_streamed_layer = 0u32;
        let mut ring_slot_bytes = 0u64;
        for (i, &b) in layer_bytes[..n_layer as usize].iter().enumerate() {
            if b > ring_slot_bytes {
                ring_slot_bytes = b;
                largest_streamed_layer = i as u32;
            }
        }
        let non_layer: u64 = g.non_layer_tensors().iter().map(|t| t.byte_size).sum();
        let mtp: u64 = mtp_layers.iter().map(|&m| layer_bytes[m as usize]).sum();
        let pinned_bytes = non_layer + mtp;

        Ok(ModelConfig {
            architecture,
            name,
            file_type,
            quantization_version,
            block_count,
            nextn_predict_layers,
            n_layer,
            context_length,
            hidden_size,
            intermediate_size,
            vocab_size,
            n_head,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            rms_norm_eps,
            full_attention_interval,
            rope_freq_base,
            rope_dim,
            rope_sections,
            dn_conv_kernel,
            dn_head_dim_k,
            dn_n_k_heads,
            dn_n_v_heads,
            dn_inner_size,
            dn_head_dim_v,
            bos_id,
            eos_ids,
            pad_id,
            add_bos,
            tokenizer_model,
            tokenizer_pre,
            full_attention_layers,
            deltanet_layers,
            mtp_layers,
            gqa_group: n_head / n_head_kv,
            q_dim,
            kv_dim,
            attn_output_gate,
            tie_word_embeddings,
            mtp_dedicated_embeddings,
            dn_k_dim,
            dn_v_dim,
            dn_qkv_dim,
            dn_state_elems_per_layer: dn_n_v_heads as u64 * dn_head_dim_k as u64 * dn_head_dim_v as u64,
            dn_conv_state_elems_per_layer: dn_qkv_dim as u64 * (dn_conv_kernel as u64 - 1),
            layer_bytes,
            largest_streamed_layer,
            ring_slot_bytes,
            pinned_bytes,
        })
    }

    /// Human-readable dump used by `aqueduct info`.
    pub fn describe(&self) -> String {
        let mut s = String::new();
        let mut line = |k: &str, v: String| s.push_str(&format!("  {k:<32} {v}\n"));
        line("architecture", format!("{} ({})", self.architecture, self.name));
        line("file_type / quant version", format!("{} / {}", self.file_type, self.quantization_version));
        line("blocks (n_layer + mtp)", format!("{} ({} + {})", self.block_count, self.n_layer, self.nextn_predict_layers));
        line("context_length", self.context_length.to_string());
        line("hidden / intermediate", format!("{} / {}", self.hidden_size, self.intermediate_size));
        line("vocab_size", self.vocab_size.to_string());
        line("attention heads q/kv, head_dim", format!("{}/{} , {} (v {})", self.n_head, self.n_head_kv, self.head_dim_k, self.head_dim_v));
        line("gqa group, q_dim, kv_dim", format!("{}, {}, {}", self.gqa_group, self.q_dim, self.kv_dim));
        line("attn_output_gate (from attn_q rows)", self.attn_output_gate.to_string());
        line("rms_norm_eps", format!("{:e}", self.rms_norm_eps));
        line("rope freq_base / dims / sections", format!("{} / {} / {:?}", self.rope_freq_base, self.rope_dim, self.rope_sections));
        line("full_attention_interval", self.full_attention_interval.to_string());
        line("full-attention layers", format!("{:?}", self.full_attention_layers));
        line("deltanet layers", format!("{:?}", self.deltanet_layers));
        line("mtp layers", format!("{:?}", self.mtp_layers));
        line("deltanet k heads x dim, v heads x dim", format!("{} x {}, {} x {}", self.dn_n_k_heads, self.dn_head_dim_k, self.dn_n_v_heads, self.dn_head_dim_v));
        line("deltanet qkv_dim, conv kernel", format!("{}, {}", self.dn_qkv_dim, self.dn_conv_kernel));
        line("deltanet state elems/layer", format!("{} (+ conv {})", self.dn_state_elems_per_layer, self.dn_conv_state_elems_per_layer));
        line("tie_word_embeddings", self.tie_word_embeddings.to_string());
        line("mtp dedicated embeddings", self.mtp_dedicated_embeddings.to_string());
        line("tokenizer model / pre", format!("{} / {}", self.tokenizer_model, self.tokenizer_pre));
        line("bos / eos / pad / add_bos", format!("{} / {:?} / {} / {}", self.bos_id, self.eos_ids, self.pad_id, self.add_bos));
        line("largest streamed layer", format!("{} ({} bytes = {:.2} MiB)", self.largest_streamed_layer, self.ring_slot_bytes, self.ring_slot_bytes as f64 / 1048576.0));
        line("pinned set bytes", format!("{} ({:.2} MiB)", self.pinned_bytes, self.pinned_bytes as f64 / 1048576.0));
        s
    }
}
