//! 1.2: every ModelConfig field (built from GGUF metadata only) equals the corresponding value in
//! tests/fixtures/config.json, by JSON path, following the table in docs/config-mapping.md / MAPPING.

mod common;

use aqueduct_core::{Gguf, ModelConfig, MAPPING};
use common::jpath;

fn u(v: &serde_json::Value) -> u64 {
    v.as_u64().unwrap_or_else(|| panic!("not an integer: {v}"))
}

#[test]
fn config_from_gguf_matches_config_json_by_path() {
    let g = Gguf::open(common::gguf_path()).expect("open");
    let c = ModelConfig::from_gguf(&g).expect("config from gguf");
    let j = common::json(&common::fixture("config.json"));
    let t = |p: &str| jpath(&j, &format!("text_config.{p}"));

    // identity: qwen35 (GGUF) vs qwen3_5 (HF model_type)
    assert_eq!(c.architecture, jpath(&j, "model_type").as_str().unwrap().replace('_', ""));
    // sizes
    assert_eq!(c.n_layer as u64, u(t("num_hidden_layers")));
    assert_eq!(c.nextn_predict_layers as u64, u(t("mtp_num_hidden_layers")));
    assert_eq!(c.block_count as u64, u(t("num_hidden_layers")) + u(t("mtp_num_hidden_layers")));
    assert_eq!(c.context_length as u64, u(t("max_position_embeddings")));
    assert_eq!(c.hidden_size as u64, u(t("hidden_size")));
    assert_eq!(c.intermediate_size as u64, u(t("intermediate_size")));
    assert_eq!(c.vocab_size as u64, u(t("vocab_size")));
    // full attention
    assert_eq!(c.n_head as u64, u(t("num_attention_heads")));
    assert_eq!(c.n_head_kv as u64, u(t("num_key_value_heads")));
    assert_eq!(c.head_dim_k as u64, u(t("head_dim")));
    assert_eq!(c.head_dim_v as u64, u(t("head_dim")));
    assert_eq!(c.rms_norm_eps, t("rms_norm_eps").as_f64().unwrap() as f32);
    assert_eq!(c.full_attention_interval as u64, u(t("full_attention_interval")));
    assert_eq!(c.rope_freq_base as f64, t("rope_parameters.rope_theta").as_f64().unwrap());
    assert_eq!(c.rope_dim as f64, u(t("head_dim")) as f64 * t("partial_rotary_factor").as_f64().unwrap());
    let sections: Vec<i64> = t("rope_parameters.mrope_section").as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
    assert_eq!(c.rope_sections.iter().take(3).map(|&x| x as i64).collect::<Vec<_>>(), sections);
    assert_eq!(c.rope_sections.len(), 4, "GGUF carries a 4th (zero) section entry");
    assert_eq!(c.rope_sections[3], 0);
    // DeltaNet (GGUF ssm.* names)
    assert_eq!(c.dn_conv_kernel as u64, u(t("linear_conv_kernel_dim")));
    assert_eq!(c.dn_head_dim_k as u64, u(t("linear_key_head_dim")));
    assert_eq!(c.dn_head_dim_v as u64, u(t("linear_value_head_dim")));
    assert_eq!(c.dn_n_k_heads as u64, u(t("linear_num_key_heads")));
    assert_eq!(c.dn_n_v_heads as u64, u(t("linear_num_value_heads")));
    assert_eq!(c.dn_inner_size as u64, u(t("linear_num_value_heads")) * u(t("linear_value_head_dim")));
    // tokenizer ids
    assert_eq!(c.bos_id as u64, u(t("bos_token_id")));
    let gen = common::json(&common::fixture("generation_config.json"));
    let gen_eos: Vec<u64> = jpath(&gen, "eos_token_id").as_array().unwrap().iter().map(u).collect();
    assert_eq!(c.eos_ids[0] as u64, gen_eos[0]);
    for e in &c.eos_ids {
        assert!(gen_eos.contains(&(*e as u64)), "GGUF eos {e} not in generation_config");
    }
    assert_ne!(c.eos_ids[0] as u64, u(t("eos_token_id")), "finding: config.json eos_token_id is the BOS/pad id, not <|im_end|>");
    assert!(t("pad_token_id").is_null());
    let tc = common::json(&common::fixture("tokenizer_config.json"));
    assert_eq!(c.add_bos, jpath(&tc, "add_bos_token").as_bool().unwrap());
    // derived layer schedule vs the explicit list in config.json
    let types: Vec<&str> = t("layer_types").as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    let full: Vec<u32> = types.iter().enumerate().filter(|(_, s)| **s == "full_attention").map(|(i, _)| i as u32).collect();
    let lin: Vec<u32> = types.iter().enumerate().filter(|(_, s)| **s == "linear_attention").map(|(i, _)| i as u32).collect();
    assert_eq!(c.full_attention_layers, full);
    assert_eq!(c.deltanet_layers, lin);
    assert_eq!(c.mtp_layers, vec![64]);
    // derived shape facts
    assert_eq!(c.attn_output_gate, t("attn_output_gate").as_bool().unwrap());
    assert_eq!(c.tie_word_embeddings, t("tie_word_embeddings").as_bool().unwrap());
    assert_eq!(c.mtp_dedicated_embeddings, t("mtp_use_dedicated_embeddings").as_bool().unwrap());
    assert_eq!(c.gqa_group as u64, u(t("num_attention_heads")) / u(t("num_key_value_heads")));
    assert_eq!(c.q_dim as u64, u(t("num_attention_heads")) * u(t("head_dim")));
    assert_eq!(c.kv_dim as u64, u(t("num_key_value_heads")) * u(t("head_dim")));
    assert_eq!(c.dn_qkv_dim, 2 * c.dn_k_dim + c.dn_v_dim);
    assert_eq!(c.dn_state_elems_per_layer, 48 * 128 * 128);
    assert_eq!(c.dn_conv_state_elems_per_layer, 10240 * 3);
    // layout-derived sizes vs the Phase 0 index fixture
    let fx = common::json(&common::fixture("gguf_index.json"));
    let mut spans = std::collections::BTreeMap::<u32, (u64, u64)>::new();
    let mut non_layer = 0u64;
    for tj in fx["tensors"].as_array().unwrap() {
        let name = tj["name"].as_str().unwrap();
        let (off, end) = (u(&tj["byte_offset"]), u(&tj["byte_end"]));
        match aqueduct_core::layer_of(name) {
            Some(l) => {
                let e = spans.entry(l).or_insert((u64::MAX, 0));
                e.0 = e.0.min(off);
                e.1 = e.1.max(end);
            }
            None => non_layer += u(&tj["byte_size"]),
        }
    }
    let mut largest = (0u32, 0u64);
    for (l, (s, e)) in spans.iter().filter(|(l, _)| **l < c.n_layer) {
        if e - s > largest.1 {
            largest = (*l, e - s);
        }
    }
    assert_eq!(c.largest_streamed_layer, largest.0);
    assert_eq!(c.ring_slot_bytes, largest.1);
    assert_eq!(c.pinned_bytes, non_layer + (spans[&64].1 - spans[&64].0));
    assert_eq!(c.layer_bytes.len(), 65);
    println!("{}", c.describe());
}

#[test]
fn mapping_table_is_documented() {
    let doc = std::fs::read_to_string(common::root().join("docs").join("config-mapping.md")).expect("docs/config-mapping.md");
    for m in MAPPING {
        assert!(doc.contains(m.gguf_key), "docs/config-mapping.md lacks GGUF key {}", m.gguf_key);
        assert!(doc.contains(m.field), "docs/config-mapping.md lacks field {}", m.field);
        if m.hf_path != "-" {
            assert!(doc.contains(m.hf_path), "docs/config-mapping.md lacks HF path {}", m.hf_path);
        }
    }
    assert!(MAPPING.len() >= 40);
}

#[test]
fn missing_key_is_a_hard_error_naming_the_key() {
    // A GGUF whose metadata lacks an architecture key: build a tiny in-memory file and open it.
    let dir = std::env::temp_dir().join("aqueduct_phase1_missing_key.gguf");
    let mut b: Vec<u8> = Vec::new();
    b.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // no tensors
    b.extend_from_slice(&1u64.to_le_bytes()); // one kv
    let key = b"general.architecture";
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key);
    b.extend_from_slice(&8u32.to_le_bytes()); // string
    let val = b"qwen35";
    b.extend_from_slice(&(val.len() as u64).to_le_bytes());
    b.extend_from_slice(val);
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    std::fs::write(&dir, &b).unwrap();
    let g = Gguf::open(&dir).expect("open minimal gguf");
    let err = ModelConfig::from_gguf(&g).unwrap_err().to_string();
    assert!(err.contains("qwen35.block_count"), "error must name the missing key: {err}");
    let _ = std::fs::remove_file(&dir);
}
