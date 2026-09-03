//! Finding #25 decision: every architecture constant in src/arch.rs mirrors a value in
//! tests/fixtures/config.json or generation_config.json. This test pins each one.

mod common;

use aqueduct_core::arch::{Activation, RopeScaling, StateDtype};
use aqueduct_core::{arch_consts, stop_ids, Gguf, ModelConfig};
use common::jpath;

#[test]
fn qwen35_constants_match_config_fixtures() {
    let a = arch_consts("qwen35").expect("qwen35 is the one known architecture");
    let j = common::json(&common::fixture("config.json"));
    let t = |p: &str| jpath(&j, &format!("text_config.{p}"));

    assert_eq!(t("hidden_act").as_str().unwrap(), "silu");
    assert_eq!(a.hidden_act, Activation::Silu);
    // config.json says "swish"; swish(x) = x * sigmoid(x) = silu(x), and Qwen3_5RMSNormGated uses ACT2FN["silu"]
    assert_eq!(t("output_gate_type").as_str().unwrap(), "swish");
    assert_eq!(a.deltanet_output_gate, Activation::Silu);
    assert_eq!(t("mamba_ssm_dtype").as_str().unwrap(), "float32");
    assert_eq!(a.recurrent_state_dtype, StateDtype::F32);
    assert_eq!(t("rope_parameters.mrope_interleaved").as_bool().unwrap(), a.mrope_interleaved);
    assert_eq!(t("rope_parameters.rope_type").as_str().unwrap(), "default");
    assert_eq!(a.rope_scaling, RopeScaling::None);
    assert!(t("rope_parameters").get("factor").is_none(), "no scaling factor in config.json");

    let gen = common::json(&common::fixture("generation_config.json"));
    let gen_eos: Vec<u32> = jpath(&gen, "eos_token_id").as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    for id in a.extra_stop_ids {
        assert!(gen_eos.contains(id), "extra stop id {id} is not in generation_config.json");
    }
}

#[test]
fn runtime_stop_set_is_gguf_eos_union_arch_extras() {
    let g = Gguf::open(common::gguf_path()).expect("open");
    let c = ModelConfig::from_gguf(&g).expect("config");
    let a = arch_consts(&c.architecture).expect("architecture from the GGUF must be in the table");
    let stops = stop_ids(&c.eos_ids, a);
    let gen = common::json(&common::fixture("generation_config.json"));
    let mut gen_eos: Vec<u32> = jpath(&gen, "eos_token_id").as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let mut got = stops.clone();
    gen_eos.sort();
    got.sort();
    assert_eq!(got, gen_eos, "stop set must equal generation_config.json eos_token_id as a set");
    assert_eq!(stops[0], c.eos_ids[0], "GGUF eos comes first");
}

#[test]
fn other_architectures_are_refused() {
    for name in ["qwen3next", "qwen35moe", "llama", ""] {
        let e = arch_consts(name).unwrap_err().to_string();
        assert!(e.contains(name) || name.is_empty(), "{e}");
        assert!(e.contains("qwen35"), "error lists the known architecture: {e}");
    }
}
