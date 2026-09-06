//! 1.3: all 45 tok_cases encode to the fixture ids and decode back to the same UTF-8 bytes; the 3 prompts
//! encode to their fixture ids; special ids from GGUF metadata agree with tokenizer_config / generation_config.

mod common;

use aqueduct_core::{Gguf, ModelConfig, SpecialIds, Tok};
use common::jpath;

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn tok_cases_encode_and_decode() {
    let tok = Tok::from_file(common::tokenizer_path()).expect("load tokenizer.json");
    let fx = common::json(&common::fixture("tok_cases.json"));
    assert_eq!(fx["add_special_tokens"].as_bool(), Some(false));
    let cases = fx["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 45);
    let mut enc_ok = 0;
    let mut dec_ok = 0;
    let mut failures = Vec::new();
    for c in cases {
        let label = c["label"].as_str().unwrap();
        let bytes = hex(c["text_utf8_hex"].as_str().unwrap());
        let text = String::from_utf8(bytes.clone()).unwrap();
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let got = tok.encode(&text).unwrap();
        if got == want {
            enc_ok += 1;
        } else {
            failures.push(format!("encode {label}: got {got:?} want {want:?}"));
        }
        let back = tok.decode(&want).unwrap();
        if back.as_bytes() == bytes.as_slice() {
            dec_ok += 1;
        } else {
            failures.push(format!("decode {label}: got {:?} want {:?}", back, text));
        }
    }
    println!("encode {enc_ok}/45, decode {dec_ok}/45");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn prompts_encode() {
    let tok = Tok::from_file(common::tokenizer_path()).expect("load tokenizer.json");
    let fx = common::json(&common::fixture("prompts.json"));
    let prompts = fx["prompts"].as_array().unwrap();
    assert_eq!(prompts.len(), 3);
    for p in prompts {
        let text = p["text"].as_str().unwrap();
        assert_eq!(hex(p["text_utf8_hex"].as_str().unwrap()), text.as_bytes());
        let want: Vec<u32> = p["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        assert_eq!(tok.encode(text).unwrap(), want, "prompt {}", p["name"]);
        assert_eq!(tok.decode(&want).unwrap(), text, "prompt {} decode", p["name"]);
    }
}

#[test]
fn special_ids_from_gguf_cross_check_hf_files() {
    let Some(gguf) = common::gguf_if_present() else { return };
    let g = Gguf::open(gguf).expect("open");
    let cfg = ModelConfig::from_gguf(&g).expect("config");
    let ids = SpecialIds::from_config(&cfg);
    let tok = Tok::from_file(common::tokenizer_path()).expect("load tokenizer.json");
    let tc = common::json(&common::fixture("tokenizer_config.json"));
    let gen = common::json(&common::fixture("generation_config.json"));

    // GGUF ids resolve to the token strings tokenizer_config.json names
    let eos_text = jpath(&tc, "eos_token").as_str().unwrap();
    let pad_text = jpath(&tc, "pad_token").as_str().unwrap();
    assert_eq!(tok.id_to_token(ids.eos[0]).as_deref(), Some(eos_text));
    assert_eq!(tok.id_to_token(ids.pad.unwrap()).as_deref(), Some(pad_text));
    assert!(jpath(&tc, "bos_token").is_null(), "HF has no BOS token");
    assert_eq!(ids.add_bos, jpath(&tc, "add_bos_token").as_bool().unwrap());
    assert!(!ids.add_bos);
    // GGUF names a bos id anyway; it is <|endoftext|>
    assert_eq!(tok.id_to_token(ids.bos.unwrap()).as_deref(), Some("<|endoftext|>"));
    tok.check_special(&ids, &[(ids.eos[0], "eos", eos_text), (ids.pad.unwrap(), "pad", pad_text)]).unwrap();

    // every EOS id: GGUF lists one, generation_config lists two
    let gen_eos: Vec<u32> = jpath(&gen, "eos_token_id").as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    for e in &ids.eos {
        assert!(gen_eos.contains(e));
    }
    let extra: Vec<u32> = gen_eos.iter().copied().filter(|e| !ids.eos.contains(e)).collect();
    println!("GGUF eos ids: {:?}; generation_config eos ids: {:?}; not in GGUF: {:?}", ids.eos, gen_eos, extra);
    assert_eq!(extra, vec![ids.bos.unwrap()], "finding: the second HF stop id is the GGUF bos/pad id <|endoftext|>");
    assert!(ids.is_eos(ids.eos[0]));

    // literal special-token text maps to single ids, as in the HF tokenizer (tok_cases special_lookalike)
    assert_eq!(tok.token_to_id("<|im_start|>"), Some(ids.eos[0] - 1));
    assert_eq!(tok.encode("<|im_start|>").unwrap(), vec![ids.eos[0] - 1]);
    assert_eq!(tok.vocab_size_with_added(), 248_077);
}
