//! Phase 5.3: the chat template rendered by `chat.rs` is byte-identical to HF `apply_chat_template` on the two
//! examples of `docs/chat-template.md` and five multi-turn cases (`tests/fixtures/chat_cases.json`,
//! `tools/chat_cases.py`), and tokenises to the same ids.

mod common;

use aqueduct_core::chat::{ChatTemplate, Message, RenderOpts};
use aqueduct_core::Tok;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn renders_byte_identical_to_hf_on_seven_cases() {
    let tpl_path = common::root().join("models").join("Qwen3.8-27B").join("chat_template.jinja");
    assert!(tpl_path.exists(), "missing {}", tpl_path.display());
    let tpl = ChatTemplate::from_file(&tpl_path).expect("template");
    let tok = Tok::from_file(common::tokenizer_path()).expect("tokenizer");
    let fx = common::json(&common::fixtures().join("chat_cases.json"));
    let mut ok = 0;
    let cases = fx["cases"].as_array().unwrap();
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let msgs: Vec<Message> = c["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Message { role: m["role"].as_str().unwrap().into(), content: m["content"].as_str().unwrap().into(), reasoning_content: m.get("reasoning_content").and_then(|v| v.as_str()).map(String::from) })
            .collect();
        let kw = &c["kwargs"];
        let opts = RenderOpts {
            enable_thinking: kw.get("enable_thinking").and_then(|v| v.as_bool()),
            reasoning_effort: kw.get("reasoning_effort").and_then(|v| v.as_str()).map(String::from),
            preserve_thinking: kw.get("preserve_thinking").and_then(|v| v.as_bool()),
            add_generation_prompt: true,
        };
        let got = tpl.render(&msgs, &opts).unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = String::from_utf8(unhex(c["rendered_utf8_hex"].as_str().unwrap())).unwrap();
        if got != want {
            let n = got.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count();
            panic!("{name}: rendering differs at byte {n}:\n--- ours ---\n{got}\n--- HF ---\n{want}");
        }
        let ids = tok.encode(&got).unwrap();
        let want_ids: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(ids, want_ids, "{name}: token ids differ");
        println!("{name}: byte-identical ({} bytes), {} ids identical", got.len(), ids.len());
        ok += 1;
    }
    assert_eq!(ok, cases.len());
    println!("template {ok}/{} byte-identical", cases.len());
}
