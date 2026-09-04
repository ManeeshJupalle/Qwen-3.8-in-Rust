//! Phase 3.3 / 3.4 gates on the fully resident model (ignored: loads the 17.8 GB GGUF into RAM; run with
//! `cargo test --release --test phase3_e2e -- --ignored --nocapture`, output filed to docs/data/phase3_timing.txt).
//!
//! 3.3: for the 3 prompts, the batched prefill against the token-by-token feed: last-position logits within
//! the frozen Q8 ceiling (both are also checked against ref_gguf), argmax at every prompt position vs ref_gguf,
//! carried DeltaNet state / conv window / KV cache max relative difference (expected 0: the batched path is
//! the same arithmetic per row).
//! 3.4: greedy 32 tokens per prompt from the batched prefill, first 16 against tests/fixtures/ref_llamacpp,
//! the top-1/top-2 logit margin at the first divergence, decoded text; load / prefill / decode timing, weight
//! bytes per token over decode seconds as a fraction of membw, peak RSS vs weights + state.

mod common;

use std::time::Instant;

use aqueduct_core::layers::MixerState;
use aqueduct_core::model::{argmax, Model};
use aqueduct_core::rss::peak_rss_bytes;
use aqueduct_core::Tok;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}

fn max_abs(v: &[f32]) -> f64 {
    v.iter().fold(0f64, |m, x| m.max(x.abs() as f64))
}

/// max |a - b| / max |b| over the state vectors (0 when identical).
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let m = max_abs(b);
    if m == 0.0 {
        max_abs_diff(a, b)
    } else {
        max_abs_diff(a, b) / m
    }
}

fn top2(v: &[f32]) -> (u32, f32, u32, f32) {
    let mut b1 = 0usize;
    for i in 1..v.len() {
        if v[i] > v[b1] {
            b1 = i;
        }
    }
    let mut b2 = if b1 == 0 { 1 } else { 0 };
    for i in 0..v.len() {
        if i != b1 && v[i] > v[b2] {
            b2 = i;
        }
    }
    (b1 as u32, v[b1], b2 as u32, v[b2])
}

#[test]
#[ignore = "loads the whole 17.8 GB model; minutes; run with --release --ignored --nocapture"]
fn prefill_gate_and_end_to_end() {
    let threads = std::env::var("AQUEDUCT_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let membw: f64 = std::env::var("AQUEDUCT_MEMBW").ok().and_then(|s| s.parse().ok()).unwrap_or(28.9);
    let t_load = Instant::now();
    let model = Model::load(common::gguf_path(), threads).expect("load");
    let load_s = t_load.elapsed().as_secs_f64();
    let rss_load = peak_rss_bytes().unwrap_or(0);
    println!("load: {load_s:.1} s, {:.3} GB of weights, peak RSS {:.3} GB, threads {threads}", model.weight_bytes as f64 / 1e9, rss_load as f64 / 1e9);
    let tok = Tok::from_file(common::tokenizer_path()).expect("tokenizer");
    let hidden = model.hidden();
    let vocab = model.vocab();

    let prompts = common::json(&common::fixtures().join("prompts.json"));
    let ceil = common::json(&common::fixture("q8_ceilings.json"));
    let factor = ceil["factor"].as_f64().unwrap();
    let ref_dir = common::fixture("ref_gguf");
    let llama_dir = common::fixture("ref_llamacpp");
    let mut summary: Vec<String> = Vec::new();
    let mut max_pos = 0usize;
    let mut n_generated = 0usize;

    for p in prompts["prompts"].as_array().unwrap() {
        let name = p["name"].as_str().unwrap();
        let ids: Vec<u32> = p["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let t = ids.len();
        println!("== prompt {name} ({t} ids) ==");

        // ---- 3.3: sequential feed vs batched prefill
        let mut sa = model.new_state();
        let t0 = Instant::now();
        let mut hs_seq: Vec<f32> = Vec::with_capacity(t * hidden);
        for &id in &ids {
            hs_seq.extend(model.forward_hidden(id, &mut sa));
        }
        let seq_s = t0.elapsed().as_secs_f64();
        let mut logits_seq_last = vec![0f32; vocab];
        model.logits_of(&hs_seq[(t - 1) * hidden..], &mut logits_seq_last);

        let mut sb = model.new_state();
        let t0 = Instant::now();
        let hs_bat = model.prefill(&ids, &mut sb);
        let bat_s = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let logits_bat = model.logits_all(&hs_bat, t);
        let head_s = t0.elapsed().as_secs_f64();
        let logits_bat_last = &logits_bat[(t - 1) * vocab..];

        let hidden_diff = max_abs_diff(&hs_bat, &hs_seq);
        let logit_diff = max_abs_diff(logits_bat_last, &logits_seq_last);
        // state comparison
        let mut state_rel = 0f64;
        let mut cache_rel = 0f64;
        for (a, b) in sa.layers.iter().zip(&sb.layers) {
            match (a, b) {
                (MixerState::DeltaNet(x), MixerState::DeltaNet(y)) => {
                    state_rel = state_rel.max(rel(&y.rec, &x.rec)).max(rel(&y.conv, &x.conv));
                }
                (MixerState::Attention(x), MixerState::Attention(y)) => {
                    assert_eq!((x.len, y.len), (t, t));
                    cache_rel = cache_rel.max(rel(&y.k, &x.k)).max(rel(&y.v, &x.v));
                }
                _ => panic!("state kinds differ"),
            }
        }
        // reference: argmax at every position, last logits
        let meta = common::json(&ref_dir.join(format!("{name}.json")));
        let ref_pos: Vec<u32> = meta["argmax_per_position"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let pos_match = (0..t).filter(|&i| argmax(&logits_bat[i * vocab..(i + 1) * vocab]) == ref_pos[i]).count();
        let ref_logits = common::read_npy_f32(&ref_dir.join(format!("{name}.logits.npy")));
        let d_ref_bat = max_abs_diff(logits_bat_last, &ref_logits);
        let d_ref_seq = max_abs_diff(&logits_seq_last, &ref_logits);
        let ceiling = factor * ceil["measured"]["logits"][name]["vs_ref_gguf_max_diff"].as_f64().unwrap();
        println!(
            "prefill: sequential {seq_s:.2} s, batched {bat_s:.2} s ({:.2} tok/s) + head {head_s:.2} s; hidden max|diff| {hidden_diff:e}; last logits batched vs sequential max|diff| {logit_diff:e} (ceiling {ceiling:.4}); state max rel {state_rel:e}, cache max rel {cache_rel:e}; argmax per position {pos_match}/{t}; vs ref_gguf max|diff| batched {d_ref_bat:.4} sequential {d_ref_seq:.4} (ceiling {ceiling:.4})",
            t as f64 / bat_s
        );
        assert!(logit_diff <= ceiling, "{name}: batched vs sequential logits {logit_diff} > ceiling {ceiling}");
        assert!(d_ref_bat <= ceiling && d_ref_seq <= ceiling, "{name}: logits vs ref_gguf exceed the frozen ceiling");
        assert_eq!(pos_match, t, "{name}: argmax per position");
        assert_eq!(argmax(logits_bat_last), argmax(&ref_logits), "{name}: last argmax vs ref_gguf");
        max_pos += t;

        // ---- 3.4: greedy 32 from the batched state
        let llama = common::json(&llama_dir.join(format!("{name}.json")));
        let llama_ids: Vec<u32> = llama["greedy_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let mut out: Vec<u32> = Vec::new();
        let mut margins: Vec<f32> = Vec::new();
        let (mut next, mut l1, _, mut l2) = top2(logits_bat_last);
        let mut logits = vec![0f32; vocab];
        let mut step_secs: Vec<f64> = Vec::new();
        let mut stopped = None;
        while out.len() < 32 {
            if model.is_stop(next) {
                stopped = Some(next);
                break;
            }
            out.push(next);
            margins.push(l1 - l2);
            if out.len() == 32 {
                break;
            }
            let ts = Instant::now();
            model.forward_token(next, &mut sb, &mut logits);
            step_secs.push(ts.elapsed().as_secs_f64());
            let (a, x1, _, x2) = top2(&logits);
            next = a;
            l1 = x1;
            l2 = x2;
        }
        n_generated += out.len();
        let n_cmp = out.len().min(16).min(llama_ids.len());
        let first_div = (0..n_cmp).find(|&i| out[i] != llama_ids[i]);
        let matched = first_div.unwrap_or(n_cmp);
        let s_per_tok = step_secs.iter().sum::<f64>() / step_secs.len().max(1) as f64;
        let gbs = model.decode_bytes_per_token() as f64 / s_per_tok / 1e9;
        let text = tok.decode(&out).unwrap_or_default();
        println!("decode: {} tokens, {s_per_tok:.3} s/token = {gbs:.2} GB/s = {:.0}% of membw {membw}{}", out.len(), 100.0 * gbs / membw, stopped.map(|s| format!(" (stopped at {s})")).unwrap_or_default());
        println!("ids: {}", out.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        println!("llama.cpp first 16 match: {matched}/{n_cmp}{}", match first_div {
            Some(i) => format!("; first divergence at step {i}: ours {} vs llama.cpp {} with top1-top2 margin {:.4} (llama.cpp text {:?})", out[i], llama_ids[i], margins[i], llama["greedy_text"].as_str().unwrap_or("")),
            None => String::new(),
        });
        println!("margins: {}", margins.iter().map(|m| format!("{m:.2}")).collect::<Vec<_>>().join(" "));
        println!("text: {text:?}");
        summary.push(format!("{name}: llama.cpp match {matched}/{n_cmp}; prefill batched {bat_s:.2} s ({:.2} tok/s) vs sequential {seq_s:.2} s; decode {s_per_tok:.3} s/token = {gbs:.2} GB/s", t as f64 / bat_s));
    }
    let rss = peak_rss_bytes().unwrap_or(0);
    let expected = model.weight_bytes + model.state_bytes(max_pos / 3 + 32);
    println!("peak RSS {:.3} GB vs expected {:.3} GB (weights {:.3} GB + state), ratio {:.3}; {n_generated} tokens generated", rss as f64 / 1e9, expected as f64 / 1e9, model.weight_bytes as f64 / 1e9, rss as f64 / expected as f64);
    println!("load {load_s:.1} s");
    for l in &summary {
        println!("summary: {l}");
    }
}
