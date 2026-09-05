//! Phase 5.1 gate: the MTP draft head on the tiny checkpoint against `tools/ref_mtp.py`
//! (`tests/fixtures/mtp/mtp/manifest.json`): the draft-1 logits within the tiny oracle's budget, then 20
//! chained drafts per prompt, 60/60 ids, with the top-1 logit at every step within the budget of that step.
//! The first target token must be the fixture's (the MTP's last prompt row pairs it with `h_{T-1}`).

mod common;

use aqueduct_core::model::Model;
use aqueduct_core::sample::{Sampler, SamplerConfig};

const EPS: f64 = f32::EPSILON as f64;

fn tiny_gguf() -> std::path::PathBuf {
    let p = common::root().join("models").join("tiny").join("tiny-f32.gguf");
    assert!(p.exists(), "tiny GGUF missing at {}: run `python tools/make_tiny_checkpoint.py`", p.display());
    p
}

fn max_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}

#[test]
fn mtp_drafts_match_the_reference_on_the_tiny_model() {
    let model = Model::load(tiny_gguf(), 4).expect("load tiny");
    let mtp = model.mtp.as_ref().expect("the tiny GGUF carries an MTP block (blk.9)");
    assert_eq!(mtp.index, model.cfg.n_layer);
    let hidden = model.hidden() as f64;
    let n_layer = model.cfg.n_layer as f64;
    let dir = common::fixture("mtp").join("mtp");
    let m = common::json(&dir.join("manifest.json"));
    let k = m["budget_k_per_layer"].as_f64().unwrap();
    let n_drafts = m["drafts"].as_u64().unwrap() as usize;
    let budget = |max_abs: f64| k * (n_layer + 3.0) * hidden.sqrt() * EPS * max_abs;
    let mut total = 0;
    let mut matched = 0;
    for c in m["cases"].as_array().unwrap() {
        let name = c["case"].as_str().unwrap();
        let ids: Vec<u32> = c["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let want_first = c["first_target_token"].as_u64().unwrap() as u32;
        let want_drafts: Vec<u32> = c["draft_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let want_logits_1 = common::read_npy_f32(&dir.join(c["draft_logits_1"]["file"].as_str().unwrap()));
        let max_abs_1 = c["draft_logits_1_max_abs"].as_f64().unwrap();

        // k = 1: the prefill's single draft is draft 1, and its logits are the draft-1 logits
        let mut sampler = Sampler::new(SamplerConfig::greedy(), model.vocab());
        let mut s1 = model.new_state_spec(1).expect("spec state");
        s1.reserve(ids.len() + 4);
        let x_t = model.spec_prefill(&ids, &mut s1, &mut sampler);
        assert_eq!(x_t, want_first, "{name}: first target token");
        let spec = s1.spec.as_ref().unwrap();
        assert_eq!(spec.drafts, vec![want_drafts[0]], "{name}: draft 1");
        assert_eq!(spec.mtp_rows(), ids.len(), "{name}: MTP rows after the prompt");
        let d1 = max_diff(&spec.mtp.logits, &want_logits_1);
        let b1 = budget(max_abs_1);
        println!("{name}: draft-1 logits max|diff| {d1:.3e} vs budget {b1:.3e} ({:.2} of budget)", d1 / b1);
        assert!(d1 <= b1, "{name}: draft-1 logits off by {d1:e} (budget {b1:e})");

        // k = 20: the whole chain in one prefill
        let mut sampler = Sampler::new(SamplerConfig::greedy(), model.vocab());
        let mut s = model.new_state_spec(n_drafts).expect("spec state");
        s.reserve(ids.len() + n_drafts + 2);
        let x_t = model.spec_prefill(&ids, &mut s, &mut sampler);
        assert_eq!(x_t, want_first);
        let spec = s.spec.as_ref().unwrap();
        let got: Vec<u32> = spec.drafts.clone();
        let ok = got.iter().zip(&want_drafts).filter(|(a, b)| a == b).count();
        println!("{name}: drafts {ok}/{n_drafts} match; got {got:?}");
        total += n_drafts;
        matched += ok;
        assert_eq!(got, want_drafts, "{name}: draft ids differ");
        assert_eq!(spec.mtp_rows(), ids.len() + n_drafts - 1, "{name}: MTP rows after the chain");
        // the last step's top-1 logit
        let last = n_drafts - 1;
        let want_top = c["top5_logits"][last][0].as_f64().unwrap() as f32;
        let got_top = spec.mtp.logits[want_drafts[last] as usize];
        let b = budget(c["max_abs"][last].as_f64().unwrap());
        let d = (got_top as f64 - want_top as f64).abs();
        println!("{name}: step {last} top-1 logit {got_top} vs {want_top} (diff {d:.3e}, budget {b:.3e}, margin {:.4})", c["margins"][last].as_f64().unwrap());
        assert!(d <= b, "{name}: step {last} top-1 logit off by {d:e}");
    }
    println!("drafts {matched}/{total}");
    assert_eq!(matched, total);
}
