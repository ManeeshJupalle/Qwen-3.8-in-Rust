//! Phase 5.2 gate on the tiny model: greedy with `--spec` emits the plain loop's ids bit for bit, at
//! `k = 1..4`, resident and streamed (2 of 9 layers pinned, 2 slots), 200 tokens on the 3 fixture prompts;
//! one disk pass per round while streaming.
//!
//! The tiny model's random MTP head almost never guesses right, which would leave the `m > 0` rollback paths
//! untested, so a second part forces the drafts: the true continuation corrupted at position `j`, cycling
//! `j = 0..k`, so every acceptance count from 0 to `k` occurs, and after every round the whole carried state
//! (DeltaNet states, conv windows, KV caches, position) is compared bit for bit with a plain state that
//! consumed the same tokens.

mod common;

use aqueduct_core::layers::MixerState;
use aqueduct_core::model::{argmax, LoadOpts, Model, State};
use aqueduct_core::os::{arena_bytes, DirectFile, DEFAULT_CHUNK};
use aqueduct_core::sample::{Sampler, SamplerConfig};
use aqueduct_core::tier::{MemoryPlan, PlanInput, PlanParams};
use aqueduct_core::{Gguf, ModelConfig};

const STEPS: usize = 200;

fn tiny_gguf() -> std::path::PathBuf {
    let p = common::root().join("models").join("tiny").join("tiny-f32.gguf");
    assert!(p.exists(), "tiny GGUF missing at {}", p.display());
    p
}

fn prompts() -> Vec<(String, Vec<u32>)> {
    let p = common::json(&common::fixtures().join("prompts.json"));
    p["prompts"].as_array().unwrap().iter().map(|x| (x["name"].as_str().unwrap().to_string(), x["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect())).collect()
}

/// The plain greedy loop: `n` emitted tokens; the state has consumed the prompt and the first `n - 1` of them.
fn plain(model: &Model, ids: &[u32], n: usize) -> (Vec<u32>, State) {
    let mut state = model.new_state();
    state.reserve(ids.len() + n + 2);
    let hs = model.prefill(ids, &mut state);
    let hidden = model.hidden();
    let mut logits = vec![0f32; model.vocab()];
    model.logits_of(&hs[(ids.len() - 1) * hidden..], &mut logits);
    let mut out = vec![argmax(&logits)];
    while out.len() < n {
        model.forward_token(*out.last().unwrap(), &mut state, &mut logits);
        out.push(argmax(&logits));
    }
    (out, state)
}

/// A plain state advanced token by token to any consumed count (the prompt, then plain tokens in order).
struct PlainTracker {
    state: State,
    consumed: usize,
    logits: Vec<f32>,
}

impl PlainTracker {
    fn new(model: &Model, ids: &[u32], max_consume: usize) -> PlainTracker {
        let mut state = model.new_state();
        state.reserve(ids.len() + max_consume + 2);
        model.prefill(ids, &mut state);
        PlainTracker { state, consumed: 0, logits: vec![0f32; model.vocab()] }
    }
    /// The state after consuming exactly `consume` of `toks` (monotone).
    fn at(&mut self, model: &Model, toks: &[u32], consume: usize) -> &State {
        assert!(consume >= self.consumed, "plain tracker only moves forward");
        for &t in &toks[self.consumed..consume] {
            model.forward_token(t, &mut self.state, &mut self.logits);
        }
        self.consumed = consume;
        &self.state
    }
}

fn assert_states_equal(a: &State, b: &State, what: &str) {
    assert_eq!(a.pos, b.pos, "{what}: position");
    for (i, (x, y)) in a.layers.iter().zip(&b.layers).enumerate() {
        match (x, y) {
            (MixerState::DeltaNet(p), MixerState::DeltaNet(q)) => {
                assert!(p.rec.iter().zip(&q.rec).all(|(u, v)| u.to_bits() == v.to_bits()), "{what}: layer {i} recurrent state differs");
                assert!(p.conv.iter().zip(&q.conv).all(|(u, v)| u.to_bits() == v.to_bits()), "{what}: layer {i} conv window differs");
            }
            (MixerState::Attention(p), MixerState::Attention(q)) => {
                assert_eq!(p.len, q.len, "{what}: layer {i} cache length");
                assert!(p.k.iter().zip(&q.k).all(|(u, v)| u.to_bits() == v.to_bits()) && p.k.len() == q.k.len(), "{what}: layer {i} K cache differs");
                assert!(p.v.iter().zip(&q.v).all(|(u, v)| u.to_bits() == v.to_bits()) && p.v.len() == q.v.len(), "{what}: layer {i} V cache differs");
            }
            _ => panic!("state kinds differ"),
        }
    }
}

type Force<'a> = Option<&'a mut dyn FnMut(usize, &[u32]) -> Vec<u32>>;
type Check<'a> = Option<&'a mut dyn FnMut(usize, &State, &[u32])>;

/// The speculative loop: `n` emitted tokens (the state may have consumed a few more than `n - 1`: the last
/// round's extra accepted tokens). `force(round, emitted_so_far) -> drafts` overrides the MTP's drafts.
fn spec(model: &Model, ids: &[u32], k: usize, n: usize, mut force: Force<'_>, mut check: Check<'_>) -> (Vec<u32>, State) {
    let mut state = model.new_state_spec(k).expect("spec state");
    state.reserve(ids.len() + n + k + 2);
    let mut sampler = Sampler::new(SamplerConfig::greedy(), model.vocab());
    let mut out = vec![model.spec_prefill(ids, &mut state, &mut sampler)];
    let mut round = 0;
    while out.len() < n {
        if let Some(f) = force.as_mut() {
            let d = f(round, &out);
            model.spec_set_drafts(&mut state, &d);
        }
        let x_p = *out.last().unwrap();
        model.spec_round(&mut state, x_p, &mut sampler);
        let emitted = state.spec.as_ref().unwrap().emitted.clone();
        out.extend_from_slice(&emitted);
        if let Some(c) = check.as_mut() {
            c(round, &state, &out);
        }
        round += 1;
    }
    out.truncate(n);
    (out, state)
}

#[test]
fn spec_greedy_ids_equal_plain_ids_resident_and_streamed() {
    let path = tiny_gguf();
    let model = Model::load(&path, 2).expect("load tiny");
    let vocab = model.vocab() as u32;
    let mut truths: Vec<(String, Vec<u32>, Vec<u32>)> = Vec::new();
    for (name, ids) in prompts() {
        let (plain_ids, _) = plain(&model, &ids, STEPS + 8);
        // ---- the MTP's own drafts, k = 1..4
        for k in 1..=4 {
            let (got, st) = spec(&model, &ids, k, STEPS, None, None);
            assert_eq!(got, plain_ids[..STEPS], "{name}: k={k} resident spec ids differ from plain");
            let s = &st.spec.as_ref().unwrap().stats;
            println!("{name} k={k} resident: {STEPS} ids identical; rounds {}, accepted {} of {} drafts (mean {:.3}/round), hist {:?}", s.rounds, s.accepted, s.drafted, s.mean_accepted(), s.hist);
        }
        // ---- forced drafts: every acceptance count 0..k, state checked after every round
        for k in 1..=4 {
            let truth = plain_ids.clone();
            let mut expect_m: Vec<usize> = Vec::new();
            let mut force = |round: usize, out: &[u32]| -> Vec<u32> {
                let j = round % (k + 1);
                let n = out.len();
                let mut d: Vec<u32> = truth[n..n + k].to_vec();
                if j < k {
                    d[j] = (d[j] + 1) % vocab;
                }
                expect_m.push(j);
                d
            };
            let mut checked = 0usize;
            let mut tracker = PlainTracker::new(&model, &ids, STEPS + k + 4);
            let mut check = |round: usize, st: &State, out: &[u32]| {
                let consumed = st.pos as usize - ids.len();
                assert_eq!(consumed, out.len() - 1, "{name}: k={k} round {round}: consumed {consumed} vs emitted {}", out.len());
                assert_eq!(&out[..out.len().min(truth.len())], &truth[..out.len().min(truth.len())], "{name}: k={k} round {round}: ids");
                let p = tracker.at(&model, &truth, consumed);
                assert_states_equal(st, p, &format!("{name} k={k} round {round}"));
                checked += 1;
            };
            let (got, st) = spec(&model, &ids, k, STEPS, Some(&mut force), Some(&mut check));
            assert_eq!(got, plain_ids[..STEPS], "{name}: k={k} forced-draft ids differ from plain");
            let s = &st.spec.as_ref().unwrap().stats;
            let mut hist_expect = vec![0u64; k + 1];
            for &j in &expect_m[..s.rounds as usize] {
                hist_expect[j] += 1;
            }
            assert_eq!(s.hist, hist_expect, "{name}: k={k} acceptance histogram");
            assert!(s.hist.iter().all(|&c| c > 0), "{name}: k={k}: every acceptance count 0..{k} must occur: {:?}", s.hist);
            println!("{name} k={k} forced: ids identical, {} rounds, hist {:?}, state equal after every round ({checked} checks)", s.rounds, s.hist);
        }
        truths.push((name, ids, plain_ids));
    }
    drop(model);

    // ---- streamed: 2 of 9 layers pinned, 2 slots, the plan sized for k = 4 (the largest run)
    let g = Gguf::open(&path).expect("open");
    let cfg = ModelConfig::from_gguf(&g).expect("config");
    let input = PlanInput::new(&g, &cfg);
    let sector = DirectFile::open(&path, DEFAULT_CHUNK, 1).expect("direct open").sector as u64;
    let (pinned, n_slots, k_max) = (2u32, 2usize, 4usize);
    let max_pos = 64 + STEPS + k_max + 2;
    let mut params = PlanParams::new(Some(0), max_pos as u64, n_slots, sector);
    params.spec_k = k_max as u64;
    let base = MemoryPlan::compute(&input, &params).resident_bytes;
    let largest = (pinned..cfg.n_layer).map(|i| input.layer_bytes(i as usize)).max().unwrap();
    let prefix: u64 = (0..pinned).map(|i| arena_bytes(input.layer_bytes(i as usize), sector)).sum();
    let budget = base + n_slots as u64 * arena_bytes(largest, sector) + prefix;
    drop(g);
    let opts = LoadOpts { budget: Some(budget), max_pos, n_slots, large_pages: false, spec_k: k_max, ..LoadOpts::default() };
    let model = Model::load_with(&path, 2, &opts).expect("streamed load");
    assert_eq!((model.plan.pinned, model.plan.n_slots), (pinned, n_slots));
    let per_pass = model.streamed_bytes_per_pass();
    for (name, ids, plain_ids) in &truths {
        for k in 1..=k_max {
            let before = model.stream_stats().unwrap().consumed_bytes;
            let (got, st) = spec(&model, ids, k, STEPS, None, None);
            let after = model.stream_stats().unwrap().consumed_bytes;
            assert_eq!(&got, &plain_ids[..STEPS], "{name}: k={k} streamed spec ids differ from plain");
            let s = &st.spec.as_ref().unwrap().stats;
            // rule 5: one pass for the prefill, one per round
            assert_eq!(after - before, (1 + s.rounds) * per_pass, "{name}: k={k}: disk bytes over {} rounds", s.rounds);
            println!("{name} k={k} streamed ({pinned} pinned): {STEPS} ids identical; {} rounds, one disk pass each ({per_pass} bytes)", s.rounds);
        }
    }
}

/// A second prompt fed to a state that already ran speculative rounds (a later chat turn): `spec_prefill`
/// forgets the MTP's draft rows, feeds the new rows at the right positions, and the ids that follow are the
/// plain loop's on the same token sequence.
#[test]
fn spec_prefill_on_a_used_state_matches_the_plain_loop() {
    let model = Model::load(tiny_gguf(), 2).expect("load tiny");
    let (name, ids) = prompts().into_iter().next().unwrap();
    let second: Vec<u32> = vec![733, 279, 1496, 7909, 11, 91420];
    let (n1, n2) = (40usize, 60usize);
    // plain: prompt, n1 tokens, the second prompt, n2 tokens
    let (mut plain_ids, mut pst) = plain(&model, &ids, n1);
    let mut logits = vec![0f32; model.vocab()];
    let last = *plain_ids.last().unwrap();
    model.forward_token(last, &mut pst, &mut logits); // consume the last emitted token before the new prompt
    let hs = model.prefill(&second, &mut pst);
    let hidden = model.hidden();
    model.logits_of(&hs[(second.len() - 1) * hidden..], &mut logits);
    let mut plain2 = vec![argmax(&logits)];
    while plain2.len() < n2 {
        model.forward_token(*plain2.last().unwrap(), &mut pst, &mut logits);
        plain2.push(argmax(&logits));
    }
    plain_ids.extend_from_slice(&plain2);
    for k in 1..=3 {
        let mut sampler = Sampler::new(SamplerConfig::greedy(), model.vocab());
        let mut st = model.new_state_spec(k).expect("spec");
        st.reserve(ids.len() + n1 + second.len() + n2 + 2 * k + 8);
        let mut out = vec![model.spec_prefill(&ids, &mut st, &mut sampler)];
        while out.len() < n1 {
            let x = *out.last().unwrap();
            model.spec_round(&mut st, x, &mut sampler);
            out.extend_from_slice(&st.spec.as_ref().unwrap().emitted);
        }
        out.truncate(n1);
        assert_eq!(out, plain_ids[..n1], "{name}: k={k} first segment");
        // the plain loop consumed out[..n1 - 1]; the spec state may have consumed more (extra accepted rows), so
        // bring both to the same point: feed the spec state whatever plain tokens it has not consumed yet
        let consumed = st.pos as usize - ids.len();
        assert!(consumed >= n1 - 1);
        // tokens the spec state consumed beyond n1 - 1 are plain tokens too (they were accepted drafts); the
        // plain state consumed exactly n1 tokens (the last emitted one, fed above): align by feeding the rest
        let mut seg2: Vec<u32> = plain_ids[consumed..n1].to_vec();
        seg2.extend_from_slice(&second);
        // if the spec state consumed past n1 (accepted drafts beyond the cut), the plain comparison is only valid
        // when those tokens equal the plain continuation; they do (identity), but the second prompt must then
        // start after them: skip such states
        if consumed > n1 {
            println!("{name}: k={k}: spec state consumed {consumed} > {n1}, second segment skipped");
            continue;
        }
        let first2 = model.spec_prefill(&seg2, &mut st, &mut sampler);
        let mut out2 = vec![first2];
        while out2.len() < n2 {
            let x = *out2.last().unwrap();
            model.spec_round(&mut st, x, &mut sampler);
            out2.extend_from_slice(&st.spec.as_ref().unwrap().emitted);
        }
        out2.truncate(n2);
        assert_eq!(out2, plain2, "{name}: k={k} second segment after a used-state prefill");
        println!("{name}: k={k}: {n1} + {n2} ids identical across a second prefill on the used state");
    }
}
