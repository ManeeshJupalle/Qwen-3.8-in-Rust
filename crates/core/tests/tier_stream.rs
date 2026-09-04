//! Phase 4.2 gate on the tiny model (no big file needed): a streamed load produces the resident load's
//! greedy ids bit for bit, at every split and ring depth, and reads each streamed layer from disk exactly
//! once per pass.
//!
//! The tiny 9-layer model (`models/tiny/tiny-f32.gguf`, 7 DeltaNet + 2 GQA layers) is loaded resident, then
//! under budgets chosen so that 0, 2, 5 and 8 layers are pinned, with 1, 2 and 3 ring slots. Every run
//! prefills the same 3-id prompt and decodes 32 greedy tokens. Checks: identical ids; the consumer's
//! disk-byte counter equals passes x streamed span bytes exactly (the model asserts it per pass too); the
//! I/O thread's reads never run more than one prefetch ahead of the consumes for any layer; after shutdown
//! the bytes read equal the bytes consumed plus at most `n_slots` prefetched layers. An impossible budget is
//! refused before anything is allocated, with the shortfall in the message.

mod common;

use aqueduct_core::model::{argmax, LoadOpts, Model, ModelError};
use aqueduct_core::os::{arena_bytes, DirectFile, DEFAULT_CHUNK};
use aqueduct_core::tier::{MemoryPlan, PlanInput, PlanParams};
use aqueduct_core::{Gguf, ModelConfig};

const PROMPT: [u32; 3] = [1, 2, 3];
const STEPS: usize = 32;

fn decode(model: &Model) -> Vec<u32> {
    let mut state = model.new_state();
    state.reserve(PROMPT.len() + STEPS + 1);
    let hs = model.prefill(&PROMPT, &mut state);
    let hidden = model.hidden();
    let mut logits = vec![0f32; model.vocab()];
    model.logits_of(&hs[(PROMPT.len() - 1) * hidden..], &mut logits);
    let mut out = Vec::new();
    let mut next = argmax(&logits);
    for _ in 0..STEPS {
        out.push(next);
        model.forward_token(next, &mut state, &mut logits);
        next = argmax(&logits);
    }
    out
}

/// The budget at which exactly `k` layers are pinned with `n_slots` slots (the plan's own arithmetic on the
/// tiny model's header, with the drive's real sector size).
fn budget_for(input: &PlanInput, sector: u64, k: u32, n_slots: usize) -> u64 {
    let base = MemoryPlan::compute(input, &PlanParams::new(Some(0), 64, n_slots, sector)).resident_bytes;
    let prefix: u64 = (0..k).map(|i| arena_bytes(input.layer_bytes(i as usize), sector)).sum();
    if k >= input.n_layer {
        return base + prefix; // everything pinned: no ring
    }
    let largest = (k..input.n_layer).map(|i| input.layer_bytes(i as usize)).max().unwrap_or(0);
    base + n_slots as u64 * arena_bytes(largest, sector) + prefix
}

#[test]
fn streamed_decode_matches_resident_and_reads_each_layer_once_per_pass() {
    let path = common::root().join("models").join("tiny").join("tiny-f32.gguf");
    assert!(path.exists(), "tiny GGUF missing at {}", path.display());
    let g = Gguf::open(&path).expect("open");
    let cfg = ModelConfig::from_gguf(&g).expect("config");
    let input = PlanInput::new(&g, &cfg);
    let sector = DirectFile::open(&path, DEFAULT_CHUNK, 1).expect("direct open").sector as u64;
    let n_layer = cfg.n_layer;

    let resident = Model::load(&path, 2).expect("resident load");
    assert_eq!(resident.plan.pinned, n_layer);
    let ids_res = decode(&resident);
    assert_eq!(ids_res.len(), STEPS);
    println!("resident ids: {ids_res:?}");
    drop(resident);

    for n_slots in [1usize, 2, 3] {
        for pinned in [0u32, 2, 5, 8] {
            let budget = budget_for(&input, sector, pinned, n_slots);
            let opts = LoadOpts { budget: Some(budget), max_pos: 64, n_slots, large_pages: false, verbose: false, ..LoadOpts::default() };
            let mut model = Model::load_with(&path, 2, &opts).unwrap_or_else(|e| panic!("load with {pinned} pinned / {n_slots} slots: {e}"));
            let everything = budget_for(&input, sector, n_layer, n_slots);
            if budget >= everything {
                // one slot for a one-layer suffix costs that layer's arena: the plan rightly pins it instead
                assert_eq!((model.plan.pinned, model.plan.n_slots), (n_layer, 0), "budget {budget} holds everything");
                assert!(model.stream_stats().is_none());
                println!("{pinned} pinned / {n_slots} slots: a ring would cost what the last layer costs, everything pinned instead");
                continue;
            }
            assert_eq!(model.plan.pinned, pinned, "budget {budget} should pin exactly {pinned} layers");
            assert_eq!(model.plan.n_slots, n_slots);
            let streamed_bytes: u64 = (pinned..n_layer).map(|i| input.layer_bytes(i as usize)).sum();
            assert_eq!(model.streamed_bytes_per_pass(), streamed_bytes);
            assert!(model.plan.total <= budget);

            let ids = decode(&model);
            assert_eq!(ids, ids_res, "{pinned} pinned / {n_slots} slots: streamed ids differ from resident");

            // rule 4: one prefill pass + 32 decode passes, each exactly the streamed spans
            let s = model.stream_stats().expect("streaming");
            let passes = 1 + STEPS as u64;
            assert_eq!(s.consumed_bytes, passes * streamed_bytes, "consumed bytes");
            assert_eq!(s.consumed_layers, passes * (n_layer - pinned) as u64);
            model.shutdown_streaming();
            let s = model.stream_stats().unwrap();
            // reads are the consumes plus the prefetch that was in flight or ready: at most n_slots layers
            assert!(s.read_bytes >= s.consumed_bytes, "read {} < consumed {}", s.read_bytes, s.consumed_bytes);
            let ahead = s.reads - s.consumed_layers;
            assert!(ahead <= n_slots as u64, "{ahead} reads ahead of the consumes with {n_slots} slots");
            // every read asked the OS for the aligned superset, so io_bytes exceeds read_bytes by at most 2 sectors per read
            assert!(s.io_bytes >= s.read_bytes && s.io_bytes <= s.read_bytes + s.reads * 2 * sector, "io {} read {} reads {}", s.io_bytes, s.read_bytes, s.reads);
            println!(
                "{pinned} pinned / {n_slots} slots: budget {budget}, total {}, {passes} passes x {streamed_bytes} bytes = {} consumed, {} read ({} reads, {} ahead), io {} bytes, wait {:.1} ms, read {:.1} ms",
                model.plan.total,
                s.consumed_bytes,
                s.read_bytes,
                s.reads,
                ahead,
                s.io_bytes,
                s.wait_ns as f64 / 1e6,
                s.read_ns as f64 / 1e6
            );
        }
    }
}

#[test]
fn impossible_budget_is_refused_before_allocation() {
    let path = common::root().join("models").join("tiny").join("tiny-f32.gguf");
    let opts = LoadOpts { budget: Some(1 << 20), max_pos: 64, n_slots: 2, verbose: false, ..LoadOpts::default() };
    match Model::load_with(&path, 2, &opts) {
        Err(ModelError::Plan(e)) => {
            let m = e.to_string();
            assert!(m.contains("short by") && m.contains("needs"), "{m}");
            println!("{m}");
        }
        Err(e) => panic!("wrong error: {e}"),
        Ok(_) => panic!("a 1 MiB budget must be refused"),
    }
}
