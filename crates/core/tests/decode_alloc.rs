//! Phase 3.6 item 1: the decode step allocates nothing after warmup.
//!
//! A counting global allocator wraps the system allocator and, while armed, counts every `alloc`,
//! `alloc_zeroed` and `realloc` (a `dealloc` is not an allocation, but a `realloc` is: it is where a
//! growing `Vec` gets its new block). The test loads the tiny 9-layer model
//! (`models/tiny/tiny-f32.gguf`, both mixer kinds: 7 DeltaNet layers and 2 GQA layers), reserves the
//! state for the sequence it is about to run, takes one warm-up token outside the counter, then arms it
//! and runs 32 `forward_token` steps.
//!
//! Why this matters beyond speed: Phase 4 runs the engine under a hard memory cap, and an allocation in
//! the per-token path is where an OOM comes from. A decode step whose working set is entirely
//! preallocated in `State` cannot fail that way, and its peak RSS is known before the first token.
//!
//! Phase 4 extends the same test to the streaming tier (part c): the model is loaded under a budget that
//! pins 2 of the 9 layers and streams the other 7 through a 2-slot ring, and the count must still be zero.
//! The counter is global, so it also sees the ring's I/O thread: its per-token work (waiting on a slot,
//! one unbuffered read into a preallocated arena, the counters) allocates nothing either. The streamed
//! ids are compared with the resident ones from part (b) at the same thread count.
//!
//! **This file holds exactly one `#[test]`, deliberately.** The allocator counter is global, so it also
//! sees allocations made by any other test running at the same time, and cargo runs the tests in a binary
//! concurrently. One test per binary is what makes the count mean what it says.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

mod common;

use aqueduct_core::model::{argmax, LoadOpts, Model};
use aqueduct_core::os::{arena_bytes, DirectFile, DEFAULT_CHUNK};
use aqueduct_core::tier::{MemoryPlan, PlanInput, PlanParams};
use aqueduct_core::{Gguf, ModelConfig};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

struct Counting;

// SAFETY: every method forwards to the system allocator with the same arguments; the counters are atomics
// and do not themselves allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        bump();
        System.realloc(p, l, new)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        bump();
        System.alloc_zeroed(l)
    }
}

#[inline]
fn bump() {
    if ARMED.load(Ordering::Relaxed) {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Run `f` with the counter armed; returns how many allocations happened while it ran.
fn count_allocs(f: impl FnOnce()) -> usize {
    ALLOCS.store(0, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    f();
    ARMED.store(false, Ordering::SeqCst);
    ALLOCS.load(Ordering::SeqCst)
}

const STEPS: usize = 32;

#[test]
fn decode_is_allocation_free_and_matches_the_allocating_path() {
    let path = common::root().join("models").join("tiny").join("tiny-f32.gguf");
    assert!(path.exists(), "tiny GGUF missing at {} (tools/make_tiny_checkpoint.py)", path.display());

    // ---- (a) the preallocated decode path computes what the allocating one does, bit for bit.
    // `forward_token` reuses buffers; `forward_hidden` + `logits_of` allocate fresh ones every step.
    {
        let model = Model::load(&path, 2).expect("load tiny model");
        let vocab = model.vocab();
        let mut sa = model.new_state();
        sa.reserve(STEPS + 2);
        let mut sb = model.new_state();
        let (mut la, mut lb) = (vec![0f32; vocab], vec![0f32; vocab]);
        let (mut ta, mut tb) = (1u32, 1u32);
        for step in 0..STEPS {
            model.forward_token(ta, &mut sa, &mut la);
            let h = model.forward_hidden(tb, &mut sb);
            model.logits_of(&h, &mut lb);
            assert!(
                la.iter().zip(&lb).all(|(x, y)| x.to_bits() == y.to_bits()),
                "step {step}: preallocated and allocating paths differ at index {:?}",
                la.iter().zip(&lb).position(|(x, y)| x.to_bits() != y.to_bits())
            );
            ta = argmax(&la);
            tb = argmax(&lb);
        }
        println!("preallocated and allocating decode agree bit for bit over {STEPS} steps");
    }

    // ---- (b) zero allocations across 32 decode steps, at several thread counts: the pool partitions
    // heads and rows differently, so a buffer that is per-participant rather than per-head or per-row
    // would only show up above one participant.
    let mut resident_ids: Vec<(usize, [u32; STEPS])> = Vec::new();
    for threads in [1usize, 2, 6] {
        let model = Model::load(&path, threads).expect("load tiny model");
        let vocab = model.vocab();
        let mut logits = vec![0f32; vocab];
        let mut state = model.new_state();
        // the whole point: everything the run needs is sized up front
        state.reserve(STEPS + 2);
        let mut ids = [0u32; STEPS];

        // warm-up: the pool's threads, the quantisers' capacities and the cache's first push happen here
        let mut tok = 1u32;
        model.forward_token(tok, &mut state, &mut logits);
        tok = argmax(&logits);

        let allocs = count_allocs(|| {
            for slot in ids.iter_mut() {
                model.forward_token(tok, &mut state, &mut logits);
                tok = argmax(&logits);
                *slot = tok;
            }
        });
        assert_eq!(allocs, 0, "{threads} threads: {allocs} allocations across {STEPS} decode steps (want 0)");
        println!("{threads} threads: 0 allocations across {STEPS} decode steps, {} positions carried", state.pos);
        resident_ids.push((threads, ids));
    }

    // ---- (c) Phase 4: the same, streaming 7 of the 9 layers through a 2-slot ring. The I/O thread is
    // counted too (the allocator is global), so the ring's per-token work must be allocation-free as well.
    {
        let g = Gguf::open(&path).expect("open");
        let cfg = ModelConfig::from_gguf(&g).expect("config");
        let input = PlanInput::new(&g, &cfg);
        let sector = DirectFile::open(&path, DEFAULT_CHUNK, 1).expect("direct open").sector as u64;
        let (pinned, n_slots) = (2u32, 2usize);
        let base = MemoryPlan::compute(&input, &PlanParams::new(Some(0), STEPS as u64 + 2, n_slots, sector)).resident_bytes;
        let largest = (pinned..cfg.n_layer).map(|i| input.layer_bytes(i as usize)).max().unwrap();
        let prefix: u64 = (0..pinned).map(|i| arena_bytes(input.layer_bytes(i as usize), sector)).sum();
        let budget = base + n_slots as u64 * arena_bytes(largest, sector) + prefix;
        drop(g);
        for &(threads, expect) in &resident_ids {
            let opts = LoadOpts { budget: Some(budget), max_pos: STEPS + 2, n_slots, large_pages: false, verbose: false, ..LoadOpts::default() };
            let model = Model::load_with(&path, threads, &opts).expect("streamed load");
            assert_eq!((model.plan.pinned, model.plan.n_slots), (pinned, n_slots));
            let vocab = model.vocab();
            let mut logits = vec![0f32; vocab];
            let mut state = model.new_state();
            state.reserve(STEPS + 2);
            let mut ids = [0u32; STEPS];
            let mut tok = 1u32;
            model.forward_token(tok, &mut state, &mut logits);
            tok = argmax(&logits);
            let before = model.stream_stats().unwrap();
            let allocs = count_allocs(|| {
                for slot in ids.iter_mut() {
                    model.forward_token(tok, &mut state, &mut logits);
                    tok = argmax(&logits);
                    *slot = tok;
                }
            });
            let after = model.stream_stats().unwrap();
            assert_eq!(allocs, 0, "{threads} threads, streaming: {allocs} allocations across {STEPS} decode steps (want 0)");
            assert_eq!(ids, expect, "{threads} threads: streamed ids differ from resident");
            assert_eq!(after.consumed_bytes - before.consumed_bytes, STEPS as u64 * model.streamed_bytes_per_pass(), "disk bytes over {STEPS} steps");
            println!(
                "{threads} threads, {pinned} pinned + {} streamed through {n_slots} slots: 0 allocations across {STEPS} decode steps, {} bytes from disk per token, ids identical to resident",
                cfg.n_layer - pinned,
                model.streamed_bytes_per_pass()
            );
        }
    }

    // ---- (d) Phase 5: speculative rounds (`--spec 2`) on the same streamed split: zero allocations across
    // 32 rounds (the MTP's re-feed and chained drafts, the verification batch, the snapshot and replay, the
    // I/O thread), and exactly one disk pass per round.
    {
        use aqueduct_core::sample::{Sampler, SamplerConfig};
        let g = Gguf::open(&path).expect("open");
        let cfg = ModelConfig::from_gguf(&g).expect("config");
        let input = PlanInput::new(&g, &cfg);
        let sector = DirectFile::open(&path, DEFAULT_CHUNK, 1).expect("direct open").sector as u64;
        let (pinned, n_slots, k) = (2u32, 2usize, 2usize);
        let max_pos = 3 * STEPS + 8;
        let mut params = PlanParams::new(Some(0), max_pos as u64, n_slots, sector);
        params.spec_k = k as u64;
        let base = MemoryPlan::compute(&input, &params).resident_bytes;
        let largest = (pinned..cfg.n_layer).map(|i| input.layer_bytes(i as usize)).max().unwrap();
        let prefix: u64 = (0..pinned).map(|i| arena_bytes(input.layer_bytes(i as usize), sector)).sum();
        let budget = base + n_slots as u64 * arena_bytes(largest, sector) + prefix;
        drop(g);
        let opts = LoadOpts { budget: Some(budget), max_pos, n_slots, large_pages: false, spec_k: k, ..LoadOpts::default() };
        let model = Model::load_with(&path, 2, &opts).expect("streamed load");
        assert_eq!((model.plan.pinned, model.plan.n_slots), (pinned, n_slots));
        let mut state = model.new_state_spec(k).expect("spec state");
        state.reserve(max_pos);
        let mut sampler = Sampler::new(SamplerConfig::greedy(), model.vocab());
        let mut x = model.spec_prefill(&[1, 2, 3], &mut state, &mut sampler);
        // warm-up round
        model.spec_round(&mut state, x, &mut sampler);
        x = *state.spec.as_ref().unwrap().emitted.last().unwrap();
        let before = model.stream_stats().unwrap();
        let allocs = count_allocs(|| {
            for _ in 0..STEPS {
                model.spec_round(&mut state, x, &mut sampler);
                x = *state.spec.as_ref().unwrap().emitted.last().unwrap();
            }
        });
        let after = model.stream_stats().unwrap();
        assert_eq!(allocs, 0, "spec k={k}, streaming: {allocs} allocations across {STEPS} rounds (want 0)");
        assert_eq!(after.consumed_bytes - before.consumed_bytes, STEPS as u64 * model.streamed_bytes_per_pass(), "disk bytes over {STEPS} rounds");
        let s = &state.spec.as_ref().unwrap().stats;
        println!("spec k={k}, {pinned} pinned + {} streamed: 0 allocations across {STEPS} rounds, one disk pass per round ({} bytes), {} tokens emitted, {} drafts accepted", cfg.n_layer - pinned, model.streamed_bytes_per_pass(), s.emitted, s.accepted);
    }
}
