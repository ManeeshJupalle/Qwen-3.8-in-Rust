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
//! **This file holds exactly one `#[test]`, deliberately.** The allocator counter is global, so it also
//! sees allocations made by any other test running at the same time, and cargo runs the tests in a binary
//! concurrently. One test per binary is what makes the count mean what it says.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

mod common;

use aqueduct_core::model::{argmax, Model};

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
    }
}
