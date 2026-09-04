//! Persistent worker pool for the row-parallel kernels (Phase 3). One pool per process, spawned on first
//! use with `available_parallelism() - 1` workers; the calling thread is participant 0. A job is
//! `Fn(tid, n)` run on `n` participants, each deciding its own row range from `(tid, n)`; `run` returns
//! only when every participant has finished, so the job may borrow from the caller's stack.
//!
//! Dispatch latency matters: a decode token is ~700 jobs of about a millisecond. Workers therefore spin on
//! the generation counter for `SPIN_US` before blocking on a condvar (no CPU when idle for longer), and the
//! caller spins on the remaining count the same way. Callers are serialised (one job at a time). Thread-count
//! invariance is the callers' business: every row is computed by exactly one participant with the same
//! kernel, whatever `n` is.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

type Job<'a> = dyn Fn(usize, usize) + Sync + 'a;

/// How long a thread spins before it blocks.
const SPIN_US: u64 = 60;

struct State {
    job: Option<&'static Job<'static>>,
    n: usize,
}

struct Inner {
    state: Mutex<State>,
    /// Incremented (under `state`) for every job.
    generation: AtomicU64,
    /// Workers still to finish the current job.
    remaining: AtomicUsize,
    start: Condvar,
    done: Condvar,
    panicked: AtomicBool,
}

pub struct Pool {
    inner: Arc<Inner>,
    workers: usize,
    /// Serialises callers: one job at a time (tests call from several threads at once).
    caller: Mutex<()>,
}

static GLOBAL: OnceLock<Pool> = OnceLock::new();

/// The process-wide pool (spawned on first call).
pub fn global() -> &'static Pool {
    GLOBAL.get_or_init(|| {
        let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Pool::spawn(n.saturating_sub(1))
    })
}

fn worker(inner: Arc<Inner>, tid: usize) {
    let mut seen = 0u64;
    loop {
        // wait for a new generation: spin, then block
        let spin_until = Instant::now() + Duration::from_micros(SPIN_US);
        loop {
            let g = inner.generation.load(Ordering::Acquire);
            if g != seen {
                seen = g;
                break;
            }
            if Instant::now() < spin_until {
                std::hint::spin_loop();
                continue;
            }
            let mut st = inner.state.lock().unwrap();
            while inner.generation.load(Ordering::Acquire) == seen {
                st = inner.start.wait(st).unwrap();
            }
            drop(st);
        }
        let (job, n) = {
            let st = inner.state.lock().unwrap();
            (st.job.expect("pool: generation without a job"), st.n)
        };
        if tid < n && catch_unwind(AssertUnwindSafe(|| job(tid, n))).is_err() {
            inner.panicked.store(true, Ordering::SeqCst);
        }
        if inner.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _g = inner.state.lock().unwrap();
            inner.done.notify_one();
        }
    }
}

impl Pool {
    /// Spawn a pool with `workers` background threads (plus the caller = `workers + 1` participants).
    pub fn spawn(workers: usize) -> Pool {
        let inner = Arc::new(Inner {
            state: Mutex::new(State { job: None, n: 0 }),
            generation: AtomicU64::new(0),
            remaining: AtomicUsize::new(0),
            start: Condvar::new(),
            done: Condvar::new(),
            panicked: AtomicBool::new(false),
        });
        for w in 0..workers {
            let inner = Arc::clone(&inner);
            std::thread::Builder::new().name(format!("aqueduct-worker-{}", w + 1)).spawn(move || worker(inner, w + 1)).expect("spawn pool worker");
        }
        Pool { inner, workers, caller: Mutex::new(()) }
    }

    /// Caller plus workers.
    pub fn max_threads(&self) -> usize {
        self.workers + 1
    }

    /// Run `f(tid, n)` on `n = min(threads, max_threads())` participants (the caller is `tid` 0) and wait.
    pub fn run(&self, threads: usize, f: &Job<'_>) {
        let n = threads.max(1).min(self.workers + 1);
        if n == 1 {
            f(0, 1);
            return;
        }
        let _one_at_a_time = self.caller.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the job pointer is cleared and never used again once `remaining` reaches 0, and this
        // function does not return before that, so the borrow outlives every use.
        let job: &'static Job<'static> = unsafe { std::mem::transmute::<&Job<'_>, &'static Job<'static>>(f) };
        {
            let mut st = self.inner.state.lock().unwrap();
            st.job = Some(job);
            st.n = n;
            self.inner.remaining.store(self.workers, Ordering::Release);
            self.inner.generation.fetch_add(1, Ordering::AcqRel);
        }
        self.inner.start.notify_all();
        let caller = catch_unwind(AssertUnwindSafe(|| f(0, n)));
        // wait for the workers: spin, then block
        let spin_until = Instant::now() + Duration::from_micros(SPIN_US);
        while self.inner.remaining.load(Ordering::Acquire) > 0 {
            if Instant::now() < spin_until {
                std::hint::spin_loop();
                continue;
            }
            let mut st = self.inner.state.lock().unwrap();
            while self.inner.remaining.load(Ordering::Acquire) > 0 {
                st = self.inner.done.wait(st).unwrap();
            }
        }
        self.inner.state.lock().unwrap().job = None;
        if let Err(p) = caller {
            std::panic::resume_unwind(p);
        }
        if self.inner.panicked.swap(false, Ordering::SeqCst) {
            panic!("pool: a worker panicked inside a job");
        }
    }
}

/// A `*mut f32` shared with a job: participants write disjoint ranges of one buffer (a head's columns of
/// every row, a head's recurrent state). The caller guarantees disjointness.
#[derive(Clone, Copy)]
pub struct SharedMut(*mut f32);
unsafe impl Send for SharedMut {}
unsafe impl Sync for SharedMut {}

impl SharedMut {
    pub fn new(v: &mut [f32]) -> SharedMut {
        SharedMut(v.as_mut_ptr())
    }
    /// A mutable view of `len` elements from `start`.
    /// SAFETY: the range must lie inside the original slice and must not overlap the range of any other
    /// participant that is running at the same time.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, start: usize, len: usize) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(start), len)
    }
}
