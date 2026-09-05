//! Phase 5.5: the blocked GEMM allocates nothing. `matmul_fn` is the verification batch's path (Phase 5,
//! `tests/decode_alloc.rs` part d) and that test runs on the tiny F32 model, which never reaches the K-quant
//! kernels; this one puts K-quant weights and Q8_K rows through `matmul_fn` under the counting allocator, for
//! batches inside one tile and across several tiles and row blocks. One `#[test]` per binary, deliberately:
//! the allocator counter is global.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use aqueduct_core::kernels::matvec::{matmul_fn, set_matmul_tile, ActBuf, WeightMat, DEFAULT_MATMUL_T, ROW_BLOCK, T_MAX};
use aqueduct_core::GgmlType;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

struct Counting;

// SAFETY: every method forwards to the system allocator with the same arguments; the counters are atomics.
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

/// Where the first counted allocation came from (captured with the counter disarmed, so the capture's own
/// allocations are not counted), for the failure message.
static FIRST: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

#[inline]
fn bump() {
    if ARMED.load(Ordering::Relaxed) && ALLOCS.fetch_add(1, Ordering::Relaxed) == 0 {
        ARMED.store(false, Ordering::SeqCst);
        let who = std::thread::current().name().unwrap_or("<unnamed>").to_string();
        let bt = format!("thread {who}:\n{}", std::backtrace::Backtrace::force_capture());
        if let Ok(mut f) = FIRST.lock() {
            *f = Some(bt);
        }
        ARMED.store(true, Ordering::SeqCst);
    }
}

#[global_allocator]
static A: Counting = Counting;

fn count_allocs(f: impl FnOnce()) -> (usize, String) {
    *FIRST.lock().unwrap() = None;
    ALLOCS.store(0, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    f();
    ARMED.store(false, Ordering::SeqCst);
    (ALLOCS.load(Ordering::SeqCst), FIRST.lock().unwrap().take().unwrap_or_default())
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

#[test]
fn blocked_matmul_fn_allocates_nothing() {
    let mut rng = Rng(0xA110_C550_0000_0003);
    let (rows, cols) = (ROW_BLOCK * 2 + 5, 5120usize);
    for &t in &[GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K] {
        let (bs, ts) = t.block_layout();
        let rb = (cols as u64 / bs * ts) as usize;
        let mut data: Vec<u8> = (0..rows * rb).map(|_| (rng.next() >> 24) as u8).collect();
        // finite f16 scales (as tests/q8k.rs)
        let offs: &[usize] = if t == GgmlType::Q6_K { &[208] } else { &[0, 2] };
        for blk in data.chunks_mut(ts as usize) {
            for &off in offs {
                let bits = u16::from_le_bytes([blk[off], blk[off + 1]]);
                let fixed = (bits & 0x83FF) | ((12 + (bits >> 10) % 7) << 10);
                blk[off..off + 2].copy_from_slice(&fixed.to_le_bytes());
            }
        }
        let w = WeightMat::new(t, rows, cols, data);
        let n = 2 * T_MAX + 3;
        let xs: Vec<Vec<f32>> = (0..n).map(|_| (0..cols).map(|_| (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect()).collect();
        let mut bufs: Vec<ActBuf> = (0..n).map(|_| ActBuf::with_capacity(cols)).collect();
        for (b, x) in bufs.iter_mut().zip(&xs) {
            b.fill(x, &[t]);
        }
        let mut y = vec![0f32; n * rows];
        // the pool's workers are spawned on first use and finish their own start-up (thread-local state) on
        // their own threads a little later; spawn them here and let them settle before anything is counted
        matmul_fn(&w, n, &|i| bufs[i].act_for(t, &xs[i]), &mut y, 6);
        std::thread::sleep(std::time::Duration::from_millis(300));
        for &tile in &[1usize, 4, DEFAULT_MATMUL_T, T_MAX] {
            set_matmul_tile(tile);
            for &threads in &[1usize, 6] {
                // warm-up at this tile and thread count
                matmul_fn(&w, n, &|i| bufs[i].act_for(t, &xs[i]), &mut y, threads);
                for &m in &[1usize, 4, n] {
                    let (allocs, first) = count_allocs(|| matmul_fn(&w, m, &|i| bufs[i].act_for(t, &xs[i]), &mut y[..m * rows], threads));
                    assert_eq!(allocs, 0, "{t:?} tile {tile} threads {threads} batch {m}: matmul_fn allocated {allocs} times; the first from:\n{first}");
                }
            }
        }
        println!("{t:?}: matmul_fn allocation-free for tiles 1, 4, {DEFAULT_MATMUL_T}, {T_MAX}, batches 1, 4, {n}, at 1 and 6 threads");
    }
    set_matmul_tile(DEFAULT_MATMUL_T);
}
