//! `aqueduct bench membw`: the memory-read ceiling every matvec is judged against (Phase 3).
//! `aqueduct bench kernels`: throughput of the row kernels, scalar vs AVX2, one thread vs all threads.
//! Matrices are `rows x cols` of random block bytes (every bit pattern is a valid ggml block), the shape of
//! `ffn_gate` by default (17408 x 5120), so they exceed the L3 cache like a real projection does; GB/s counts
//! the weight bytes read per second, which is the number that bounds decode speed.

use std::time::Instant;

use aqueduct_core::kernels::matvec::{matvec, Act, ActVec, WeightMat};
use aqueduct_core::kernels::q8::Q8Row;
use aqueduct_core::kernels::q8k::Q8KRow;
use aqueduct_core::kernels::rmsnorm::rmsnorm;
use aqueduct_core::kernels::simd::{avx2_detected, force_scalar};
use aqueduct_core::kernels::softmax::softmax;
use aqueduct_core::GgmlType;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn arg(args: &[String], name: &str, default: usize) -> Result<usize, String> {
    match args.iter().position(|a| a == name) {
        Some(i) => args.get(i + 1).ok_or_else(|| format!("{name} needs a value"))?.parse().map_err(|e| format!("{name}: {e}")),
        None => Ok(default),
    }
}

/// Time `f` for at least `reps` runs and about `min_secs` seconds; returns seconds per run.
fn time_it(mut f: impl FnMut(), reps: usize, min_secs: f64) -> f64 {
    f();
    let mut n = 0;
    let t0 = Instant::now();
    while n < reps || t0.elapsed().as_secs_f64() < min_secs {
        f();
        n += 1;
    }
    t0.elapsed().as_secs_f64() / n as f64
}

pub fn kernels(args: &[String]) -> Result<(), String> {
    let all_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads = arg(args, "--threads", all_threads)?;
    let rows = arg(args, "--rows", 17408)?;
    let cols = arg(args, "--cols", 5120)?;
    let reps = arg(args, "--reps", 3)?;
    let avx2 = avx2_detected();
    let with_scalar = !args.iter().any(|a| a == "--no-scalar");
    let mut thread_list = vec![1usize];
    if threads >= 4 {
        thread_list.push(threads / 2);
    }
    if threads > 1 {
        thread_list.push(threads);
    }
    println!("# aqueduct bench kernels: rows={rows} cols={cols} reps>={reps}, threads {thread_list:?} (available {all_threads}); avx2+f16c detected: {avx2}");
    println!("# matvec: weight GB/s = rows * row_bytes / seconds per matvec (the bytes a projection streams per token)");
    println!("# act: q8_0 = Q8_0 activations (the production path for every type); q8_k = ggml's Q8_K activations (opt-in for K-quants)");
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let x: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
    let xa = ActVec::new(&x);
    let paths: Vec<(&str, bool)> = match (avx2, with_scalar) {
        (true, true) => vec![("scalar", true), ("avx2", false)],
        (true, false) => vec![("avx2", false)],
        (false, _) => vec![("scalar", true)],
    };
    println!("{:<8} {:<5} {:<7} {:>4} {:>12} {:>10} {:>10}", "kernel", "act", "path", "thr", "ms/matvec", "GB/s", "Gelem/s");
    for t in [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K, GgmlType::Q8_0, GgmlType::Q4_0] {
        let (bs, ts) = t.block_layout();
        let row_bytes = (cols as u64 / bs * ts) as usize;
        let data: Vec<u8> = (0..rows * row_bytes).map(|_| (rng.next() >> 24) as u8).collect();
        let w = WeightMat::new(t, rows, cols, data);
        let bytes = (rows * row_bytes) as f64;
        let mut y = vec![0f32; rows];
        let kquant = matches!(t, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K);
        let acts: Vec<(&str, Act<'_>)> = if kquant { vec![("q8_0", Act::Q8(xa.q8())), ("q8_k", Act::Q8K(xa.q8k()))] } else { vec![("q8_0", Act::Q8(xa.q8()))] };
        for (act_name, act) in &acts {
            for (name, scalar) in &paths {
                for &thr in &thread_list {
                    force_scalar(*scalar);
                    let s = time_it(|| matvec(&w, *act, &mut y, thr), reps, 1.0);
                    println!("{:<8} {:<5} {:<7} {:>4} {:>12.2} {:>10.2} {:>10.2}", t.name(), act_name, name, thr, s * 1e3, bytes / s / 1e9, (rows * cols) as f64 / s / 1e9);
                }
            }
        }
    }
    force_scalar(false);
    // row kernels: single thread, GB/s of f32 input
    let n_rows = 2048;
    let xs: Vec<f32> = (0..n_rows * cols).map(|_| rng.f32() * 4.0).collect();
    let wn: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
    println!();
    println!("{:<12} {:<7} {:>12} {:>10}", "row kernel", "path", "us/row", "GB/s in");
    for (name, scalar) in &paths {
        force_scalar(*scalar);
        let s = time_it(
            || {
                for r in 0..n_rows {
                    std::hint::black_box(Q8Row::quantize(&xs[r * cols..(r + 1) * cols]));
                }
            },
            reps,
            0.5,
        ) / n_rows as f64;
        println!("{:<12} {:<7} {:>12.2} {:>10.2}", "q8_quantize", name, s * 1e6, (cols * 4) as f64 / s / 1e9);
        let s = time_it(
            || {
                for r in 0..n_rows {
                    std::hint::black_box(Q8KRow::quantize(&xs[r * cols..(r + 1) * cols]));
                }
            },
            reps,
            0.5,
        ) / n_rows as f64;
        println!("{:<12} {:<7} {:>12.2} {:>10.2}", "q8k_quantize", name, s * 1e6, (cols * 4) as f64 / s / 1e9);
        let mut y = vec![0f32; cols];
        let s = time_it(
            || {
                for r in 0..n_rows {
                    rmsnorm(&xs[r * cols..(r + 1) * cols], &wn, 1e-6, &mut y);
                    std::hint::black_box(&y);
                }
            },
            reps,
            0.5,
        ) / n_rows as f64;
        println!("{:<12} {:<7} {:>12.2} {:>10.2}", "rmsnorm", name, s * 1e6, (cols * 4) as f64 / s / 1e9);
        let s = time_it(
            || {
                for r in 0..n_rows {
                    softmax(&xs[r * cols..(r + 1) * cols], &mut y);
                    std::hint::black_box(&y);
                }
            },
            reps,
            0.5,
        ) / n_rows as f64;
        println!("{:<12} {:<7} {:>12.2} {:>10.2}", "softmax", name, s * 1e6, (cols * 4) as f64 / s / 1e9);
    }
    force_scalar(false);
    Ok(())
}

// ------------------------------------------------------------------------------------------------ membw

/// Wrapping sum of `x`, four 256-bit accumulators (AVX2), scalar tail.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(x: &[u64]) -> u64 {
    use std::arch::x86_64::*;
    let p = x.as_ptr();
    let n16 = x.len() / 16 * 16;
    let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256());
    let mut i = 0;
    while i < n16 {
        a0 = _mm256_add_epi64(a0, _mm256_loadu_si256(p.add(i) as *const __m256i));
        a1 = _mm256_add_epi64(a1, _mm256_loadu_si256(p.add(i + 4) as *const __m256i));
        a2 = _mm256_add_epi64(a2, _mm256_loadu_si256(p.add(i + 8) as *const __m256i));
        a3 = _mm256_add_epi64(a3, _mm256_loadu_si256(p.add(i + 12) as *const __m256i));
        i += 16;
    }
    let s = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
    let mut lanes = [0u64; 4];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, s);
    let mut total = lanes.iter().fold(0u64, |a, &b| a.wrapping_add(b));
    for &v in &x[n16..] {
        total = total.wrapping_add(v);
    }
    total
}

fn sum_chunk(x: &[u64]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: feature checked at runtime.
        return unsafe { sum_avx2(x) };
    }
    x.iter().fold(0u64, |a, &b| a.wrapping_add(b))
}

/// Sum-reduce `buf` with `threads` threads over contiguous chunks; returns (sum, seconds).
fn sum_threads(buf: &[u64], threads: usize) -> (u64, f64) {
    let t0 = Instant::now();
    let total = if threads <= 1 {
        sum_chunk(buf)
    } else {
        let chunk = buf.len().div_ceil(threads);
        std::thread::scope(|s| {
            let handles: Vec<_> = buf.chunks(chunk).map(|c| s.spawn(move || sum_chunk(c))).collect();
            handles.into_iter().fold(0u64, |a, h| a.wrapping_add(h.join().unwrap()))
        })
    };
    (total, t0.elapsed().as_secs_f64())
}

/// `aqueduct bench membw [--gib 2] [--runs 5] [--threads N]`: read bandwidth of a buffer larger than any
/// cache, measured as a multi-threaded sum-reduction (every byte read once, nothing written), at one thread,
/// at half the hardware threads (the physical cores on an SMT machine) and at every hardware thread.
/// Best of `runs` is the ceiling; all runs are printed so throttling is visible.
pub fn membw(args: &[String]) -> Result<(), String> {
    let all_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let gib = arg(args, "--gib", 2)?;
    let runs = arg(args, "--runs", 5)?;
    let extra = arg(args, "--threads", 0)?;
    let n = gib << 27; // u64 elements
    let bytes = (n * 8) as f64;
    println!("# aqueduct bench membw: {gib} GiB buffer of u64, sum-reduce (AVX2 loads: {}), best of {runs}; hardware threads {all_threads}", cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx2"));
    let t0 = Instant::now();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut buf: Vec<u64> = Vec::with_capacity(n);
    buf.extend((0..n).map(|_| rng.next()));
    let expected = sum_chunk(&buf);
    println!("# fill + first pass: {:.2} s", t0.elapsed().as_secs_f64());
    let mut list = vec![1usize];
    if all_threads >= 4 {
        list.push(all_threads / 2);
    }
    list.push(all_threads);
    if extra > 0 && !list.contains(&extra) {
        list.push(extra);
    }
    println!("{:<8} {:>10} {:>10}   runs (GB/s)", "threads", "best GB/s", "worst GB/s");
    for &t in &list {
        let mut gbs = Vec::with_capacity(runs);
        for _ in 0..runs {
            let (s, secs) = sum_threads(&buf, t);
            if s != expected {
                return Err(format!("membw: sum mismatch at {t} threads"));
            }
            gbs.push(bytes / secs / 1e9);
        }
        let best = gbs.iter().cloned().fold(0f64, f64::max);
        let worst = gbs.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("{:<8} {:>10.2} {:>10.2}   {}", t, best, worst, gbs.iter().map(|g| format!("{g:.2}")).collect::<Vec<_>>().join(" "));
    }
    Ok(())
}
