//! `aqueduct bench membw`: the memory-read ceiling every matvec is judged against (Phase 3).
//! `aqueduct bench kernels`: throughput of the row kernels, scalar vs AVX2, one thread vs all threads.
//! Matrices are `rows x cols` of random block bytes (every bit pattern is a valid ggml block), the shape of
//! `ffn_gate` by default (17408 x 5120), so they exceed the L3 cache like a real projection does; GB/s counts
//! the weight bytes read per second, which is the number that bounds decode speed.

use std::time::Instant;

use aqueduct_core::kernels::matvec::{matmul, matvec, set_matmul_threads, set_matmul_tile, Act, ActVec, WeightMat, DEFAULT_MATMUL_T};
use aqueduct_core::kernels::q8::Q8Row;
use aqueduct_core::kernels::q8k::Q8KRow;
use aqueduct_core::kernels::rmsnorm::rmsnorm;
use aqueduct_core::kernels::simd::{avx2_detected, force_scalar};
use aqueduct_core::kernels::softmax::softmax;
use aqueduct_core::{logical_cores, physical_cores, GgmlType};

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
    let (physical, logical) = (physical_cores(), logical_cores());
    let threads = arg(args, "--threads", logical)?;
    let rows = arg(args, "--rows", 17408)?;
    let cols = arg(args, "--cols", 5120)?;
    let reps = arg(args, "--reps", 3)?;
    let avx2 = avx2_detected();
    let with_scalar = !args.iter().any(|a| a == "--no-scalar");
    // one thread, the physical cores (the engine's default) and the hardware threads, so the SMT step is visible
    let mut thread_list = vec![1usize];
    if physical > 1 && physical < threads {
        thread_list.push(physical);
    }
    if threads > 1 {
        thread_list.push(threads);
    }
    println!("# aqueduct bench kernels: rows={rows} cols={cols} reps>={reps}, threads {thread_list:?} ({physical} physical cores, {logical} logical; the engine defaults to {physical}); avx2+f16c detected: {avx2}");
    println!("# matvec: weight GB/s = rows * row_bytes / seconds per matvec (the bytes a projection streams per token)");
    println!("# act: q8_k = ggml's Q8_K activations (the production path for K-quant weights, Phase 3.5); q8_0 = Q8_0 activations (--q8-fine for K-quants; the only form for Q8_0 / Q4_0 weights)");
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

// ------------------------------------------------------------------------------------------------ matmul tiles

/// Median of a non-empty slice (sorted copy).
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

/// `aqueduct bench matmul [--n 4,32] [--tiles 0,1,2,4,8,16] [--reps 7] [--threads N] [--shapes gate,down]`:
/// the blocked GEMM (Phase 5.5) against the Phase 3.3 per-row loop, per activation tile. For each K-quant
/// type and shape (`gate` = 17408 x 5120, the ffn_gate shape with 5120-wide activations; `down` = 5120 x 17408,
/// the ffn_down shape with 17408-wide activations) and each batch size `n`, every tile in `--tiles` runs a
/// `matmul` of `n` random Q8_K rows; the tiles are alternated within each repetition (an interleaved A/B, so
/// every variant sees the same machine state) and the median per tile is reported: ms per call, ms per
/// activation row, ns per (super-block, activation) and the speedup over tile 0. The single-activation
/// `matvec` of the same matrix is the memory-bound floor each line is judged against.
pub fn matmul_tiles(args: &[String]) -> Result<(), String> {
    let (physical, logical) = (physical_cores(), logical_cores());
    let threads = arg(args, "--threads", physical)?;
    let reps = arg(args, "--reps", 7)?;
    let list = |name: &str, default: &str| -> Result<Vec<usize>, String> {
        let s = match args.iter().position(|a| a == name) {
            Some(i) => args.get(i + 1).ok_or_else(|| format!("{name} needs a value"))?.clone(),
            None => default.to_string(),
        };
        s.split(',').map(|x| x.trim().parse::<usize>().map_err(|e| format!("{name}: {e}"))).collect()
    };
    let ns = list("--n", "4,32")?;
    let tiles = list("--tiles", "0,1,2,4,8,16")?;
    // --same-act: every activation of the batch is the same row, so the tile is one Q8_K row (L1-resident):
    // separates the cost of the arithmetic from the cost of fetching the tile
    let same_act = args.iter().any(|a| a == "--same-act");
    let shapes_arg = match args.iter().position(|a| a == "--shapes") {
        Some(i) => args.get(i + 1).ok_or_else(|| "--shapes needs a value".to_string())?.clone(),
        None => "gate,down".to_string(),
    };
    let mut shapes: Vec<(&str, usize, usize)> = Vec::new();
    for s in shapes_arg.split(',') {
        match s.trim() {
            "gate" => shapes.push(("gate", 17408, 5120)),
            "down" => shapes.push(("down", 5120, 17408)),
            "small" => shapes.push(("small", 512, 5120)),
            "tiny" => shapes.push(("tiny", 4, 5120)),
            other => return Err(format!("--shapes: unknown shape {other} (gate, down, small, tiny)")),
        }
    }
    println!("# aqueduct bench matmul: threads {threads} ({physical} physical, {logical} logical), reps {reps}, n {ns:?}, tiles {tiles:?} (0 = the Phase 3.3 per-row loop){}; avx2+f16c: {}", if same_act { "; --same-act: one activation row repeated (L1-resident tile)" } else { "" }, avx2_detected());
    // the clock the ns numbers below are to be read against: a dependent chain of 1-cycle adds, on one
    // thread and on `threads` threads at once (the all-core AVX clock is what the kernels run at)
    let clock = |n: usize| -> f64 {
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..n)
                .map(|_| {
                    s.spawn(|| {
                        // `x -> (x ^ i) + (x >> 7)`: a 2-cycle dependent chain (shift and xor in parallel, then
                        // the add) that has no closed form and cannot be vectorised; nothing else in the loop
                        let iters = 300_000_000u64;
                        let mut x = std::hint::black_box(1u64);
                        let t0 = Instant::now();
                        for i in 0..iters {
                            x = (x ^ i).wrapping_add(x >> 7);
                        }
                        std::hint::black_box(x);
                        2.0 * iters as f64 / t0.elapsed().as_secs_f64() / 1e9
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).fold(0f64, f64::max)
        })
    };
    println!("# clock (dependent 1-cycle adds): {:.2} GHz on one thread, {:.2} GHz with {threads} threads busy", clock(1), clock(threads));
    println!("# blocked GEMM (matmul_t): each weight super-block unpacked once per tile of T activation rows; tiles alternated per repetition, medians reported");
    println!("# ms/row = ms per call / n (the marginal cost of one activation row when the weights are shared); ns/(sb,act) = ms per call * 1e6 / (rows * super-blocks * n)");
    println!("# matvec = one activation, the memory-bound floor of the same matrix (ms, and weight GB/s)");
    set_matmul_threads(threads); // the batch policy would otherwise take every hardware thread
    let mut rng = Rng(0x7A11_E5B7_0000_0001);
    for &(sname, rows, cols) in &shapes {
        let nb = cols / 256;
        for t in [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K] {
            let (bs, ts) = t.block_layout();
            let row_bytes = (cols as u64 / bs * ts) as usize;
            let mut data: Vec<u8> = (0..rows * row_bytes).map(|_| (rng.next() >> 24) as u8).collect();
            // finite f16 scales so no NaN reaches the accumulators (the timing does not care, the sums do)
            let offs: &[usize] = if t == GgmlType::Q6_K { &[208] } else { &[0, 2] };
            for blk in data.chunks_mut(ts as usize) {
                for &off in offs {
                    let bits = u16::from_le_bytes([blk[off], blk[off + 1]]);
                    let fixed = (bits & 0x83FF) | ((12 + (bits >> 10) % 7) << 10);
                    blk[off..off + 2].copy_from_slice(&fixed.to_le_bytes());
                }
            }
            let w = WeightMat::new(t, rows, cols, data);
            let bytes = (rows * row_bytes) as f64;
            let n_max = *ns.iter().max().unwrap_or(&1);
            let xs: Vec<Vec<f32>> = (0..n_max).map(|_| (0..cols).map(|_| rng.f32()).collect()).collect();
            let acts: Vec<ActVec<'_>> = xs.iter().map(|x| ActVec::new(x)).collect();
            let batch: Vec<Act<'_>> = acts.iter().map(|a| Act::Q8K(if same_act { acts[0].q8k() } else { a.q8k() })).collect();
            let mut y1 = vec![0f32; rows];
            let s1 = time_it(|| matvec(&w, batch[0], &mut y1, threads), 3, 0.5);
            println!();
            println!("{:<5} {:<5} {:>5}x{:<5} matvec {:>8.3} ms  {:>7.2} GB/s of weights", sname, t.name(), rows, cols, s1 * 1e3, bytes / s1 / 1e9);
            println!("{:<5} {:<5} {:>4} {:>5} {:>10} {:>9} {:>12} {:>9} {:>11}", "shape", "type", "n", "tile", "ms/call", "ms/row", "ns/(sb,act)", "x tile0", "eff GB/s");
            for &n in &ns {
                let mut y = vec![0f32; n * rows];
                let mut times: Vec<Vec<f64>> = vec![Vec::new(); tiles.len()];
                for rep in 0..reps {
                    for k in 0..tiles.len() {
                        let idx = (k + rep) % tiles.len(); // rotate the order so no tile always runs first
                        set_matmul_tile(tiles[idx]);
                        matmul(&w, &batch[..n], &mut y, threads); // warm
                        let t0 = Instant::now();
                        matmul(&w, &batch[..n], &mut y, threads);
                        times[idx].push(t0.elapsed().as_secs_f64());
                    }
                }
                let base = times.iter().zip(&tiles).find(|(_, &tl)| tl == 0).map(|(v, _)| median(v));
                for (k, &tile) in tiles.iter().enumerate() {
                    let m = median(&times[k]);
                    let per_row = m / n as f64;
                    let ns_sb = m * 1e9 / (rows * nb * n) as f64;
                    let x0 = base.map_or(String::from("-"), |b| format!("{:.2}", b / m));
                    println!("{:<5} {:<5} {:>4} {:>5} {:>10.3} {:>9.3} {:>12.2} {:>9} {:>11.2}", sname, t.name(), n, tile, m * 1e3, per_row * 1e3, ns_sb, x0, bytes * n as f64 / m / 1e9);
                }
            }
            set_matmul_tile(DEFAULT_MATMUL_T);
        }
    }
    set_matmul_threads(0);
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

/// Best-of-`runs` read bandwidth (GB/s) of a `gib` GiB buffer at `threads` threads: the short form
/// `aqueduct doctor` uses. Returns (best, all runs).
pub fn membw_best(gib: usize, runs: usize, threads: usize) -> Result<(f64, Vec<f64>), String> {
    let n = gib << 27;
    let bytes = (n * 8) as f64;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut buf: Vec<u64> = Vec::with_capacity(n);
    buf.extend((0..n).map(|_| rng.next()));
    let expected = sum_chunk(&buf);
    let mut gbs = Vec::with_capacity(runs);
    for _ in 0..runs {
        let (s, secs) = sum_threads(&buf, threads);
        if s != expected {
            return Err(format!("membw: sum mismatch at {threads} threads"));
        }
        gbs.push(bytes / secs / 1e9);
    }
    Ok((gbs.iter().cloned().fold(0f64, f64::max), gbs))
}

/// `aqueduct bench membw [--gib 2] [--runs 5] [--threads N]`: read bandwidth of a buffer larger than any
/// cache, measured as a multi-threaded sum-reduction (every byte read once, nothing written), at one thread,
/// at half the hardware threads (the physical cores on an SMT machine) and at every hardware thread.
/// Best of `runs` is the ceiling; all runs are printed so throttling is visible.
pub fn membw(args: &[String]) -> Result<(), String> {
    let (physical, all_threads) = (physical_cores(), logical_cores());
    let gib = arg(args, "--gib", 2)?;
    let runs = arg(args, "--runs", 5)?;
    let extra = arg(args, "--threads", 0)?;
    let n = gib << 27; // u64 elements
    let bytes = (n * 8) as f64;
    println!(
        "# aqueduct bench membw: {gib} GiB buffer of u64, sum-reduce (AVX2 loads: {}), best of {runs}; {physical} physical cores, {all_threads} logical",
        cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx2")
    );
    let t0 = Instant::now();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut buf: Vec<u64> = Vec::with_capacity(n);
    buf.extend((0..n).map(|_| rng.next()));
    let expected = sum_chunk(&buf);
    println!("# fill + first pass: {:.2} s", t0.elapsed().as_secs_f64());
    let mut list = vec![1usize];
    if physical > 1 && physical < all_threads {
        list.push(physical);
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
