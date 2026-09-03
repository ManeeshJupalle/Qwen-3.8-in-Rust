//! `aqueduct bench kernels`: throughput of the row kernels, scalar vs AVX2, one thread vs all threads.
//! Matrices are `rows x cols` of random block bytes (every bit pattern is a valid ggml block), the shape of
//! `ffn_gate` by default (17408 x 5120), so they exceed the L3 cache like a real projection does; GB/s counts
//! the weight bytes read per second, which is the number that bounds decode speed.

use std::time::Instant;

use aqueduct_core::kernels::matvec::{matvec, Act, WeightMat};
use aqueduct_core::kernels::q8::Q8Row;
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
    println!("# aqueduct bench kernels: rows={rows} cols={cols} reps>={reps}, threads: 1 and {threads} (available {all_threads}); avx2+f16c detected: {avx2}");
    println!("# matvec: weight GB/s = rows * row_bytes / seconds per matvec (the bytes a projection streams per token)");
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let x: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
    let xq = Q8Row::quantize(&x);
    let paths: Vec<(&str, bool)> = if avx2 { vec![("scalar", true), ("avx2", false)] } else { vec![("scalar", true)] };
    println!("{:<8} {:<7} {:>4} {:>12} {:>10} {:>10}", "kernel", "path", "thr", "ms/matvec", "GB/s", "Gelem/s");
    for t in [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K] {
        let (bs, ts) = t.block_layout();
        let row_bytes = (cols as u64 / bs * ts) as usize;
        let data: Vec<u8> = (0..rows * row_bytes).map(|_| (rng.next() >> 24) as u8).collect();
        let w = WeightMat::new(t, rows, cols, data);
        let bytes = (rows * row_bytes) as f64;
        let mut y = vec![0f32; rows];
        for (name, scalar) in &paths {
            for &thr in &[1usize, threads] {
                force_scalar(*scalar);
                let s = time_it(|| matvec(&w, Act::Q8(&xq), &mut y, thr), reps, 1.0);
                println!("{:<8} {:<7} {:>4} {:>12.2} {:>10.2} {:>10.2}", t.name(), name, thr, s * 1e3, bytes / s / 1e9, (rows * cols) as f64 / s / 1e9);
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
