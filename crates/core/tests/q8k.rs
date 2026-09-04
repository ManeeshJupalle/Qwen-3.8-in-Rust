//! Phase 3.1: the Q8_K activation path.
//! (a) `Q8KRow::quantize` against the Python port of ggml's `quantize_row_q8_K_ref` (bit-exact);
//! (b) the scalar K-quant x Q8_K kernels against an f64 reference on the Phase 2a dot fixture rows
//!     (derived budget `k * sqrt(n) * eps * max|term|`, k = 8, as `tests/kernels.rs`);
//! (c) AVX2 vs scalar: exact equality of every dot and of the quantiser on fixture rows and on random /
//!     hostile rows (rule 2: not a contract, but the kernels are written to have it and the test says so);
//! (d) `matvec` thread invariance (both activation forms), `matvec2` and `matmul` against the single-row
//!     kernel (bit-exact).
//! The Q8_0-grain K-quant kernels (the production path) are covered by `tests/kernels.rs` (fixtures, terms
//! budget) and `tests/avx2.rs` (AVX2 vs scalar).

mod common;

use std::path::PathBuf;

use aqueduct_core::kernels::kdot::{dot_q8k, dot_q8k_scalar};
use aqueduct_core::kernels::matvec::{matmul, matvec, matvec2, Act, ActVec, WeightMat};
use aqueduct_core::kernels::q8k::Q8KRow;
use aqueduct_core::kernels::simd::{avx2_detected, force_scalar};
use aqueduct_core::{dequantize, GgmlType};

const EPS: f64 = f32::EPSILON as f64;
const K_TYPES: [GgmlType; 3] = [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K];

struct Fx {
    dir: PathBuf,
    manifest: serde_json::Value,
}

impl Fx {
    fn load(name: &str) -> Fx {
        let dir = common::fixture("kernels").join(name);
        let manifest = common::json(&dir.join("manifest.json"));
        Fx { dir, manifest }
    }
    fn cases(&self) -> &Vec<serde_json::Value> {
        self.manifest["cases"].as_array().unwrap()
    }
    fn arr(&self, field: &serde_json::Value) -> Vec<f32> {
        common::read_npy_f32(&self.dir.join(field["file"].as_str().unwrap()))
    }
    fn raw(&self, field: &serde_json::Value) -> Vec<u8> {
        std::fs::read(self.dir.join(field["file"].as_str().unwrap())).unwrap()
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() >> 24) as u8).collect()
    }
    fn f32(&mut self, scale: f32) -> f32 {
        let u: f32 = (0..4).map(|_| (self.next() >> 40) as f32 / (1u64 << 24) as f32).sum::<f32>() - 2.0;
        u * scale
    }
    fn row(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(scale)).collect()
    }
}

fn type_by_name(name: &str) -> GgmlType {
    match name {
        "Q4_K" => GgmlType::Q4_K,
        "Q5_K" => GgmlType::Q5_K,
        "Q6_K" => GgmlType::Q6_K,
        "Q4_0" => GgmlType::Q4_0,
        "Q8_0" => GgmlType::Q8_0,
        "F32" => GgmlType::F32,
        other => panic!("unknown type {other}"),
    }
}

fn row_bytes(t: GgmlType, width: usize) -> usize {
    let (bs, ts) = t.block_layout();
    (width as u64 / bs * ts) as usize
}

/// Random block bytes whose f16 scale fields are finite (exponent forced into [2^-3, 2^4)): random bytes
/// alone include inf / NaN scales, whose NaN payloads differ between the scalar and AVX2 paths.
fn finite_weights(t: GgmlType, rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let rb = row_bytes(t, cols);
    let (bs, ts) = t.block_layout();
    let (bs, ts) = (bs as usize, ts as usize);
    let mut data = rng.bytes(rows * rb);
    let scale_offsets: &[usize] = match t {
        GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        GgmlType::Q8_0 | GgmlType::Q4_0 => &[0],
        _ => &[],
    };
    for blk in data.chunks_mut(ts) {
        for &off in scale_offsets {
            let bits = u16::from_le_bytes([blk[off], blk[off + 1]]);
            let exp = 12 + (bits >> 10) % 7; // biased exponent 12..19: 2^-3 .. 2^4
            let fixed = (bits & 0x83FF) | (exp << 10);
            blk[off..off + 2].copy_from_slice(&fixed.to_le_bytes());
        }
    }
    let _ = bs;
    data
}

/// Hostile activation rows for a width that is a multiple of 256.
fn hostile_acts(width: usize, rng: &mut Rng) -> Vec<(String, Vec<f32>)> {
    let mut v = vec![
        ("random".to_string(), rng.row(width, 1.0)),
        ("large".to_string(), rng.row(width, 1e30)),
        ("tiny".to_string(), rng.row(width, 1e-30)),
        ("all_equal".to_string(), vec![3.0; width]),
        ("neg_zero".to_string(), vec![-0.0; width]),
        ("alternating".to_string(), (0..width).map(|i| if i % 2 == 0 { 5.0 } else { -5.0 }).collect()),
    ];
    // ties to even: max element -127 makes iscale = 1, so values k + 0.5 are exact ties
    let mut ties: Vec<f32> = (0..width).map(|i| (i % 9) as f32 - 4.0 + 0.5).collect();
    ties[0] = -127.0;
    ties[300 % width] = 127.0;
    v.push(("ties_even".to_string(), ties));
    // the same magnitude with both signs: the first occurrence must decide the sign of the scale
    let mut pm: Vec<f32> = rng.row(width, 0.5);
    pm[7] = 4.0;
    pm[9] = -4.0;
    v.push(("first_max_positive".to_string(), pm.clone()));
    pm[7] = -4.0;
    pm[9] = 4.0;
    v.push(("first_max_negative".to_string(), pm));
    // one all-zero block inside a normal row
    let mut z = rng.row(width, 1.0);
    for x in &mut z[256..512.min(width)] {
        *x = 0.0;
    }
    v.push(("zero_block".to_string(), z));
    v
}

// ------------------------------------------------------------------------------------ (a) quantiser

#[test]
fn q8k_quantiser_matches_ported_reference_bit_exact() {
    let fx = Fx::load("q8k_quantize");
    assert_eq!(fx.manifest["exact"], true);
    let mut n_rows = 0;
    for c in fx.cases() {
        let width = c["width"].as_u64().unwrap() as usize;
        let x = fx.arr(&c["x"]);
        let raw = fx.raw(&c["raw"]);
        let rows = x.len() / width;
        let bpr = width / 256 * Q8KRow::BLOCK_BYTES;
        assert_eq!(raw.len(), rows * bpr);
        for r in 0..rows {
            let want = &raw[r * bpr..(r + 1) * bpr];
            let got = Q8KRow::quantize_scalar(&x[r * width..(r + 1) * width]);
            assert_eq!(got.to_ggml_bytes(), want, "{} row {} ({}): scalar quantiser differs from the port", c["case"], r, c["rows"][r]);
            let back = Q8KRow::from_ggml_bytes(want, width);
            assert_eq!(back, got);
            if avx2_detected() {
                force_scalar(false);
                let got_v = Q8KRow::quantize(&x[r * width..(r + 1) * width]);
                assert_eq!(got_v, got, "{} row {} ({}): AVX2 quantiser differs from scalar", c["case"], r, c["rows"][r]);
            }
            n_rows += 1;
        }
    }
    println!("q8k quantiser: {n_rows} fixture rows bit-exact with the Python port (scalar and AVX2)");
}

// ------------------------------------------------------------------------------------ (b) scalar vs f64

#[test]
fn kquant_q8k_dots_match_f64_reference() {
    let fx = Fx::load("dot");
    let k = fx.manifest["budget_k"].as_f64().unwrap();
    for c in fx.cases() {
        let t = type_by_name(c["ggml_type"].as_str().unwrap());
        if !K_TYPES.contains(&t) {
            continue;
        }
        let width = c["width"].as_u64().unwrap() as usize;
        let n_rows = c["n_rows"].as_u64().unwrap() as usize;
        let w_raw = fx.raw(&c["w_raw"]);
        let x_deq = fx.arr(&c["x_deq"]);
        let rb = row_bytes(t, width);
        assert_eq!(w_raw.len(), n_rows * rb);
        let mut worst = 0f64;
        let mut worst_cancel = 0f64;
        for r in 0..n_rows {
            let wrow = &w_raw[r * rb..(r + 1) * rb];
            let x = Q8KRow::quantize_scalar(&x_deq[r * width..(r + 1) * width]);
            let wd = dequantize(t, wrow, width).unwrap();
            let xd = x.dequantize();
            // The integer part of every super-block is exact; all f32 rounding happens on the 8 (main) and
            // 4 (min, Q4_K/Q5_K) lane products `d * lane`, on their accumulation across super-blocks and
            // on the final reduction. Lanes are partial sums over elements with random signs and cancel
            // against each other (docs/kquant-dot.md), so the error is bounded by the magnitudes of the
            // terms, not by the (cancelled) result: budget = k * eps * sum_j (|d*sc*q_j| + |dmin*m|) |x_j|,
            // the main and the min part of each weight taken separately because they are summed separately.
            let refsum: f64 = wd.iter().zip(&xd).map(|(&w, &x)| w as f64 * x as f64).sum();
            let bud = common::kquant_terms_budget(t, wrow, &wd, &xd, k);
            let mags = bud / (k * EPS);
            let got = dot_q8k_scalar(t, wrow, &x) as f64;
            let d = (got - refsum).abs();
            assert!(d <= bud, "{} row {} ({}): got {got} want {refsum} diff {d:e} > budget {bud:e}", c["case"], r, c["rows"][r]);
            if bud > 0.0 {
                worst = worst.max(d / bud);
            }
            if refsum != 0.0 {
                worst_cancel = worst_cancel.max(mags / refsum.abs());
            }
        }
        println!("{}: {n_rows} rows within budget (worst {:.4} of budget; largest sum|terms| / |result| = {:.1})", c["case"], worst, worst_cancel);
    }
}

// ------------------------------------------------------------------------------------ (c) AVX2 vs scalar

#[test]
fn avx2_q8k_kernels_match_scalar_bit_for_bit() {
    if !avx2_detected() {
        println!("AVX2 not available: skipped");
        return;
    }
    let fx = Fx::load("dot");
    let mut n = 0;
    for c in fx.cases() {
        let t = type_by_name(c["ggml_type"].as_str().unwrap());
        if !K_TYPES.contains(&t) {
            continue;
        }
        let width = c["width"].as_u64().unwrap() as usize;
        let n_rows = c["n_rows"].as_u64().unwrap() as usize;
        let w_raw = fx.raw(&c["w_raw"]);
        let x_deq = fx.arr(&c["x_deq"]);
        let rb = row_bytes(t, width);
        for r in 0..n_rows {
            let wrow = &w_raw[r * rb..(r + 1) * rb];
            let x = Q8KRow::quantize_scalar(&x_deq[r * width..(r + 1) * width]);
            force_scalar(false);
            let v = dot_q8k(t, wrow, &x);
            let s = dot_q8k_scalar(t, wrow, &x);
            assert_eq!(v.to_bits(), s.to_bits(), "{} row {}: avx2 {v} vs scalar {s}", c["case"], r);
            n += 1;
        }
    }
    // random weight bytes (every pattern is a valid block) x random and hostile activations
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut max_rel = 0f64;
    let mut non_finite = 0;
    for &t in &K_TYPES {
        for width in [256usize, 5120] {
            let rb = row_bytes(t, width);
            let acts = hostile_acts(width, &mut rng);
            for i in 0..200 {
                let wrow = rng.bytes(rb);
                let xf = if i < acts.len() { acts[i].1.clone() } else { rng.row(width, [1.0, 1e-3, 1e3][i % 3]) };
                force_scalar(true);
                let xs = Q8KRow::quantize(&xf);
                force_scalar(false);
                let xv = Q8KRow::quantize(&xf);
                assert_eq!(xv, xs, "{t:?} width {width} act {i}: AVX2 quantiser differs from scalar");
                let v = dot_q8k(t, &wrow, &xv);
                let s = dot_q8k_scalar(t, &wrow, &xs);
                // random f16 scale bytes include inf / NaN patterns: NaN == NaN counts as identical
                assert!(v.to_bits() == s.to_bits() || (v.is_nan() && s.is_nan()), "{t:?} width {width} row {i}: avx2 {v} vs scalar {s}");
                if !s.is_finite() {
                    non_finite += 1;
                } else if s != 0.0 {
                    max_rel = max_rel.max(((v - s) / s).abs() as f64);
                }
                n += 1;
            }
        }
    }
    println!("avx2 vs scalar: {n} rows identical, max diff 0 (max rel {max_rel:e}; {non_finite} rows with inf/NaN scale bytes)");
}

// ------------------------------------------------------------------------------------ (d) matvec forms

#[test]
fn matvec_forms_are_thread_invariant_and_agree_with_single_rows() {
    let mut rng = Rng(0x0BAD_F00D_0000_0007);
    let (rows, cols) = (301usize, 5120usize);
    for &t in &[GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K, GgmlType::Q8_0] {
        let w1 = WeightMat::new(t, rows, cols, finite_weights(t, rows, cols, &mut rng));
        let w2 = WeightMat::new(t, rows, cols, finite_weights(t, rows, cols, &mut rng));
        let xs: Vec<Vec<f32>> = (0..3).map(|i| rng.row(cols, [1.0, 0.01, 100.0][i])).collect();
        let acts: Vec<ActVec<'_>> = xs.iter().map(|x| ActVec::new(x)).collect();
        // thread invariance for every activation form the type accepts (Q8_0 default; Q8_K opt-in for K-quants)
        let forms: Vec<Act<'_>> = if common::is_kquant(t) { vec![Act::Q8(acts[0].q8()), Act::Q8K(acts[0].q8k())] } else { vec![Act::Q8(acts[0].q8())] };
        for form in &forms {
            let mut y_ref = vec![0f32; rows];
            matvec(&w1, *form, &mut y_ref, 1);
            for &threads in &[2usize, 3, 5, 7, 12, 64] {
                let mut y = vec![0f32; rows];
                matvec(&w1, *form, &mut y, threads);
                assert!(y.iter().zip(&y_ref).all(|(a, b)| a.to_bits() == b.to_bits()), "{t:?}: matvec differs at {threads} threads");
            }
        }
        // reference for the fused / batched forms: one thread, the default form
        let mut y_ref = vec![0f32; rows];
        matvec(&w1, acts[0].act_for(t), &mut y_ref, 1);
        // fused
        let mut g = vec![0f32; rows];
        let mut u = vec![0f32; rows];
        matvec2(&w1, &w2, acts[0].act_for(t), acts[0].act_for(t), &mut g, &mut u, 12);
        let mut u_ref = vec![0f32; rows];
        matvec(&w2, acts[0].act_for(t), &mut u_ref, 1);
        assert!(g.iter().zip(&y_ref).all(|(a, b)| a.to_bits() == b.to_bits()), "{t:?}: matvec2 gate differs");
        assert!(u.iter().zip(&u_ref).all(|(a, b)| a.to_bits() == b.to_bits()), "{t:?}: matvec2 up differs");
        // batched
        let batch: Vec<Act<'_>> = acts.iter().map(|a| a.act_for(t)).collect();
        let mut yb = vec![0f32; 3 * rows];
        matmul(&w1, &batch, &mut yb, 12);
        for (ti, a) in acts.iter().enumerate() {
            let mut y = vec![0f32; rows];
            matvec(&w1, a.act_for(t), &mut y, 5);
            assert!(yb[ti * rows..(ti + 1) * rows].iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()), "{t:?}: matmul token {ti} differs");
        }
        println!("{t:?}: matvec invariant over 1..64 threads; matvec2 and matmul bit-exact with matvec");
    }
}
