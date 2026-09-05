//! Phase 5.5: the blocked GEMM (`matvec::matmul_t`, `kdot::dot_q8k_t`) against the single-row kernels and
//! the Phase 2a / 3 fixtures. Bit-identity with the single-row kernels is a property these tests check, not
//! a contract (the contract is the frozen Q8_K ceilings, `tests/real_layers.rs`); the kernels are written to
//! have it, so any difference is a bug in the unpack.
//! (a) the scalar blocked kernels against the scalar single-row kernels on the dot fixture rows: every weight
//!     row of a case against every activation row of the case as one tile, bit-exact; and every one of those
//!     dots within the derived terms budget of the f64 reference (rule 5: `budget_k` from the manifest,
//!     the budget computed from the row, never typed; as `tests/q8k.rs`);
//! (b) AVX2 blocked against scalar blocked: bit-exact on the fixtures and on random weight bytes with random /
//!     hostile activation rows, for every tile size `1..=T_MAX` and both widths;
//! (c) `matmul` (the blocked path) against `matvec` per activation: bit-exact for every tile size, thread
//!     count and batch size, batches larger than the tile (tiling and row blocking) and than `ROW_BLOCK`
//!     rows, the Phase 3.3 per-row loop (tile 0) included; `matmul_fn` and `matmul_t` agree with `matmul`.

mod common;

use std::path::PathBuf;

use aqueduct_core::kernels::kdot::{dot_q8k, dot_q8k_scalar, dot_q8k_t, dot_q8k_t_scalar, T_MAX};
use aqueduct_core::kernels::matvec::{matmul, matmul_fn, matmul_t, matvec, set_matmul_threads, set_matmul_tile, Act, ActBuf, ActVec, WeightMat, ROW_BLOCK};
use aqueduct_core::kernels::q8k::Q8KRow;
use aqueduct_core::kernels::simd::{avx2_detected, force_scalar};
use aqueduct_core::{dequantize, GgmlType};

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

/// Random block bytes whose f16 scale fields are finite (as `tests/q8k.rs`): random bytes alone include
/// inf / NaN scales, whose NaN payloads may differ between paths.
fn finite_weights(t: GgmlType, rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
    let rb = row_bytes(t, cols);
    let (_, ts) = t.block_layout();
    let ts = ts as usize;
    let mut data = rng.bytes(rows * rb);
    let scale_offsets: &[usize] = match t {
        GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        _ => &[],
    };
    for blk in data.chunks_mut(ts) {
        for &off in scale_offsets {
            let bits = u16::from_le_bytes([blk[off], blk[off + 1]]);
            let exp = 12 + (bits >> 10) % 7;
            let fixed = (bits & 0x83FF) | (exp << 10);
            blk[off..off + 2].copy_from_slice(&fixed.to_le_bytes());
        }
    }
    data
}

/// Hostile activation rows for a width that is a multiple of 256 (as `tests/q8k.rs`).
fn hostile_acts(width: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut v = vec![rng.row(width, 1.0), rng.row(width, 1e30), rng.row(width, 1e-30), vec![3.0; width], vec![-0.0; width], (0..width).map(|i| if i % 2 == 0 { 5.0 } else { -5.0 }).collect()];
    let mut ties: Vec<f32> = (0..width).map(|i| (i % 9) as f32 - 4.0 + 0.5).collect();
    ties[0] = -127.0;
    ties[300 % width] = 127.0;
    v.push(ties);
    let mut pm: Vec<f32> = rng.row(width, 0.5);
    pm[7] = 4.0;
    pm[9] = -4.0;
    v.push(pm.clone());
    pm[7] = -4.0;
    pm[9] = 4.0;
    v.push(pm);
    let mut z = rng.row(width, 1.0);
    for x in &mut z[256..512.min(width)] {
        *x = 0.0;
    }
    v.push(z);
    v
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

// ------------------------------------------------------------------- (a) scalar blocked vs scalar single-row, and the f64 reference

#[test]
fn blocked_scalar_equals_single_row_scalar_and_meets_the_terms_budget_on_fixtures() {
    let fx = Fx::load("dot");
    let k = fx.manifest["budget_k"].as_f64().unwrap();
    let mut n_dots = 0usize;
    for c in fx.cases() {
        let t = type_by_name(c["ggml_type"].as_str().unwrap());
        if !K_TYPES.contains(&t) {
            continue;
        }
        let width = c["width"].as_u64().unwrap() as usize;
        let n_rows = c["n_rows"].as_u64().unwrap() as usize;
        assert!(n_rows <= T_MAX);
        let w_raw = fx.raw(&c["w_raw"]);
        let x_deq = fx.arr(&c["x_deq"]);
        let rb = row_bytes(t, width);
        // every activation row of the case as one tile
        let xs: Vec<Q8KRow> = (0..n_rows).map(|r| Q8KRow::quantize_scalar(&x_deq[r * width..(r + 1) * width])).collect();
        let refs: Vec<&Q8KRow> = xs.iter().collect();
        let mut worst = 0f64;
        for r in 0..n_rows {
            let wrow = &w_raw[r * rb..(r + 1) * rb];
            let mut out = vec![0f32; n_rows];
            dot_q8k_t_scalar(t, wrow, &refs, &mut out);
            let wd = dequantize(t, wrow, width).unwrap();
            for (i, x) in xs.iter().enumerate() {
                let single = dot_q8k_scalar(t, wrow, x);
                assert_eq!(out[i].to_bits(), single.to_bits(), "{} row {r} act {i}: blocked {} vs single-row {single}", c["case"], out[i]);
                // rule 5: the derived terms budget of the f64 reference (docs/kquant-dot.md), never typed
                let xd = x.dequantize();
                let refsum: f64 = wd.iter().zip(&xd).map(|(&w, &x)| w as f64 * x as f64).sum();
                let bud = common::kquant_terms_budget(t, wrow, &wd, &xd, k);
                let d = (out[i] as f64 - refsum).abs();
                assert!(d <= bud, "{} row {r} act {i}: got {} want {refsum} diff {d:e} > budget {bud:e}", c["case"], out[i]);
                if bud > 0.0 {
                    worst = worst.max(d / bud);
                }
                n_dots += 1;
            }
        }
        println!("{}: {n_rows} rows x {n_rows} activations blocked == single-row bit for bit; worst {:.4} of the terms budget", c["case"], worst);
    }
    println!("blocked scalar: {n_dots} fixture dots bit-identical to the single-row scalar kernels and within budget");
}

// ---------------------------------------------------------------------------------- (b) AVX2 blocked vs scalar blocked

#[test]
fn avx2_blocked_matches_scalar_blocked_bit_for_bit() {
    if !avx2_detected() {
        println!("AVX2 not available: skipped");
        return;
    }
    let fx = Fx::load("dot");
    let mut n = 0usize;
    // fixtures: every weight row against the case's activation rows, and against each tile size 1..=n_rows
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
        let xs: Vec<Q8KRow> = (0..n_rows).map(|r| Q8KRow::quantize_scalar(&x_deq[r * width..(r + 1) * width])).collect();
        for tile in 1..=n_rows {
            let refs: Vec<&Q8KRow> = xs.iter().take(tile).collect();
            for r in 0..n_rows {
                let wrow = &w_raw[r * rb..(r + 1) * rb];
                let mut v = vec![0f32; tile];
                let mut s = vec![0f32; tile];
                force_scalar(false);
                dot_q8k_t(t, wrow, &refs, &mut v);
                dot_q8k_t_scalar(t, wrow, &refs, &mut s);
                assert!(bits_equal(&v, &s), "{} row {r} tile {tile}: avx2 {v:?} vs scalar {s:?}", c["case"]);
                // and the AVX2 single-row kernel, the production decode kernel
                for (i, x) in refs.iter().enumerate() {
                    let one = dot_q8k(t, wrow, x);
                    assert_eq!(v[i].to_bits(), one.to_bits(), "{} row {r} tile {tile} act {i}: blocked {} vs single-row avx2 {one}", c["case"], v[i]);
                }
                n += tile;
            }
        }
    }
    // random weight bytes (every pattern is a valid block; finite scales) x random and hostile activations
    let mut rng = Rng(0x5EED_5500_0000_0001);
    for &t in &K_TYPES {
        for width in [256usize, 5120] {
            let mut acts = hostile_acts(width, &mut rng);
            while acts.len() < T_MAX + 3 {
                acts.push(rng.row(width, [1.0, 1e-3, 1e3][acts.len() % 3]));
            }
            let qs: Vec<Q8KRow> = acts.iter().map(|a| Q8KRow::quantize(a)).collect();
            for tile in 1..=T_MAX {
                for rep in 0..4 {
                    // a different subset of the hostile rows per repetition
                    let refs: Vec<&Q8KRow> = (0..tile).map(|i| &qs[(i + rep * 5) % qs.len()]).collect();
                    let wrow = finite_weights(t, 1, width, &mut rng);
                    let mut v = vec![0f32; tile];
                    let mut s = vec![0f32; tile];
                    force_scalar(false);
                    dot_q8k_t(t, &wrow, &refs, &mut v);
                    dot_q8k_t_scalar(t, &wrow, &refs, &mut s);
                    assert!(bits_equal(&v, &s), "{t:?} width {width} tile {tile} rep {rep}: avx2 {v:?} vs scalar {s:?}");
                    for (i, x) in refs.iter().enumerate() {
                        assert_eq!(v[i].to_bits(), dot_q8k_scalar(t, &wrow, x).to_bits(), "{t:?} width {width} tile {tile} act {i}: blocked vs single-row");
                    }
                    n += tile;
                }
            }
        }
    }
    force_scalar(false);
    println!("avx2 blocked vs scalar blocked (and vs the single-row kernels): {n} dots identical, every tile 1..={T_MAX}");
}

// ------------------------------------------------------------------------------ (c) matmul forms: tiles, threads, batches

#[test]
fn matmul_blocked_equals_matvec_for_every_tile_thread_count_and_batch() {
    let mut rng = Rng(0x0BAD_5500_0000_0007);
    // 301 rows: several row blocks per participant plus a remainder; 37 x 17408: the wide (ffn_down) shape
    let shapes: [(GgmlType, usize, usize); 4] = [(GgmlType::Q4_K, 301, 5120), (GgmlType::Q5_K, 301, 5120), (GgmlType::Q6_K, 301, 5120), (GgmlType::Q6_K, 37, 17408)];
    for &(t, rows, cols) in &shapes {
        let w = WeightMat::new(t, rows, cols, finite_weights(t, rows, cols, &mut rng));
        assert!(rows > ROW_BLOCK * 2 || cols > 5120, "the 5120-wide cases must span several row blocks");
        let n_max = 2 * T_MAX + 3; // more activations than a tile, an odd tail
        let xs: Vec<Vec<f32>> = (0..n_max).map(|i| rng.row(cols, [1.0, 0.01, 100.0][i % 3])).collect();
        let acts: Vec<ActVec<'_>> = xs.iter().map(|x| ActVec::new(x)).collect();
        let batch: Vec<Act<'_>> = acts.iter().map(|a| a.act_for(t)).collect();
        assert!(batch.iter().all(|a| matches!(a, Act::Q8K(_))), "production activations for K-quants are Q8_K");
        // the reference: matvec per activation, one thread
        let mut y_ref = vec![0f32; n_max * rows];
        for (i, a) in batch.iter().enumerate() {
            matvec(&w, *a, &mut y_ref[i * rows..(i + 1) * rows], 1);
        }
        for &tile in &[0usize, 1, 2, 3, 4, 8, 16] {
            set_matmul_tile(tile);
            for &n in &[1usize, 3, 4, 8, 9, 16, 17, n_max] {
                for &threads in &[1usize, 2, 5, 12] {
                    set_matmul_threads(threads); // pinned, so the thread count under test is the one used
                    let mut y = vec![0f32; n * rows];
                    matmul(&w, &batch[..n], &mut y, threads);
                    assert!(bits_equal(&y, &y_ref[..n * rows]), "{t:?} {rows}x{cols}: matmul tile {tile} n {n} threads {threads} differs from matvec");
                }
                set_matmul_threads(0);
                // the closure form (the verification batch's path) and the explicit blocked entry
                let bufs: Vec<ActBuf> = xs[..n]
                    .iter()
                    .map(|x| {
                        let mut b = ActBuf::with_capacity(cols);
                        b.fill(x, &[t]);
                        b
                    })
                    .collect();
                let mut yf = vec![0f32; n * rows];
                matmul_fn(&w, n, &|i| bufs[i].act_for(t, &xs[i]), &mut yf, 6);
                assert!(bits_equal(&yf, &y_ref[..n * rows]), "{t:?} {rows}x{cols}: matmul_fn tile {tile} n {n} differs from matvec");
                if tile > 0 {
                    let mut yt = vec![0f32; n * rows];
                    matmul_t(&w, &batch[..n], &mut yt, 6);
                    assert!(bits_equal(&yt, &y_ref[..n * rows]), "{t:?} {rows}x{cols}: matmul_t tile {tile} n {n} differs from matvec");
                }
            }
        }
        set_matmul_tile(aqueduct_core::kernels::matvec::DEFAULT_MATMUL_T);
        println!("{t:?} {rows}x{cols}: matmul / matmul_fn / matmul_t bit-exact with matvec for tiles 0..16, batches 1..{n_max}, 1..12 threads");
    }
    // the blocked entry refuses what it cannot block
    let w8 = WeightMat::new(GgmlType::Q8_0, 8, 256, rng.bytes(8 * row_bytes(GgmlType::Q8_0, 256)));
    let x = rng.row(256, 1.0);
    let a = ActVec::new(&x);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut y = vec![0f32; 8];
        matmul_t(&w8, &[a.act_for(GgmlType::Q8_0)], &mut y, 1);
    }));
    assert!(r.is_err(), "matmul_t must refuse Q8_0 weights");
}
