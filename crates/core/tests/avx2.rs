//! 2b.3: the AVX2 kernels against the scalar reference: on every kernel fixture and on random / hostile rows.
//! The contract (kernels/avx2.rs) is bit-identity, so every comparison here is exact; the reported "max diff"
//! is expected to be 0. Skips (with a message) on a CPU without AVX2 + F16C.

mod common;

use std::path::PathBuf;

use aqueduct_core::kernels::dot::{dot_q8, dot_q8_scalar};
use aqueduct_core::kernels::matvec::{matvec, Act, WeightMat};
use aqueduct_core::kernels::q8::Q8Row;
use aqueduct_core::kernels::rmsnorm::{rmsnorm, rmsnorm_scalar};
use aqueduct_core::kernels::simd::{avx2_detected, force_scalar, use_avx2};
use aqueduct_core::kernels::softmax::{softmax, softmax_scalar};
use aqueduct_core::GgmlType;

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

fn avx2_or_skip() -> bool {
    if !avx2_detected() {
        println!("AVX2 + F16C not available on this CPU: AVX2 tests skipped");
        return false;
    }
    force_scalar(false);
    assert!(use_avx2());
    true
}

/// xorshift64*: deterministic random bytes / floats without a dependency.
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
    /// Roughly Gaussian f32 with the given scale (sum of 4 uniforms, centered).
    fn f32(&mut self, scale: f32) -> f32 {
        let u: f32 = (0..4).map(|_| (self.next() >> 40) as f32 / (1u64 << 24) as f32).sum::<f32>() - 2.0;
        u * scale
    }
    fn row(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(scale)).collect()
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

const QUANT_TYPES: [GgmlType; 5] = [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K];

#[test]
fn avx2_dot_kernels_match_scalar_on_fixtures() {
    if !avx2_or_skip() {
        return;
    }
    let fx = Fx::load("dot");
    let mut n = 0;
    for c in fx.cases() {
        let t = GgmlType::from_id(c["ggml_type_id"].as_u64().unwrap() as u32, "fixture").unwrap();
        if t == GgmlType::F32 {
            continue;
        }
        let width = c["width"].as_u64().unwrap() as usize;
        let n_rows = c["n_rows"].as_u64().unwrap() as usize;
        let w_raw = fx.raw(&c["w_raw"]);
        let x_raw = fx.raw(&c["x_raw"]);
        let (bs, ts) = t.block_layout();
        let wrb = (width as u64 / bs * ts) as usize;
        let xrb = width / 32 * 34;
        for r in 0..n_rows {
            let x = Q8Row::from_ggml_bytes(&x_raw[r * xrb..(r + 1) * xrb], width);
            let w = &w_raw[r * wrb..(r + 1) * wrb];
            let a = dot_q8(t, w, &x);
            let s = dot_q8_scalar(t, w, &x);
            assert_eq!(a.to_bits(), s.to_bits(), "{} row {}: avx2 {a} vs scalar {s}", c["case"], r);
            n += 1;
        }
    }
    println!("dot fixtures: {n} rows bit-identical (max diff 0)");
}

#[test]
fn avx2_dot_kernels_match_scalar_on_random_rows() {
    if !avx2_or_skip() {
        return;
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for t in QUANT_TYPES {
        let (bs, ts) = t.block_layout();
        let mut n = 0;
        for i in 0..1000 {
            let width = match i % 4 {
                0 => 256,
                1 => 5120,
                2 => 1024,
                _ => 17408,
            };
            let wrb = (width as u64 / bs * ts) as usize;
            // random block bytes are valid for every ggml layout (random scales, mins, codes), including
            // f16 scales that are huge, tiny, denormal, or negative
            let mut w = rng.bytes(wrb);
            if i % 7 == 3 {
                // all-equal codes
                for b in w.iter_mut() {
                    *b = 0x77;
                }
            }
            let x = match i % 5 {
                0 => Q8Row::quantize(&rng.row(width, 1.0)),
                1 => Q8Row::quantize(&rng.row(width, 1e6)),
                2 => Q8Row::quantize(&rng.row(width, 1e-6)),
                3 => Q8Row::quantize(&vec![3.0f32; width]),
                _ => {
                    // extreme codes: -127 / 127 / 0 mixed
                    let v: Vec<f32> = (0..width).map(|j| [127.0, -127.0, 0.0, 1.0][j % 4]).collect();
                    Q8Row::quantize(&v)
                }
            };
            let a = dot_q8(t, &w, &x);
            let s = dot_q8_scalar(t, &w, &x);
            assert!(a.to_bits() == s.to_bits() || (a.is_nan() && s.is_nan()), "{t:?} random row {i} (width {width}): avx2 {a} vs scalar {s}");
            n += 1;
        }
        println!("{t:?}: {n} random rows bit-identical (max diff 0)");
    }
}

#[test]
fn avx2_q8_quantize_matches_scalar() {
    if !avx2_or_skip() {
        return;
    }
    let fx = Fx::load("q8_quantize");
    for c in fx.cases() {
        let width = c["width"].as_u64().unwrap() as usize;
        let x = fx.arr(&c["x"]);
        for r in 0..x.len() / width {
            let row = &x[r * width..(r + 1) * width];
            assert_eq!(Q8Row::quantize(row), Q8Row::quantize_scalar(row), "{} row {r}", c["case"]);
        }
    }
    let mut rng = Rng(42);
    let mut ties = 0;
    for i in 0..1000 {
        let width = [32, 5120, 64, 17408][i % 4];
        let mut row = match i % 6 {
            0 => rng.row(width, 1.0),
            1 => rng.row(width, 1e30),
            2 => rng.row(width, 1e-30),
            3 => {
                // exact ties: amax = 127 so d = 1, id = 1, and values k + 0.5 sit on the rounding boundary
                let mut v: Vec<f32> = (0..width).map(|j| (j % 254) as f32 - 126.5).collect();
                v[0] = 127.0;
                v
            }
            4 => (0..width).map(|j| if j % 3 == 0 { -0.0 } else { 1e-40 * (j as f32) }).collect(),
            _ => rng.row(width, 3.0),
        };
        if i % 6 == 3 {
            ties += 1;
        }
        if i % 11 == 5 {
            row[width / 2] = 1e5; // one outlier in a block
        }
        let a = Q8Row::quantize(&row);
        let s = Q8Row::quantize_scalar(&row);
        assert_eq!(a, s, "random q8 row {i} (width {width}) differs");
    }
    println!("q8 quantize: fixtures + 1000 random rows ({ties} with exact .5 ties) bit-identical");
}

#[test]
fn avx2_rmsnorm_and_softmax_match_scalar() {
    if !avx2_or_skip() {
        return;
    }
    let fx = Fx::load("rmsnorm");
    for c in fx.cases() {
        if c["kind"].as_str().unwrap() != "rmsnorm" {
            continue;
        }
        let width = c["width"].as_u64().unwrap() as usize;
        let eps = c["eps"].as_f64().unwrap() as f32;
        let x = fx.arr(&c["x"]);
        let w = fx.arr(&c["w_gguf"]);
        for r in 0..x.len() / width {
            let xr = &x[r * width..(r + 1) * width];
            let mut a = vec![0f32; width];
            let mut s = vec![0f32; width];
            rmsnorm(xr, &w, eps, &mut a);
            rmsnorm_scalar(xr, &w, eps, &mut s);
            assert_eq!(bits(&a), bits(&s), "{} row {r}", c["case"]);
        }
    }
    let fx = Fx::load("softmax");
    for c in fx.cases() {
        if c["kind"].as_str().unwrap() != "softmax" {
            continue;
        }
        let width = c["width"].as_u64().unwrap() as usize;
        let x = fx.arr(&c["x"]);
        for r in 0..x.len() / width {
            let xr = &x[r * width..(r + 1) * width];
            let mut a = vec![0f32; width];
            let mut s = vec![0f32; width];
            softmax(xr, &mut a);
            softmax_scalar(xr, &mut s);
            assert_eq!(bits(&a), bits(&s), "{} row {r}", c["case"]);
        }
    }
    let mut rng = Rng(7);
    for i in 0..1000 {
        let width = [5120, 100, 4096, 256, 13][i % 5];
        let x = rng.row(width, [1.0, 1e10, 1e-10, 30.0][i % 4]);
        let w = rng.row(width, 0.5);
        let mut a = vec![0f32; width];
        let mut s = vec![0f32; width];
        rmsnorm(&x, &w, 1e-6, &mut a);
        rmsnorm_scalar(&x, &w, 1e-6, &mut s);
        assert_eq!(bits(&a), bits(&s), "rmsnorm random row {i}");
        softmax(&x, &mut a);
        softmax_scalar(&x, &mut s);
        assert_eq!(bits(&a), bits(&s), "softmax random row {i}");
    }
    println!("rmsnorm and softmax: fixtures + 1000 random rows bit-identical");
}

#[test]
fn matvec_is_path_invariant() {
    if !avx2_or_skip() {
        return;
    }
    let fx = Fx::load("matvec");
    for c in fx.cases() {
        let t = GgmlType::from_id(c["ggml_type_id"].as_u64().unwrap() as u32, "fixture").unwrap();
        if t == GgmlType::F32 {
            continue;
        }
        let rows = c["n_rows"].as_u64().unwrap() as usize;
        let width = c["width"].as_u64().unwrap() as usize;
        let w = WeightMat::new(t, rows, width, fx.raw(&c["w_raw"]));
        let x = Q8Row::from_ggml_bytes(&fx.raw(&c["x_raw"]), width);
        let mut ya = vec![0f32; rows];
        let mut ys = vec![0f32; rows];
        force_scalar(false);
        matvec(&w, Act::Q8(&x), &mut ya, 4);
        force_scalar(true);
        matvec(&w, Act::Q8(&x), &mut ys, 4);
        force_scalar(false);
        assert_eq!(bits(&ya), bits(&ys), "{}: avx2 matvec differs from scalar", c["case"]);
    }
    println!("matvec: avx2 and scalar paths bit-identical on the fixtures");
}
