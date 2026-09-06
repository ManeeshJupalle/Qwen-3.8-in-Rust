//! Shared helpers for the Phase 1 integration tests: fixture paths, model paths, a minimal .npy reader.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

pub fn fixtures() -> PathBuf {
    root().join("tests").join("fixtures")
}

pub fn fixture(name: &str) -> PathBuf {
    fixtures().join(name)
}

/// The primary GGUF (bartowski Q4_K_M). It lives on the SSD at `C:\models` since Phase 3 (finding 34);
/// `AQUEDUCT_GGUF` overrides the path. Tests that need it fail loudly if it is absent.
pub const DEFAULT_GGUF: &str = r"C:\models\Qwen3.8-27B-Q4_K_M.gguf";

pub fn gguf_path() -> PathBuf {
    let p = std::env::var_os("AQUEDUCT_GGUF").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(DEFAULT_GGUF));
    assert!(
        p.exists(),
        "primary GGUF missing at {}\nrun: hf download bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf --local-dir C:/models (or set AQUEDUCT_GGUF)",
        p.display()
    );
    p
}

/// The primary GGUF if it is on this machine, else `None` after printing why the test is skipped (CI runs
/// the suite without the 17.8 GB file; the tests that read its header or a tensor skip there). With
/// `AQUEDUCT_REQUIRE_MODEL=1` in the environment a missing file fails instead, so a local run cannot skip
/// silently.
pub fn gguf_if_present() -> Option<PathBuf> {
    let p = std::env::var_os("AQUEDUCT_GGUF").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(DEFAULT_GGUF));
    if p.exists() {
        return Some(p);
    }
    if std::env::var_os("AQUEDUCT_REQUIRE_MODEL").is_some() {
        panic!("primary GGUF missing at {} and AQUEDUCT_REQUIRE_MODEL is set", p.display());
    }
    eprintln!("SKIPPED: primary GGUF missing at {} (set AQUEDUCT_GGUF, or AQUEDUCT_REQUIRE_MODEL=1 to make this a failure)", p.display());
    None
}

pub fn tokenizer_path() -> PathBuf {
    let p = root().join("models").join("Qwen3.8-27B").join("tokenizer.json");
    assert!(p.exists(), "tokenizer.json missing at {}\nrun: hf download Qwen/Qwen3.8-27B tokenizer.json --local-dir models/Qwen3.8-27B", p.display());
    p
}

pub fn json(path: &Path) -> serde_json::Value {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Look up a dotted JSON path like `text_config.rope_parameters.rope_theta`.
pub fn jpath<'a>(v: &'a serde_json::Value, path: &str) -> &'a serde_json::Value {
    let mut cur = v;
    for part in path.split('.') {
        cur = cur.get(part).unwrap_or_else(|| panic!("JSON path {path}: missing component {part}"));
    }
    cur
}

/// Minimal reader for little-endian float32 .npy files written by numpy (format 1.0 / 2.0, C order).
pub fn read_npy_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "{}: not an npy file", path.display());
    let major = bytes[6];
    let (header_len, data_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        _ => panic!("{}: unsupported npy version {major}", path.display()),
    };
    let header = std::str::from_utf8(&bytes[data_start..data_start + header_len]).unwrap();
    assert!(header.contains("'descr': '<f4'"), "{}: expected '<f4' descr, got header {header}", path.display());
    assert!(header.contains("'fortran_order': False"), "{}: fortran order not supported", path.display());
    let data = &bytes[data_start + header_len..];
    assert_eq!(data.len() % 4, 0);
    data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Exact comparison of two f32 slices by bit pattern; returns (mismatches, max abs diff, first mismatch index).
pub fn compare_bits(a: &[f32], b: &[f32]) -> (usize, f64, Option<usize>) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mut n = 0usize;
    let mut max = 0f64;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x.to_bits() != y.to_bits() {
            n += 1;
            if first.is_none() {
                first = Some(i);
            }
            let d = if x.is_finite() && y.is_finite() { (*x as f64 - *y as f64).abs() } else { f64::INFINITY };
            if d > max {
                max = d;
            }
        }
    }
    (n, max, first)
}

/// Phase 3 accuracy model of the K-quant kernels (docs/kquant-dot.md): their f32 lanes are partial sums that
/// cancel against each other, so the rounding error is bounded by the term magnitudes,
/// `k * eps * sum_j (|main_j| + |min_j|) |x_j|`, with `main`/`min` the two parts of each dequantised weight
/// (`w = main - min`, `min = dmin * m` for Q4_K / Q5_K, 0 for Q6_K), not by `sqrt(n) * max|term|`.
pub fn kquant_terms_budget(t: aqueduct_core::GgmlType, wrow: &[u8], w_deq: &[f32], x_deq: &[f32], k: f64) -> f64 {
    use aqueduct_core::GgmlType;
    let width = w_deq.len();
    assert_eq!(x_deq.len(), width);
    let nsb = width / 256;
    let bb = wrow.len() / nsb;
    let mut mags = 0f64;
    for sb in 0..nsb {
        let wb = &wrow[sb * bb..(sb + 1) * bb];
        let dmin = if t == GgmlType::Q6_K { 0.0 } else { half::f16::from_le_bytes([wb[2], wb[3]]).to_f64() };
        for s in 0..8 {
            let m = if t == GgmlType::Q6_K { 0 } else { aqueduct_core::quant::get_scale_min_k4(s, &wb[4..16]).1 };
            let min_part = dmin * m as f64;
            for j in sb * 256 + s * 32..sb * 256 + (s + 1) * 32 {
                mags += ((w_deq[j] as f64 + min_part).abs() + min_part.abs()) * (x_deq[j] as f64).abs();
            }
        }
    }
    k * f32::EPSILON as f64 * mags
}

pub fn is_kquant(t: aqueduct_core::GgmlType) -> bool {
    matches!(t, aqueduct_core::GgmlType::Q4_K | aqueduct_core::GgmlType::Q5_K | aqueduct_core::GgmlType::Q6_K)
}

/// The K-quant activation format of a model-scale test run (Phase 3.5): `AQUEDUCT_ACT=q8_0` (per-32 Q8_0
/// rows) or `AQUEDUCT_ACT=q8k` (ggml's Q8_K rows) overrides the production default. Sets the matvec flag and
/// returns the format's name and its frozen rule-5 ceilings fixture. Rule 5 as amended in 3.5: the ceilings
/// are per activation format, each frozen from that format's own measurement (`q8_ceilings.json` for Q8_0,
/// `q8k_ceilings.json` for Q8_K); a format without a frozen file runs ungated, as its measurement.
pub fn configure_act() -> (&'static str, PathBuf) {
    use aqueduct_core::kernels::matvec::{q8k_activations, set_q8_fine};
    match std::env::var("AQUEDUCT_ACT").as_deref() {
        Ok("q8k") => set_q8_fine(false),
        Ok("q8_0") => set_q8_fine(true),
        Ok(other) => panic!("AQUEDUCT_ACT={other}: expected q8_0 or q8k"),
        Err(_) => {}
    }
    if q8k_activations() {
        ("q8k", fixture("q8k_ceilings.json"))
    } else {
        ("q8_0", fixture("q8_ceilings.json"))
    }
}
