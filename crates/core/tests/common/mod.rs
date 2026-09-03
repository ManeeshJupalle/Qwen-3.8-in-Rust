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

/// The primary GGUF (bartowski Q4_K_M). Tests that need it fail loudly if it is absent.
pub fn gguf_path() -> PathBuf {
    let p = root().join("models").join("Qwen3.8-27B-GGUF-bartowski").join("Qwen3.8-27B-Q4_K_M.gguf");
    assert!(
        p.exists(),
        "primary GGUF missing at {}\nrun: hf download bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf --local-dir models/Qwen3.8-27B-GGUF-bartowski",
        p.display()
    );
    p
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
