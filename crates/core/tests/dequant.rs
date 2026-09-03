//! 1.4: Rust scalar dequantisation must reproduce gguf-py's output bit-for-bit for every type in the file
//! (prefix blocks, the full smallest tensor via CRC-32 of the f32 stream, and the hostile hand-built blocks).

mod common;

use std::fs;

use aqueduct_core::quant::crc32;
use aqueduct_core::{dequantize, GgmlType, Gguf};

fn gtype(name: &str) -> GgmlType {
    match name {
        "F32" => GgmlType::F32,
        "Q4_0" => GgmlType::Q4_0,
        "Q8_0" => GgmlType::Q8_0,
        "Q4_K" => GgmlType::Q4_K,
        "Q5_K" => GgmlType::Q5_K,
        "Q6_K" => GgmlType::Q6_K,
        other => panic!("unexpected type in manifest: {other}"),
    }
}

fn dq_dir() -> std::path::PathBuf {
    common::fixture("dequant")
}

fn assert_bit_exact(label: &str, got: &[f32], want: &[f32]) {
    let (n, max, first) = common::compare_bits(got, want);
    if n > 0 {
        let i = first.unwrap();
        panic!("{label}: {n} of {} values differ (max abs diff {max:e}); first at {i}: got {} ({:#010x}) want {} ({:#010x})", got.len(), got[i], got[i].to_bits(), want[i], want[i].to_bits());
    }
}

#[test]
fn prefix_blocks_bit_exact() {
    let m = common::json(&dq_dir().join("manifest.json"));
    let mut done = Vec::new();
    for (name, e) in m["types"].as_object().unwrap() {
        let t = gtype(name);
        let p = &e["prefix"];
        let raw = fs::read(dq_dir().join(p["raw_bin"].as_str().unwrap())).unwrap();
        let want = common::read_npy_f32(&dq_dir().join(p["npy"].as_str().unwrap()));
        let n = p["n_elements"].as_u64().unwrap() as usize;
        assert_eq!(want.len(), n);
        let got = dequantize(t, &raw, n).unwrap();
        assert_bit_exact(&format!("{name} prefix ({})", p["tensor"]), &got, &want);
        done.push(name.clone());
    }
    println!("prefix bit-exact: {done:?}");
    assert_eq!(done.len(), 6);
}

#[test]
fn full_smallest_tensor_bit_exact_via_digest() {
    let m = common::json(&dq_dir().join("manifest.json"));
    let head_n = m["head_values"].as_u64().unwrap() as usize;
    let mut g: Option<Gguf> = None;
    let mut report = Vec::new();
    for (name, e) in m["types"].as_object().unwrap() {
        let t = gtype(name);
        let f = &e["full"];
        let tensor = f["tensor"].as_str().unwrap();
        let n = f["n_elements"].as_u64().unwrap() as usize;
        let byte_size = f["byte_size"].as_u64().unwrap();
        let raw = match f["raw_bin"].as_str() {
            Some(bin) => fs::read(dq_dir().join(bin)).unwrap(),
            None => {
                // too large for the repo: read it from the GGUF through the reader under test
                let gg = g.get_or_insert_with(|| Gguf::open(common::gguf_path()).expect("open"));
                let info = gg.tensor(tensor).unwrap();
                assert_eq!(info.byte_size, byte_size);
                assert_eq!(info.absolute_file_offset, f["abs_offset"].as_u64().unwrap());
                gg.read_raw(tensor).unwrap()
            }
        };
        assert_eq!(raw.len() as u64, byte_size, "{name}: raw size");
        let got = dequantize(t, &raw, n).unwrap();
        let head = common::read_npy_f32(&dq_dir().join(f["head_npy"].as_str().unwrap()));
        assert_bit_exact(&format!("{name} full head ({tensor})"), &got[..head_n.min(n)], &head);
        let mut bytes = Vec::with_capacity(n * 4);
        for v in &got {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let crc = crc32(&bytes);
        let want_crc = f["digest"]["crc32"].as_u64().unwrap() as u32;
        let max_abs = got.iter().fold(0f32, |a, v| a.max(v.abs()));
        let want_max = f["digest"]["max_abs"].as_f64().unwrap() as f32;
        assert_eq!(crc, want_crc, "{name} full ({tensor}, {n} values): CRC-32 of f32 stream differs -> not bit-exact");
        assert_eq!(max_abs, want_max, "{name} full: max|x|");
        report.push(format!("{name}: {tensor} {n} values crc32={crc:#010x} ok"));
    }
    println!("{}", report.join("\n"));
    assert_eq!(report.len(), 6);
}

#[test]
fn hostile_blocks_bit_exact() {
    let m = common::json(&dq_dir().join("manifest.json"));
    let cases = m["hostile"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for h in cases {
        let label = h["label"].as_str().unwrap();
        let t = gtype(h["ggml_type"].as_str().unwrap());
        let raw = fs::read(dq_dir().join(h["raw_bin"].as_str().unwrap())).unwrap();
        let want = common::read_npy_f32(&dq_dir().join(h["npy"].as_str().unwrap()));
        let n = h["n_elements"].as_u64().unwrap() as usize;
        let got = dequantize(t, &raw, n).unwrap();
        assert_bit_exact(&format!("hostile {label}: {}", h["description"]), &got, &want);
        // the hostile blocks must actually exercise distinct fields: no two sub-block scales equal
        if matches!(t, GgmlType::Q4_K | GgmlType::Q5_K) {
            let mut seen = std::collections::HashSet::new();
            for j in 0..8 {
                let (sc, mn) = aqueduct_core::quant::get_scale_min_k4(j, &raw[4..16]);
                assert!(seen.insert(sc) && seen.insert(mn), "hostile {label}: scale/min fields not distinct");
            }
        }
        println!("hostile {label}: {} values ok, first {:?}", n, &got[..4.min(n)]);
    }
}

#[test]
fn wrong_unpack_is_caught_by_hostile_q4_k() {
    // Sanity: a deliberately wrong unpack (swapping scale and min) must NOT reproduce the fixture,
    // proving the hostile fixture has teeth.
    let m = common::json(&dq_dir().join("manifest.json"));
    let h = m["hostile"].as_array().unwrap().iter().find(|h| h["label"] == "q4_k").unwrap();
    let raw = fs::read(dq_dir().join(h["raw_bin"].as_str().unwrap())).unwrap();
    let want = common::read_npy_f32(&dq_dir().join(h["npy"].as_str().unwrap()));
    let d = aqueduct_core::quant::f16_to_f32(u16::from_le_bytes([raw[0], raw[1]]));
    let dmin = aqueduct_core::quant::f16_to_f32(u16::from_le_bytes([raw[2], raw[3]]));
    let mut wrong = vec![0f32; 256];
    let mut is = 0;
    let mut q = 16;
    let mut yo = 0;
    for _ in 0..4 {
        let (sc, mn) = aqueduct_core::quant::get_scale_min_k4(is, &raw[4..16]);
        let (d1, m1) = (d * mn as f32, dmin * sc as f32); // swapped on purpose
        let (sc, mn) = aqueduct_core::quant::get_scale_min_k4(is + 1, &raw[4..16]);
        let (d2, m2) = (d * mn as f32, dmin * sc as f32);
        for l in 0..32 {
            wrong[yo + l] = d1 * (raw[q + l] & 0xF) as f32 - m1;
            wrong[yo + 32 + l] = d2 * (raw[q + l] >> 4) as f32 - m2;
        }
        q += 32;
        is += 2;
        yo += 64;
    }
    let (n, _, _) = common::compare_bits(&wrong, &want);
    assert!(n > 200, "swapping scale/min should change most values, changed {n}");
}
