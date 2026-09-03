//! 1.5: the numbers `aqueduct info` derives (per-type bytes, per-layer spans and contiguity, largest layer,
//! non-layer total) must match Phase 0's docs/data/gguf_layout.txt line for line.

mod common;

use std::collections::BTreeMap;

use aqueduct_core::{Gguf, ModelConfig};

/// The first unsigned integer after `key` in `line` (spaces after the key are allowed: the report pads columns).
fn field(line: &str, key: &str) -> u64 {
    let start = line.find(key).unwrap_or_else(|| panic!("{key} not in line: {line}")) + key.len();
    let rest = line[start..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().unwrap_or_else(|_| panic!("no number after {key} in: {line}"))
}

#[test]
fn info_numbers_match_phase0_layout_report() {
    let text = std::fs::read_to_string(common::root().join("docs").join("data").join("gguf_layout.txt")).expect("docs/data/gguf_layout.txt");
    let g = Gguf::open(common::gguf_path()).expect("open");
    let cfg = ModelConfig::from_gguf(&g).expect("config");
    let tensors = g.tensors();
    let total: u64 = tensors.iter().map(|t| t.byte_size).sum();

    let mut by_type: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for t in tensors {
        let e = by_type.entry(t.ggml_type.name()).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.byte_size;
    }

    let mut checked = 0;
    let mut layer_lines = 0;
    for line in text.lines() {
        let l = line.trim_end();
        if let Some(rest) = l.strip_prefix("file size bytes: ") {
            assert_eq!(rest.parse::<u64>().unwrap(), g.file_size());
            checked += 1;
        } else if let Some(rest) = l.strip_prefix("data section starts at byte: ") {
            assert_eq!(rest.parse::<u64>().unwrap(), g.data_offset());
            checked += 1;
        } else if let Some(rest) = l.strip_prefix("total tensors: ") {
            assert_eq!(rest.parse::<usize>().unwrap(), tensors.len());
            checked += 1;
        } else if l.starts_with("total tensor bytes: ") {
            assert_eq!(field(l, "total tensor bytes: "), total);
            checked += 1;
        } else if l.starts_with("  ") && l.contains("tensors=") && l.contains("bytes=") && !l.contains("min_offset") {
            let name = l.split_whitespace().next().unwrap();
            let (n, b) = by_type.get(name).unwrap_or_else(|| panic!("type {name} from the report is not in the file"));
            assert_eq!(field(l, "tensors="), *n, "{name} tensor count");
            assert_eq!(field(l, "bytes="), *b, "{name} bytes");
            checked += 1;
        } else if l.starts_with("layer ") && l.contains("min_offset=") {
            let layer: u32 = l[6..].trim_start().split(':').next().unwrap().parse().unwrap();
            let s = g.layer_span(layer).unwrap();
            assert_eq!(field(l, "tensors="), s.tensor_count as u64, "layer {layer} count");
            assert_eq!(field(l, "min_offset="), s.start, "layer {layer} start");
            assert_eq!(field(l, "max_end="), s.end, "layer {layer} end");
            assert_eq!(field(l, "sum_sizes="), s.sum_bytes, "layer {layer} sum");
            assert_eq!(field(l, "span="), s.end - s.start, "layer {layer} span");
            assert_eq!(field(l, "gap bytes="), s.gap_bytes, "layer {layer} gap");
            assert_eq!(field(l, "foreign tensors inside span="), s.foreign_inside as u64, "layer {layer} foreign");
            let contiguous = l.contains("contiguous: yes");
            assert_eq!(contiguous, s.contiguous, "layer {layer} contiguity");
            layer_lines += 1;
        } else if l.starts_with("ALL LAYERS CONTIGUOUS: ") {
            let all = g.layer_ids().iter().all(|&i| g.layer_span(i).unwrap().contiguous);
            assert_eq!(l.contains("CONTIGUOUS: yes"), all);
            checked += 1;
        } else if l.starts_with("LARGEST LAYER (by span") {
            let tail = &l[l.find("): layer ").expect("largest-layer line format")..];
            let layer = field(tail, "layer ");
            let bytes = field(tail, ", ");
            // first layer with the maximum span, as tools/gguf_index.py picks it
            let mut largest = (0u32, 0u64);
            for i in g.layer_ids() {
                let s = g.layer_span(i).unwrap();
                if s.end - s.start > largest.1 {
                    largest = (i, s.end - s.start);
                }
            }
            let (largest, span) = largest;
            assert_eq!(layer, largest as u64);
            assert_eq!(bytes, span);
            // the report's largest block is the MTP block only if it beats every streamed layer
            assert!(cfg.ring_slot_bytes <= span);
            checked += 1;
        } else if l.starts_with("  non-layer total bytes: ") {
            let nl: u64 = g.non_layer_tensors().iter().map(|t| t.byte_size).sum();
            assert_eq!(field(l, "non-layer total bytes: "), nl);
            assert_eq!(cfg.pinned_bytes, nl + cfg.mtp_layers.iter().map(|&m| cfg.layer_bytes[m as usize]).sum::<u64>());
            checked += 1;
        }
    }
    assert_eq!(layer_lines, 65, "expected one line per layer in the report");
    assert!(checked >= 12, "only {checked} summary lines matched");
    println!("matched {layer_lines} layer lines and {checked} summary lines of gguf_layout.txt");
}
