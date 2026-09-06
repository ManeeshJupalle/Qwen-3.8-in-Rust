//! 1.1: the tensor index of the primary GGUF must match tests/fixtures/gguf_index.json exactly,
//! open must take < 1 s and read zero tensor bytes, and the metadata must match gguf_metadata.json.

mod common;

use std::time::Instant;

use aqueduct_core::{Gguf, Value};

#[test]
fn index_matches_fixture_and_open_reads_no_tensor_bytes() {
    let Some(path) = common::gguf_if_present() else { return };
    let t0 = Instant::now();
    let g = Gguf::open(&path).expect("open");
    let open = t0.elapsed();
    let fx = common::json(&common::fixture("gguf_index.json"));

    // open cost and the byte counters
    println!("open: {:.1} ms, bytes read on open {}, header_end {}, data_offset {}", open.as_secs_f64() * 1000.0, g.bytes_read_on_open(), g.header_end(), g.data_offset());
    assert!(open.as_secs_f64() < 1.0, "open took {:?}", open);
    assert_eq!(g.tensor_bytes_read(), 0, "tensor bytes read during open");
    assert!(g.bytes_read_on_open() <= g.data_offset(), "open read {} bytes, data section starts at {}", g.bytes_read_on_open(), g.data_offset());
    assert!(g.bytes_read_on_open() >= g.header_end());

    // file-level numbers
    assert_eq!(g.file_size(), fx["file_size"].as_u64().unwrap());
    assert_eq!(g.alignment() as u64, fx["alignment"].as_u64().unwrap());
    assert_eq!(g.data_offset(), fx["data_offset"].as_u64().unwrap());
    assert_eq!(g.tensors().len() as u64, fx["n_tensors"].as_u64().unwrap());

    // every tensor, in file order
    let fts = fx["tensors"].as_array().unwrap();
    assert_eq!(fts.len(), g.tensors().len());
    for (i, (t, f)) in g.tensors().iter().zip(fts).enumerate() {
        assert_eq!(t.name, f["name"].as_str().unwrap(), "tensor #{i} name");
        assert_eq!(t.ggml_type.name(), f["ggml_type"].as_str().unwrap(), "{}: type", t.name);
        assert_eq!(t.ggml_type.id() as u64, f["ggml_type_id"].as_u64().unwrap(), "{}: type id", t.name);
        let shape: Vec<u64> = f["shape_gguf_order"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        assert_eq!(t.shape, shape, "{}: shape (gguf order)", t.name);
        let np: Vec<u64> = f["shape_numpy_order"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        assert_eq!(t.shape_numpy_order(), np, "{}: shape (numpy order)", t.name);
        assert_eq!(t.element_count, f["n_elements"].as_u64().unwrap(), "{}: elements", t.name);
        assert_eq!(t.rel_offset, f["rel_offset"].as_u64().unwrap(), "{}: rel offset", t.name);
        assert_eq!(t.absolute_file_offset, f["byte_offset"].as_u64().unwrap(), "{}: abs offset", t.name);
        assert_eq!(t.byte_size, f["byte_size"].as_u64().unwrap(), "{}: byte size", t.name);
        assert_eq!(t.end_offset(), f["byte_end"].as_u64().unwrap(), "{}: end", t.name);
        assert_eq!(g.tensor(&t.name).unwrap().name, t.name);
    }
}

fn value_matches(key: &str, v: &Value, j: &serde_json::Value) {
    match v {
        Value::U8(x) => assert_eq!(*x as u64, j.as_u64().unwrap(), "{key}"),
        Value::I8(x) => assert_eq!(*x as i64, j.as_i64().unwrap(), "{key}"),
        Value::U16(x) => assert_eq!(*x as u64, j.as_u64().unwrap(), "{key}"),
        Value::I16(x) => assert_eq!(*x as i64, j.as_i64().unwrap(), "{key}"),
        Value::U32(x) => assert_eq!(*x as u64, j.as_u64().unwrap(), "{key}"),
        Value::I32(x) => assert_eq!(*x as i64, j.as_i64().unwrap(), "{key}"),
        Value::U64(x) => assert_eq!(*x, j.as_u64().unwrap(), "{key}"),
        Value::I64(x) => assert_eq!(*x, j.as_i64().unwrap(), "{key}"),
        Value::F32(x) => assert_eq!(*x as f64, j.as_f64().unwrap(), "{key}"),
        Value::F64(x) => assert_eq!(*x, j.as_f64().unwrap(), "{key}"),
        Value::Bool(x) => assert_eq!(*x, j.as_bool().unwrap(), "{key}"),
        Value::Str(s) => assert_eq!(s, j.as_str().unwrap(), "{key}"),
        Value::Array { items, .. } => {
            let arr = j.as_array().unwrap();
            assert_eq!(items.len(), arr.len(), "{key}: array length");
            for (i, (a, b)) in items.iter().zip(arr).enumerate() {
                value_matches(&format!("{key}[{i}]"), a, b);
            }
        }
    }
}

#[test]
fn metadata_matches_fixture() {
    let Some(gguf) = common::gguf_if_present() else { return };
    let g = Gguf::open(gguf).expect("open");
    let fx = common::json(&common::fixture("gguf_metadata.json"));
    let obj = fx.as_object().unwrap();
    // the python fixture adds three synthetic GGUF.* keys for the header fields
    assert_eq!(obj["GGUF.version"]["value"].as_u64().unwrap(), g.version() as u64);
    assert_eq!(obj["GGUF.tensor_count"]["value"].as_u64().unwrap(), g.tensors().len() as u64);
    assert_eq!(obj["GGUF.kv_count"]["value"].as_u64().unwrap(), g.kv().len() as u64);
    let mut checked = 0;
    for (key, v) in g.kv() {
        let f = obj.get(key).unwrap_or_else(|| panic!("key {key} not in fixture"));
        let types: Vec<&str> = f["types"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
        assert_eq!(types[0], v.type_name(), "{key}: type");
        if let Value::Array { elem_type, .. } = v {
            assert_eq!(types[1], elem_type.name(), "{key}: element type");
        }
        value_matches(key, v, &f["value"]);
        checked += 1;
    }
    assert_eq!(checked + 3, obj.len(), "fixture has keys the reader did not produce");
    // typed getters and their error paths
    assert_eq!(g.get_str("general.architecture").unwrap(), "qwen35");
    let e = g.get_u32("general.architecture").unwrap_err().to_string();
    assert!(e.contains("general.architecture") && e.contains("STRING"), "{e}");
    let e = g.get_u32("no.such.key").unwrap_err().to_string();
    assert!(e.contains("no.such.key"), "{e}");
}

#[test]
fn layer_spans_are_contiguous_and_read_raw_counts_bytes() {
    let Some(gguf) = common::gguf_if_present() else { return };
    let g = Gguf::open(gguf).expect("open");
    let layers = g.layer_ids();
    assert_eq!(layers.len(), 65);
    assert_eq!(layers.first(), Some(&0));
    assert_eq!(layers.last(), Some(&64));
    for &l in &layers {
        let s = g.layer_span(l).unwrap();
        assert!(s.contiguous, "layer {l} not contiguous: {s:?}");
        assert_eq!(s.gap_bytes, 0, "layer {l}");
        assert_eq!(s.foreign_inside, 0, "layer {l}");
        assert_eq!(s.end - s.start, s.sum_bytes, "layer {l}");
    }
    assert!(g.layer_span(65).is_none());
    // read the smallest tensor; the counter must grow by exactly its size and the offset must be aligned
    let t = g.tensors().iter().min_by_key(|t| t.byte_size).unwrap().clone();
    let raw = g.read_raw(&t.name).unwrap();
    assert_eq!(raw.len() as u64, t.byte_size);
    assert_eq!(g.tensor_bytes_read(), t.byte_size);
    assert_eq!(t.absolute_file_offset % 32, 0);
    let mut buf = vec![0u8; t.byte_size as usize];
    g.read_into(&t.name, &mut buf).unwrap();
    assert_eq!(buf, raw);
    assert_eq!(g.tensor_bytes_read(), 2 * t.byte_size);
    let mut wrong = vec![0u8; t.byte_size as usize + 1];
    assert!(g.read_into(&t.name, &mut wrong).is_err());
    assert!(g.read_raw("no.such.tensor").is_err());
}
