//! aqueduct CLI: `info <gguf>`, `tok encode|decode` (Phase 1), `bench kernels|membw` (Phase 2b / 3) and
//! `run` (Phase 3: fully resident greedy decode).

use std::collections::BTreeMap;
use std::process::ExitCode;
use std::time::Instant;

use aqueduct_core::{layer_of, Gguf, ModelConfig, Tok};

mod bench;
mod rss;
mod run;

const DEFAULT_TOKENIZER: &str = "models/Qwen3.8-27B/tokenizer.json";

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  aqueduct info <model.gguf>\n  aqueduct tok [--tokenizer <tokenizer.json>] encode <text>\n  aqueduct tok [--tokenizer <tokenizer.json>] decode <id,id,...>\n  aqueduct bench kernels [--threads N] [--rows R] [--cols C] [--reps N]
  aqueduct bench membw [--gib 2] [--runs 5] [--threads N]
  aqueduct run [--model <gguf>] [--tokenizer <json>] [--threads N] [--max-tokens N] [--ids-only] (--ids <csv> | --prompt <text>)"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") if args.len() == 2 => match info(&args[1]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("bench") if args.get(1).map(String::as_str) == Some("membw") => match bench::membw(&args[2..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("bench") if args.get(1).map(String::as_str) == Some("kernels") => match bench::kernels(&args[2..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("run") => match run::parse(&args[1..], DEFAULT_TOKENIZER).and_then(run::run) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Some("tok") => match tok(&args[1..]) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => usage(),
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        _ => usage(),
    }
}

fn info(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let g = Gguf::open(path)?;
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let cfg = ModelConfig::from_gguf(&g)?;

    println!("file: {}", g.path().display());
    println!("file size bytes: {}", g.file_size());
    println!("gguf version: {}   alignment: {}   header end: {}   data offset: {}", g.version(), g.alignment(), g.header_end(), g.data_offset());
    println!("open: {open_ms:.1} ms, bytes read on open: {}, tensor bytes read: {}", g.bytes_read_on_open(), g.tensor_bytes_read());
    println!("metadata keys: {}", g.kv().len());
    println!();
    println!("config (from GGUF metadata only):");
    print!("{}", cfg.describe());
    println!();

    let tensors = g.tensors();
    let total_bytes: u64 = tensors.iter().map(|t| t.byte_size).sum();
    println!("total tensors: {}", tensors.len());
    println!("total tensor bytes: {}  ({:.3} GiB)", total_bytes, total_bytes as f64 / (1u64 << 30) as f64);
    println!("header+index bytes (file - tensor bytes): {}", g.file_size() - total_bytes);
    println!();
    println!("bytes per ggml type:");
    let mut by_type: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for t in tensors {
        let e = by_type.entry(t.ggml_type.name()).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.byte_size;
    }
    let mut rows: Vec<_> = by_type.into_iter().collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1 .1));
    for (name, (n, b)) in rows {
        println!("  {name:<10} tensors={n:5} bytes={b:14} ({:8.3} GiB, {:5.1}%)", b as f64 / (1u64 << 30) as f64, 100.0 * b as f64 / total_bytes as f64);
    }
    println!();

    let layers = g.layer_ids();
    println!("layers found: {} (indices {}..{}, 0-based as written in tensor names)", layers.len(), layers.first().copied().unwrap_or(0), layers.last().copied().unwrap_or(0));
    let mut all = true;
    let mut largest = (0u32, 0u64);
    let mut padding_only = 0;
    for &l in &layers {
        let s = g.layer_span(l).unwrap();
        let span = s.end - s.start;
        if !s.contiguous {
            all = false;
        } else if s.gap_bytes > 0 {
            padding_only += 1;
        }
        if span > largest.1 {
            largest = (l, span);
        }
        println!(
            "layer {l:2}: tensors={:2} min_offset={} max_end={} sum_sizes={} span={} contiguous: {} (gap bytes={}, max single gap={}, adjacent in file order={}, foreign tensors inside span={})",
            s.tensor_count,
            s.start,
            s.end,
            s.sum_bytes,
            span,
            if s.contiguous { "yes" } else { "no" },
            s.gap_bytes,
            s.max_single_gap,
            if s.adjacent_in_file_order { "yes" } else { "no" },
            s.foreign_inside
        );
    }
    println!();
    println!("ALL LAYERS CONTIGUOUS: {}  (layers with padding-only gaps: {padding_only})", if all { "yes" } else { "no" });
    println!("LARGEST LAYER (by span, sizes the ring slot): layer {}, {} bytes = {:.2} MiB", largest.0, largest.1, largest.1 as f64 / 1048576.0);
    println!("largest STREAMED layer (excluding MTP blocks): layer {}, {} bytes = {:.2} MiB", cfg.largest_streamed_layer, cfg.ring_slot_bytes, cfg.ring_slot_bytes as f64 / 1048576.0);
    println!();
    println!("non-layer tensors (embed, output, norms, anything else):");
    let mut nl_total = 0u64;
    for t in g.non_layer_tensors() {
        nl_total += t.byte_size;
        println!("  {:<40} {:<8} shape={:?} off={} size={} ({:.2} MiB)", t.name, t.ggml_type.name(), t.shape_numpy_order(), t.absolute_file_offset, t.byte_size, t.byte_size as f64 / 1048576.0);
    }
    println!("  non-layer total bytes: {nl_total} ({:.2} MiB)", nl_total as f64 / 1048576.0);
    println!("pinned set (non-layer + MTP blocks {:?}): {} bytes ({:.2} MiB)", cfg.mtp_layers, cfg.pinned_bytes, cfg.pinned_bytes as f64 / 1048576.0);
    println!();
    let vision = tensors.iter().any(|t| t.name.starts_with("v.") || t.name.starts_with("mm."));
    println!("vision tower present in GGUF: {}", if vision { "YES" } else { "no" });
    println!("output.weight present: {}; token_embd.weight present: {}", yn(g.has_tensor("output.weight")), yn(g.has_tensor("token_embd.weight")));
    println!("embeddings tied (no separate output.weight): {}", if cfg.tie_word_embeddings { "YES" } else { "no" });
    let mtp: Vec<&str> = tensors.iter().filter(|t| t.name.contains(".nextn.")).map(|t| t.name.as_str()).collect();
    println!("MTP tensors present in GGUF: {}  ({:?})", if mtp.is_empty() { "NO" } else { "YES" }, mtp);
    let other: Vec<&str> = tensors
        .iter()
        .filter(|t| layer_of(&t.name).is_none() && !matches!(t.name.as_str(), "token_embd.weight" | "output.weight" | "output_norm.weight"))
        .map(|t| t.name.as_str())
        .collect();
    println!("other non-layer tensors not in (token_embd, output, output_norm): {other:?}");
    Ok(())
}

fn yn(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

fn tok(args: &[String]) -> Result<bool, Box<dyn std::error::Error>> {
    let mut path = DEFAULT_TOKENIZER.to_string();
    let mut rest: Vec<&String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--tokenizer" && i + 1 < args.len() {
            path = args[i + 1].clone();
            i += 2;
        } else {
            rest.push(&args[i]);
            i += 1;
        }
    }
    if rest.len() != 2 {
        return Ok(false);
    }
    let t = Tok::from_file(&path)?;
    match rest[0].as_str() {
        "encode" => {
            let ids = t.encode(rest[1])?;
            println!("{}", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            Ok(true)
        }
        "decode" => {
            let ids: Result<Vec<u32>, _> = rest[1].split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse::<u32>()).collect();
            let ids = ids?;
            println!("{}", t.decode(&ids)?);
            Ok(true)
        }
        _ => Ok(false),
    }
}
