//! 2b.2: real-layer parity on the bartowski GGUF against tests/fixtures/ref_gguf (tools/ref_forward.py
//! --weights gguf: the same bytes dequantised with gguf-py and run through the HF modules in float32).
//!
//! The model is streamed one layer at a time (the GGUF sits on a slow disk and, dequantised, would not fit
//! in RAM), every prompt goes through the layer in two engine modes, and the residual stream after every
//! layer is compared at every position:
//!
//! - `f32`: weights dequantised in Rust (bit-identical to gguf-py, Phase 1 `tests/dequant.rs`) and used
//!   through the f32 dot path. Same numbers as the reference, so the derived f32 budget applies (rule 5):
//!   after layer L, `k * (L + 1) * sqrt(W) * eps * max|h_ref_L|` with `W` the longest reduction in a layer.
//! - `q8`: the engine's real path, quantised rows times Q8_0 activations. Quantising an activation block to
//!   8 bits adds up to `max|block| / 254` per element (about 2^-8 relative), 2^15 times f32 rounding, so
//!   this mode cannot meet an f32-derived budget by construction; it is MEASURED against the reference and
//!   against the f32 mode, and gated on the logits (argmax on every prompt, top-10 overlap, max|diff|).
//!
//! Incremental check: positions `0..T-1` are prefilled, the state is cloned, and the last token goes through
//! `forward_token` on the clone (the decode path); its output must equal the prefill's last position. Today
//! the prefill IS T sequential `forward_token` calls, so this holds bit-for-bit by construction; the check is
//! kept so a batched prefill (Phase 3) has to keep it true.
//!
//! `real_layers_0_7` runs two full 3:1 blocks; `real_layers_0_63_and_logits` (ignored: minutes, all 64
//! layers, the whole file read once) also does the final norm and lm_head and compares the logits with
//! `ref_gguf` and with llama.cpp (`tests/fixtures/ref_llamacpp`).
//!
//! Phase 3.5 (rule 5 amended): the q8 mode's regression ceilings are per K-quant activation format
//! (`common::configure_act`, `AQUEDUCT_ACT=q8_0|q8k`, default = production), each frozen from that format's
//! own 0..63 measurement at 2 x the max over the three prompts per layer (`tests/fixtures/q8_ceilings.json`,
//! `q8k_ceilings.json`). A format with no frozen file runs ungated and prints the numbers to freeze
//! (`tools/freeze_ceilings.py`).

mod common;

use std::path::PathBuf;
use std::time::Instant;

use aqueduct_core::kernels::matvec::{matvec, matvec_f32_in, Act, WeightMat};
use aqueduct_core::kernels::rmsnorm::rmsnorm;
use aqueduct_core::layers::{DecoderLayer, LayerError, MixerState, TensorSource};
use aqueduct_core::{dequantize, Gguf, ModelConfig};

const EPS: f64 = f32::EPSILON as f64;
/// Rule 5 constant, as in the tiny oracle (`budget_k_per_layer`).
const K: f64 = 16.0;
const THREADS: usize = 6;
const LM_HEAD_CHUNK_ROWS: usize = 8192;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    F32,
    Q8,
}

/// Weights dequantised to f32 in Rust (the reference's numbers, bit for bit).
struct DequantSource<'a>(&'a Gguf);

impl TensorSource for DequantSource<'_> {
    fn mat(&self, name: &str, rows: usize, cols: usize) -> Result<WeightMat, LayerError> {
        let t = self.0.tensor(name).map_err(|_| LayerError::Missing(name.to_string()))?;
        let found: Vec<usize> = t.shape_numpy_order().iter().map(|&d| d as usize).collect();
        if found != vec![rows, cols] {
            return Err(LayerError::Shape { name: name.into(), expected: vec![rows, cols], found });
        }
        let bytes = self.0.read_raw(name)?;
        let f = dequantize(t.ggml_type, &bytes, rows * cols)?;
        Ok(WeightMat::from_f32(rows, cols, &f))
    }
    fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, LayerError> {
        self.0.vec(name, len)
    }
}

struct Prompt {
    name: String,
    ids: Vec<u32>,
}

struct Ref {
    dir: PathBuf,
    prompts: Vec<Prompt>,
}

impl Ref {
    fn load() -> Ref {
        let dir = common::fixture("ref_gguf");
        let prompts: Vec<Prompt> = common::json(&common::fixtures().join("prompts.json"))["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| Prompt { name: p["name"].as_str().unwrap().to_string(), ids: p["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() })
            .collect();
        for p in &prompts {
            assert!(dir.join(format!("{}__hidden_embed.npy", p.name)).exists(), "reference missing at {}: run `python tools/ref_forward.py models/Qwen3.8-27B --weights gguf --dump-hidden`", dir.display());
        }
        Ref { dir, prompts }
    }
    /// The per-prompt json (argmax at every position, logits stats) is written at the end of the reference run.
    fn meta(&self, name: &str) -> serde_json::Value {
        common::json(&self.dir.join(format!("{name}.json")))
    }
    fn hidden(&self, name: &str, what: &str) -> Vec<f32> {
        common::read_npy_f32(&self.dir.join(format!("{name}__hidden_{what}.npy")))
    }
    fn logits(&self, name: &str) -> Vec<f32> {
        common::read_npy_f32(&self.dir.join(format!("{name}.logits.npy")))
    }
}

fn max_abs(v: &[f32]) -> f64 {
    v.iter().fold(0f64, |m, x| m.max(x.abs() as f64))
}

fn max_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

fn top_k(v: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..v.len() as u32).collect();
    idx.sort_by(|&a, &b| v[b as usize].partial_cmp(&v[a as usize]).unwrap().then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

fn overlap(a: &[u32], b: &[u32]) -> usize {
    a.iter().filter(|x| b.contains(x)).count()
}

/// Per prompt, per mode: the residual stream after the current layer at every position, and the layer state.
struct Stream {
    h: Vec<f32>,
}

/// One layer over one prompt: prefill all positions; then the incremental check on the last position.
fn run_layer(layer: &DecoderLayer, x: &[f32], t: usize, hidden: usize) -> (Vec<f32>, bool) {
    let mut state = layer.new_state();
    let mut y = vec![0f32; t * hidden];
    for i in 0..t - 1 {
        layer.forward_token(&x[i * hidden..(i + 1) * hidden], i as u32, &mut state, &mut y[i * hidden..(i + 1) * hidden], THREADS);
    }
    let mut decode_state = state.clone();
    let last = t - 1;
    layer.forward_token(&x[last * hidden..], last as u32, &mut state, &mut y[last * hidden..], THREADS);
    let mut y_inc = vec![0f32; hidden];
    layer.forward_token(&x[last * hidden..], last as u32, &mut decode_state, &mut y_inc, THREADS);
    let same = bits_equal(&y_inc, &y[last * hidden..]);
    let _ = MixerState::DeltaNet;
    (y, same)
}

struct Report {
    /// (layer, prompt, mode, max|diff| vs ref, budget, max|ref|)
    worst_f32: (usize, String, f64, f64),
    worst_q8_rel: (usize, String, f64),
    worst_q8_vs_f32_rel: (usize, String, f64),
    incremental_ok: bool,
}

#[allow(clippy::needless_range_loop)] // `l` names the layer in messages, the reference files and the ceilings
fn run_parity(n_layers: usize, with_logits: bool) -> Report {
    let g = Gguf::open(common::gguf_path()).expect("open gguf");
    let cfg = ModelConfig::from_gguf(&g).expect("config");
    let hidden = cfg.hidden_size as usize;
    let width = (cfg.intermediate_size as usize).max(hidden).max((cfg.dn_head_dim_k * cfg.dn_head_dim_v) as usize);
    let r = Ref::load();
    // Rule 5 (Phase 3, amended in 3.5: per activation format): the q8 mode's per-layer relative errors and
    // logit max|diff| are frozen as regression ceilings (x `factor`) for each K-quant activation format,
    // from that format's own measurement; any kernel change that exceeds one fails here. With no frozen file
    // for the selected format this is that format's measurement run: everything is printed, nothing gated.
    let (act_name, ceil_path) = common::configure_act();
    let ceil: Option<serde_json::Value> = ceil_path.exists().then(|| common::json(&ceil_path));
    println!(
        "q8 mode activations: {act_name}; ceilings: {}",
        match &ceil {
            Some(_) => ceil_path.display().to_string(),
            None => "none frozen for this format (measurement run, not gated)".to_string(),
        }
    );
    let ceil_factor = ceil.as_ref().map(|c| c["factor"].as_f64().unwrap()).unwrap_or(f64::NAN);
    // per layer (rule 5): the factor times the largest of the three prompts' measured values
    let layer_ceiling = |l: usize| -> Option<f64> {
        let c = ceil.as_ref()?;
        Some(ceil_factor * c["measured"]["layers"][l]["rel"].as_object().unwrap().values().map(|v| v.as_f64().unwrap()).fold(0f64, f64::max))
    };
    let mut worst_ceiling = (0usize, String::new(), 0f64); // fraction of ceiling
    let modes = [Mode::F32, Mode::Q8];
    let t0 = Instant::now();

    // embeddings: dequantised rows of token_embd, both modes, must equal the reference bit for bit
    let embd = g.tensor("token_embd.weight").unwrap();
    let (bs, ts) = embd.ggml_type.block_layout();
    let row_bytes = (hidden as u64 / bs * ts) as usize;
    let mut streams: Vec<Vec<Stream>> = Vec::new(); // [prompt][mode]
    for p in &r.prompts {
        let mut h = Vec::with_capacity(p.ids.len() * hidden);
        for &id in &p.ids {
            let mut buf = vec![0u8; row_bytes];
            let base = embd.absolute_file_offset + id as u64 * row_bytes as u64;
            read_at(&g, base, &mut buf);
            h.extend(dequantize(embd.ggml_type, &buf, hidden).unwrap());
        }
        let want = r.hidden(&p.name, "embed");
        assert!(bits_equal(&h, &want), "{}: embedding rows differ from the reference", p.name);
        streams.push(modes.iter().map(|_| Stream { h: h.clone() }).collect());
    }
    println!("embeddings: {} prompts bit-exact with the reference", r.prompts.len());

    let mut rep = Report { worst_f32: (0, String::new(), 0.0, 0.0), worst_q8_rel: (0, String::new(), 0.0), worst_q8_vs_f32_rel: (0, String::new(), 0.0), incremental_ok: true };
    for l in 0..n_layers {
        let tl = Instant::now();
        let refs: Vec<Vec<f32>> = r.prompts.iter().map(|p| r.hidden(&p.name, &format!("after_layer_{l:02}"))).collect();
        let mut line = format!("layer {l:2} {:<16}", if cfg.full_attention_layers.contains(&(l as u32)) { "full_attention" } else { "linear_attention" });
        for (mi, &mode) in modes.iter().enumerate() {
            let tm = Instant::now();
            let deq = DequantSource(&g);
            let src: &dyn TensorSource = match mode {
                Mode::F32 => &deq,
                Mode::Q8 => &g,
            };
            let layer = DecoderLayer::load(src, &cfg, l as u32).unwrap_or_else(|e| panic!("layer {l}: {e}"));
            let load_s = tm.elapsed().as_secs_f64();
            for (pi, p) in r.prompts.iter().enumerate() {
                let t = p.ids.len();
                let (y, same) = run_layer(&layer, &streams[pi][mi].h, t, hidden);
                rep.incremental_ok &= same;
                let want = &refs[pi];
                let ma = max_abs(want);
                let d = max_diff(&y, want);
                match mode {
                    Mode::F32 => {
                        let bud = K * (l as f64 + 1.0) * (width as f64).sqrt() * EPS * ma;
                        assert!(d <= bud, "{} layer {l} (f32 mode): max abs diff {d:e} exceeds budget {bud:e} (max|ref| {ma:e})", p.name);
                        if bud > 0.0 && d / bud > rep.worst_f32.2 / rep.worst_f32.3.max(1e-300) {
                            rep.worst_f32 = (l, p.name.clone(), d, bud);
                        }
                        line += &format!(" | {:<8} f32 {:.2e}/{:.2e}", p.name, d, bud);
                    }
                    Mode::Q8 => {
                        let rel = d / ma;
                        if rel > rep.worst_q8_rel.2 {
                            rep.worst_q8_rel = (l, p.name.clone(), rel);
                        }
                        if let Some(ceiling) = layer_ceiling(l) {
                            assert!(rel <= ceiling, "{} layer {l} (q8 mode, {act_name}): relative error {rel:.3e} exceeds the frozen ceiling {ceiling:.3e} (measured x {ceil_factor})", p.name);
                            if rel / ceiling > worst_ceiling.2 {
                                worst_ceiling = (l, p.name.clone(), rel / ceiling);
                            }
                        }
                        let d_f32 = max_diff(&y, &streams[pi][0].h);
                        let rel_f32 = d_f32 / ma;
                        if rel_f32 > rep.worst_q8_vs_f32_rel.2 {
                            rep.worst_q8_vs_f32_rel = (l, p.name.clone(), rel_f32);
                        }
                        line += &format!(" q8 {:.2e} ({:.2e} rel, vs f32 {:.2e})", d, rel, rel_f32);
                    }
                }
                streams[pi][mi].h = y;
            }
            line += &format!(" [{}: load {:.1}s run {:.1}s]", if mode == Mode::F32 { "f32" } else { "q8" }, load_s, tm.elapsed().as_secs_f64() - load_s);
        }
        println!("{line} incremental={} ({:.0}s, total {:.0}s)", if rep.incremental_ok { "bit-exact" } else { "DIFFERS" }, tl.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64());
    }
    assert!(rep.incremental_ok, "incremental (decode) step differs from the prefill's last position");

    if with_logits {
        assert_eq!(n_layers, cfg.n_layer as usize);
        let output_norm = g.vec("output_norm.weight", hidden).unwrap();
        let out_t = g.tensor("output.weight").unwrap();
        let vocab = cfg.vocab_size as usize;
        let raw = g.read_raw("output.weight").unwrap();
        let (obs, ots) = out_t.ggml_type.block_layout();
        let out_row_bytes = (hidden as u64 / obs * ots) as usize;
        let q8_head = WeightMat::new(out_t.ggml_type, vocab, hidden, raw.clone());
        let llama_dir = common::fixture("ref_llamacpp");
        for (mi, &mode) in modes.iter().enumerate() {
            let tm = Instant::now();
            for (pi, p) in r.prompts.iter().enumerate() {
                let t = p.ids.len();
                let h = &streams[pi][mi].h;
                let mut normed = vec![0f32; t * hidden];
                for i in 0..t {
                    rmsnorm(&h[i * hidden..(i + 1) * hidden], &output_norm, cfg.rms_norm_eps, &mut normed[i * hidden..(i + 1) * hidden]);
                }
                if mode == Mode::F32 {
                    let want = r.hidden(&p.name, "final_norm");
                    let ma = max_abs(&want);
                    let d = max_diff(&normed, &want);
                    let bud = K * (n_layers as f64 + 1.0) * (width as f64).sqrt() * EPS * ma;
                    assert!(d <= bud, "{} final norm: diff {d:e} > budget {bud:e}", p.name);
                    println!("{:<8} final norm (f32): max diff {d:.3e} vs budget {bud:.3e}", p.name);
                }
                let mut logits = vec![0f32; t * vocab];
                match mode {
                    Mode::F32 => {
                        for a in (0..vocab).step_by(LM_HEAD_CHUNK_ROWS) {
                            let b = (a + LM_HEAD_CHUNK_ROWS).min(vocab);
                            let f = dequantize(out_t.ggml_type, &raw[a * out_row_bytes..b * out_row_bytes], (b - a) * hidden).unwrap();
                            let w = WeightMat::from_f32(b - a, hidden, &f);
                            let mut y = vec![0f32; b - a];
                            for i in 0..t {
                                matvec(&w, Act::F32(&normed[i * hidden..(i + 1) * hidden]), &mut y, THREADS);
                                logits[i * vocab + a..i * vocab + b].copy_from_slice(&y);
                            }
                        }
                    }
                    Mode::Q8 => {
                        for i in 0..t {
                            matvec_f32_in(&q8_head, &normed[i * hidden..(i + 1) * hidden], &mut logits[i * vocab..(i + 1) * vocab], THREADS);
                        }
                    }
                }
                let meta = r.meta(&p.name);
                let ref_argmax_per_pos: Vec<u32> = meta["argmax_per_position"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
                let per_pos: Vec<u32> = (0..t).map(|i| argmax(&logits[i * vocab..(i + 1) * vocab])).collect();
                let pos_match = per_pos.iter().zip(&ref_argmax_per_pos).filter(|(a, b)| a == b).count();
                let last = &logits[(t - 1) * vocab..];
                let want = r.logits(&p.name);
                let d = max_diff(last, &want);
                let top10 = top_k(last, 10);
                let ov = overlap(&top10, &top_k(&want, 10));
                let am = argmax(last) == argmax(&want);
                let llama = common::read_npy_f32(&llama_dir.join(format!("{}.logits.npy", p.name)));
                let dl = max_diff(last, &llama);
                let ovl = overlap(&top10, &top_k(&llama, 10));
                let aml = argmax(last) == argmax(&llama);
                let mode_name = if mode == Mode::F32 { "f32" } else { "q8" };
                if mode == Mode::F32 {
                    let bud = K * (n_layers as f64 + 2.0) * (width as f64).sqrt() * EPS * meta["logits_stats"]["max_abs"].as_f64().unwrap();
                    assert!(d <= bud, "{} logits (f32 mode): diff {d:e} > budget {bud:e}", p.name);
                    println!("{:<8} logits {mode_name}: vs ref_gguf max|diff| {d:.3e} (budget {bud:.3e}) argmax {} top10 {ov}/10 | vs llama.cpp max|diff| {dl:.4} argmax {} top10 {ovl}/10 | argmax per position {pos_match}/{t}",
                        p.name, if am { "match" } else { "DIFF" }, if aml { "match" } else { "DIFF" });
                } else {
                    println!("{:<8} logits {mode_name}: vs ref_gguf max|diff| {d:.4} argmax {} top10 {ov}/10 | vs llama.cpp max|diff| {dl:.4} argmax {} top10 {ovl}/10 | argmax per position {pos_match}/{t}",
                        p.name, if am { "match" } else { "DIFF" }, if aml { "match" } else { "DIFF" });
                }
                if mode == Mode::Q8 {
                    if let Some(c) = &ceil {
                        let ceiling = ceil_factor * c["measured"]["logits"][&p.name]["vs_ref_gguf_max_diff"].as_f64().unwrap();
                        assert!(d <= ceiling, "{} logits (q8 mode, {act_name}): max|diff| {d:.4} vs ref_gguf exceeds the frozen ceiling {ceiling:.4}", p.name);
                        println!("{:<8} logits q8: {:.3} of the frozen ceiling {ceiling:.4}", p.name, d / ceiling);
                    }
                }
                if mode == Mode::F32 || ceil.is_some() {
                    assert!(am, "{} ({mode_name}): argmax differs from ref_gguf: got {} want {}", p.name, argmax(last), argmax(&want));
                    assert!(aml, "{} ({mode_name}): argmax differs from llama.cpp: got {} want {}", p.name, argmax(last), argmax(&llama));
                }
            }
            println!("lm_head {}: {:.0}s", if mode == Mode::F32 { "f32" } else { "q8" }, tm.elapsed().as_secs_f64());
        }
    }
    println!(
        "summary: worst f32 layer {} {}: diff {:.3e} vs budget {:.3e} ({:.3} of budget); worst q8 rel error vs ref: layer {} {} {:.3e}; worst q8 vs f32 rel: layer {} {} {:.3e}; incremental {}",
        rep.worst_f32.0, rep.worst_f32.1, rep.worst_f32.2, rep.worst_f32.3, rep.worst_f32.2 / rep.worst_f32.3.max(1e-300),
        rep.worst_q8_rel.0, rep.worst_q8_rel.1, rep.worst_q8_rel.2, rep.worst_q8_vs_f32_rel.0, rep.worst_q8_vs_f32_rel.1, rep.worst_q8_vs_f32_rel.2,
        if rep.incremental_ok { "bit-exact" } else { "DIFFERS" }
    );
    match &ceil {
        Some(_) => println!("ceilings ({act_name}): worst layer {} {} at {:.3} of its frozen ceiling (factor {ceil_factor})", worst_ceiling.0, worst_ceiling.1, worst_ceiling.2),
        None => println!("ceilings ({act_name}): none frozen; this run is the measurement to freeze with tools/freeze_ceilings.py"),
    }
    rep
}

/// Read `buf.len()` bytes at absolute file offset `off` (embedding rows are fetched individually so the
/// 682 MiB table is never resident).
fn read_at(g: &Gguf, off: u64, buf: &mut [u8]) {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(g.path()).unwrap();
    f.seek(SeekFrom::Start(off)).unwrap();
    f.read_exact(buf).unwrap();
}

#[test]
fn real_layers_0_7() {
    run_parity(8, false);
}

#[test]
#[ignore = "all 64 layers on the real GGUF: minutes; run with --release --ignored"]
fn real_layers_0_63_and_logits() {
    run_parity(64, true);
}
