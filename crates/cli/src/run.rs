//! `aqueduct run`: load the whole GGUF into RAM, feed a prompt (raw ids or raw text, no chat template),
//! decode greedily. Stats go to stderr; ids and text to stdout (`--ids-only`: the generated ids alone, the
//! test channel).

use std::time::Instant;

use aqueduct_core::kernels::matvec::set_q8k_activations;
use aqueduct_core::model::{argmax, Model};
use aqueduct_core::Tok;

use crate::rss::peak_rss_bytes;

pub const DEFAULT_GGUF: &str = r"C:\models\Qwen3.8-27B-Q4_K_M.gguf";

pub struct RunArgs {
    pub model: String,
    pub tokenizer: String,
    pub threads: usize,
    pub max_tokens: usize,
    pub ids: Option<Vec<u32>>,
    pub prompt: Option<String>,
    pub ids_only: bool,
}

pub fn parse(args: &[String], default_tokenizer: &str) -> Result<RunArgs, String> {
    let all = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut a = RunArgs { model: DEFAULT_GGUF.into(), tokenizer: default_tokenizer.into(), threads: all, max_tokens: 32, ids: None, prompt: None, ids_only: false };
    let mut i = 0;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]));
        match args[i].as_str() {
            "--model" => {
                a.model = next(i)?.clone();
                i += 2;
            }
            "--tokenizer" => {
                a.tokenizer = next(i)?.clone();
                i += 2;
            }
            "--threads" => {
                a.threads = next(i)?.parse().map_err(|e| format!("--threads: {e}"))?;
                i += 2;
            }
            "--max-tokens" => {
                a.max_tokens = next(i)?.parse().map_err(|e| format!("--max-tokens: {e}"))?;
                i += 2;
            }
            "--ids" => {
                let ids: Result<Vec<u32>, _> = next(i)?.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse::<u32>()).collect();
                a.ids = Some(ids.map_err(|e| format!("--ids: {e}"))?);
                i += 2;
            }
            "--prompt" => {
                a.prompt = Some(next(i)?.clone());
                i += 2;
            }
            "--ids-only" => {
                a.ids_only = true;
                i += 1;
            }
            "--q8k" => {
                set_q8k_activations(true);
                i += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.ids.is_none() == a.prompt.is_none() {
        return Err("give exactly one of --ids <csv> or --prompt <text>".into());
    }
    Ok(a)
}

pub fn run(a: RunArgs) -> Result<(), String> {
    let e = |m: String| m;
    let tok = if a.prompt.is_some() || !a.ids_only { Some(Tok::from_file(&a.tokenizer).map_err(|x| e(format!("tokenizer: {x}")))?) } else { None };
    let prompt_ids: Vec<u32> = match (&a.ids, &a.prompt) {
        (Some(ids), _) => ids.clone(),
        (None, Some(text)) => tok.as_ref().unwrap().encode(text).map_err(|x| e(format!("encode: {x}")))?,
        _ => unreachable!(),
    };
    if prompt_ids.is_empty() {
        return Err("empty prompt".into());
    }
    eprintln!("loading {} with {} threads ...", a.model, a.threads);
    let model = Model::load(&a.model, a.threads).map_err(|x| e(format!("load: {x}")))?;
    let rss_after_load = peak_rss_bytes();
    eprintln!(
        "loaded in {:.1} s: {} layers, {:.3} GB of weights, peak RSS {}",
        model.load_secs,
        model.layers.len(),
        model.weight_bytes as f64 / 1e9,
        rss_after_load.map(|b| format!("{:.3} GB", b as f64 / 1e9)).unwrap_or_else(|| "n/a".into())
    );

    let mut state = model.new_state();
    let mut logits = vec![0f32; model.vocab()];
    // prefill: token by token (3.3 batches this), logits only for the last position
    let t0 = Instant::now();
    let mut h = Vec::new();
    for &id in &prompt_ids {
        h = model.forward_hidden(id, &mut state);
    }
    model.logits_of(&h, &mut logits);
    let prefill_s = t0.elapsed().as_secs_f64();
    eprintln!("prefill: {} tokens in {:.2} s ({:.2} tok/s)", prompt_ids.len(), prefill_s, prompt_ids.len() as f64 / prefill_s);

    let mut out: Vec<u32> = Vec::new();
    let mut next = argmax(&logits);
    let t1 = Instant::now();
    let mut stopped = None;
    for _ in 0..a.max_tokens {
        if model.is_stop(next) {
            stopped = Some(next);
            break;
        }
        out.push(next);
        if out.len() == a.max_tokens {
            break;
        }
        model.forward_token(next, &mut state, &mut logits);
        next = argmax(&logits);
    }
    let decode_s = t1.elapsed().as_secs_f64();
    let steps = out.len().saturating_sub(1).max(1);
    eprintln!(
        "decode: {} tokens, {:.3} s/token ({:.2} tok/s), {} threads{}",
        out.len(),
        decode_s / steps as f64,
        steps as f64 / decode_s,
        model.threads,
        stopped.map(|s| format!(", stopped at id {s}")).unwrap_or_default()
    );
    if let Some(rss) = peak_rss_bytes() {
        let expected = model.weight_bytes + model.state_bytes(prompt_ids.len() + out.len());
        eprintln!("peak RSS {:.3} GB vs expected {:.3} GB (weights + state)", rss as f64 / 1e9, expected as f64 / 1e9);
    }
    let ids_line = out.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    if a.ids_only {
        println!("{ids_line}");
    } else {
        println!("prompt ids: {}", prompt_ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        println!("generated ids: {ids_line}");
        let text = tok.as_ref().unwrap().decode(&out).map_err(|x| e(format!("decode: {x}")))?;
        println!("text: {text:?}");
    }
    Ok(())
}
