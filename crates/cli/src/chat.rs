//! `aqueduct chat` (Phase 5.3): a multi-turn REPL on the chat template. Each turn renders the whole
//! conversation with `chat_template.jinja` (thinking on by default, `--no-think` for the template's
//! thinking-off form), tokenises it, and feeds the model only the tokens it has not consumed yet (the rendering
//! of a finished turn is a prefix of the next one's when the model's own output re-renders to itself; when it
//! does not, the context is re-encoded from scratch and said so). Tokens stream to the terminal as they are
//! emitted; the turn stops on the full stop set (`<|im_end|>`, `<|endoftext|>`), on `--max-tokens`, or on
//! Ctrl-C, which ends the turn and not the process. Sampling defaults come from `generation_config.json`;
//! `--spec k` drafts with the MTP head and the turn's acceptance rate is printed. `--show-config` prints every
//! setting and exits.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use aqueduct_core::chat::{ChatTemplate, Message, RenderOpts};
use aqueduct_core::model::{argmax, LoadOpts, Model, State};
use aqueduct_core::os::{apply_job_memory_limit, DirectFile, DEFAULT_CHUNK};
use aqueduct_core::sample::{Sampler, SamplerConfig};
use aqueduct_core::tier::{parse_bytes, MemoryPlan, PlanInput, PlanParams};
use aqueduct_core::{Gguf, ModelConfig, Tok};

use crate::run::DEFAULT_GGUF;

pub const DEFAULT_TEMPLATE: &str = "models/Qwen3.8-27B/chat_template.jinja";
pub const DEFAULT_GEN_CONFIG: &str = "models/Qwen3.8-27B/generation_config.json";

static INTERRUPT: AtomicBool = AtomicBool::new(false);

#[cfg(windows)]
fn install_ctrl_c() {
    extern "system" {
        fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
    }
    unsafe extern "system" fn handler(ctrl: u32) -> i32 {
        // CTRL_C_EVENT = 0, CTRL_BREAK_EVENT = 1: end the turn; anything else (close, logoff) proceeds
        if ctrl <= 1 {
            INTERRUPT.store(true, Ordering::SeqCst);
            1
        } else {
            0
        }
    }
    // SAFETY: the handler only touches an atomic.
    unsafe {
        SetConsoleCtrlHandler(Some(handler), 1);
    }
}

#[cfg(not(windows))]
fn install_ctrl_c() {
    extern "C" {
        fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    }
    extern "C" fn handler(_sig: i32) {
        INTERRUPT.store(true, Ordering::SeqCst);
    }
    // SAFETY: SIGINT = 2; the handler only touches an atomic.
    unsafe {
        signal(2, handler);
    }
}

pub struct ChatArgs {
    pub model: String,
    pub tokenizer: String,
    pub template: String,
    pub gen_config: String,
    pub threads: usize,
    pub budget: Option<u64>,
    pub job_limit: Option<u64>,
    pub slots: usize,
    pub qd: usize,
    pub large_pages: bool,
    pub spec: usize,
    pub max_tokens: usize,
    pub max_pos: usize,
    pub think: bool,
    pub reasoning_effort: Option<String>,
    pub preserve_thinking: bool,
    pub system: Option<String>,
    pub sampler: SamplerConfig,
    pub sampler_given: Vec<String>,
    pub show_config: bool,
    pub verbose: bool,
}

pub fn parse(args: &[String], default_tokenizer: &str) -> Result<ChatArgs, String> {
    let mut a = ChatArgs {
        model: DEFAULT_GGUF.into(),
        tokenizer: default_tokenizer.into(),
        template: DEFAULT_TEMPLATE.into(),
        gen_config: DEFAULT_GEN_CONFIG.into(),
        threads: aqueduct_core::physical_cores(),
        budget: None,
        job_limit: None,
        slots: 2,
        qd: 2,
        large_pages: true,
        spec: 0,
        max_tokens: 1024,
        max_pos: 4096,
        think: true,
        reasoning_effort: None,
        preserve_thinking: true,
        system: None,
        sampler: SamplerConfig::greedy(),
        sampler_given: Vec::new(),
        show_config: false,
        verbose: false,
    };
    let mut greedy = false;
    let (mut temperature, mut top_k, mut top_p, mut min_p, mut seed) = (None, None, None, None, None);
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
            "--template" => {
                a.template = next(i)?.clone();
                i += 2;
            }
            "--gen-config" => {
                a.gen_config = next(i)?.clone();
                i += 2;
            }
            "--threads" => {
                a.threads = next(i)?.parse().map_err(|e| format!("--threads: {e}"))?;
                i += 2;
            }
            "--budget" => {
                a.budget = Some(parse_bytes(next(i)?)?);
                i += 2;
            }
            "--job-limit" => {
                a.job_limit = Some(parse_bytes(next(i)?)?);
                i += 2;
            }
            "--slots" => {
                a.slots = next(i)?.parse().map_err(|e| format!("--slots: {e}"))?;
                i += 2;
            }
            "--qd" => {
                a.qd = next(i)?.parse().map_err(|e| format!("--qd: {e}"))?;
                i += 2;
            }
            "--spec" => match args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                Some(k) => {
                    a.spec = k;
                    i += 2;
                }
                None => {
                    a.spec = crate::run::DEFAULT_SPEC_K;
                    i += 1;
                }
            },
            "--max-tokens" => {
                a.max_tokens = next(i)?.parse().map_err(|e| format!("--max-tokens: {e}"))?;
                i += 2;
            }
            "--max-pos" => {
                a.max_pos = next(i)?.parse().map_err(|e| format!("--max-pos: {e}"))?;
                i += 2;
            }
            "--reasoning-effort" => {
                a.reasoning_effort = Some(next(i)?.clone());
                i += 2;
            }
            "--system" => {
                a.system = Some(next(i)?.clone());
                i += 2;
            }
            "--temperature" => {
                temperature = Some(next(i)?.parse::<f32>().map_err(|e| format!("--temperature: {e}"))?);
                i += 2;
            }
            "--top-k" => {
                top_k = Some(next(i)?.parse::<usize>().map_err(|e| format!("--top-k: {e}"))?);
                i += 2;
            }
            "--top-p" => {
                top_p = Some(next(i)?.parse::<f32>().map_err(|e| format!("--top-p: {e}"))?);
                i += 2;
            }
            "--min-p" => {
                min_p = Some(next(i)?.parse::<f32>().map_err(|e| format!("--min-p: {e}"))?);
                i += 2;
            }
            "--seed" => {
                seed = Some(next(i)?.parse::<u64>().map_err(|e| format!("--seed: {e}"))?);
                i += 2;
            }
            "--no-think" => {
                a.think = false;
                i += 1;
            }
            "--no-preserve-thinking" => {
                a.preserve_thinking = false;
                i += 1;
            }
            "--greedy" => {
                greedy = true;
                i += 1;
            }
            "--no-large-pages" => {
                a.large_pages = false;
                i += 1;
            }
            "--show-config" => {
                a.show_config = true;
                i += 1;
            }
            "-v" | "--verbose" => {
                a.verbose = true;
                i += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    // sampling: generation_config.json's defaults, then the flags; --greedy wins
    let (mut cfg, given) = match std::fs::read_to_string(&a.gen_config) {
        Ok(text) => SamplerConfig::from_generation_config_json(&text).map_err(|e| format!("{}: {e}", a.gen_config))?,
        Err(e) => return Err(format!("read {}: {e} (give --gen-config)", a.gen_config)),
    };
    a.sampler_given = given;
    if let Some(t) = temperature {
        cfg.temperature = t;
    }
    if let Some(k) = top_k {
        cfg.top_k = k;
    }
    if let Some(p) = top_p {
        cfg.top_p = p;
    }
    if let Some(p) = min_p {
        cfg.min_p = p;
    }
    cfg.seed = seed.unwrap_or(0);
    if greedy {
        cfg = SamplerConfig::greedy();
    }
    a.sampler = cfg;
    Ok(a)
}

fn render_opts(a: &ChatArgs) -> RenderOpts {
    RenderOpts { enable_thinking: if a.think { None } else { Some(false) }, reasoning_effort: a.reasoning_effort.clone(), preserve_thinking: if a.preserve_thinking { None } else { Some(false) }, add_generation_prompt: true }
}

fn show_config(a: &ChatArgs, stop: &[u32], plan: Option<&MemoryPlan>) {
    println!("chat configuration");
    println!("  model             : {}", a.model);
    println!("  tokenizer         : {}", a.tokenizer);
    println!("  chat template     : {}", a.template);
    println!("  generation config : {} (gives: {})", a.gen_config, if a.sampler_given.is_empty() { "nothing".to_string() } else { a.sampler_given.join(", ") });
    println!("  sampling          : {}", a.sampler);
    println!("  thinking          : {}", if a.think { "on (the template's default: <think> opened for the model)".to_string() } else { "off (--no-think: the template pre-fills an empty <think></think>)".to_string() });
    println!("  reasoning effort  : {}", a.reasoning_effort.clone().unwrap_or_else(|| "xhigh (template default)".into()));
    println!("  preserve thinking : {}", a.preserve_thinking);
    println!("  system prompt     : {}", a.system.as_deref().unwrap_or("(none)"));
    println!("  stop ids          : {stop:?}");
    println!("  max tokens / turn : {}", a.max_tokens);
    println!("  max positions     : {} (context the KV caches are sized for)", a.max_pos);
    println!("  threads           : {}", a.threads);
    println!("  speculative       : {}", if a.spec > 0 { format!("--spec {}: {} MTP drafts per round, exact verification", a.spec, a.spec) } else { "off".into() });
    println!("  budget            : {}", a.budget.map_or("none (everything resident)".to_string(), |b| format!("{b} bytes = {:.2} GiB, ring {} slots, qd {}", b as f64 / (1u64 << 30) as f64, a.slots, a.qd)));
    println!("  job limit         : {}", a.job_limit.map_or("none".to_string(), |b| format!("{b} bytes")));
    println!("  large pages       : {}", if a.large_pages { "attempted" } else { "off" });
    if let Some(p) = plan {
        print!("{}", p.table());
    }
}

/// Split a finished (or interrupted) assistant output into `(reasoning_content, content)` the way the
/// template renders them back: with thinking on the prompt opened `<think>\n`, so everything before
/// `</think>` is reasoning; with thinking off the prompt closed the block itself, so it is all content.
fn split_output(text: &str, think: bool) -> (Option<String>, String) {
    if !think {
        return (None, text.trim().to_string());
    }
    match text.find("</think>") {
        Some(i) => (Some(text[..i].trim().to_string()), text[i + "</think>".len()..].trim().to_string()),
        None => (Some(text.trim().to_string()), String::new()),
    }
}

pub fn chat(args: &[String], default_tokenizer: &str) -> Result<(), String> {
    let a = parse(args, default_tokenizer)?;
    let e = |m: String| m;
    if a.show_config {
        let plan = match a.budget {
            Some(b) => {
                let g = Gguf::open(&a.model).map_err(|x| e(format!("open: {x}")))?;
                let cfg = ModelConfig::from_gguf(&g).map_err(|x| e(format!("config: {x}")))?;
                let input = PlanInput::new(&g, &cfg);
                let sector = DirectFile::open(std::path::Path::new(&a.model), DEFAULT_CHUNK, 1).map_err(|x| e(format!("open: {x}")))?.sector as u64;
                let mut params = PlanParams::new(Some(b), a.max_pos as u64, a.slots, sector);
                params.spec_k = a.spec as u64;
                Some(MemoryPlan::compute(&input, &params))
            }
            None => None,
        };
        let stop = match Gguf::open(&a.model).ok().and_then(|g| ModelConfig::from_gguf(&g).ok()) {
            Some(cfg) => aqueduct_core::stop_ids(&cfg.eos_ids, aqueduct_core::arch_consts(&cfg.architecture).map_err(|x| e(x.to_string()))?),
            None => Vec::new(),
        };
        show_config(&a, &stop, plan.as_ref());
        return Ok(());
    }
    if let Some(limit) = a.job_limit {
        apply_job_memory_limit(limit).map_err(|x| e(format!("job memory limit: {x}")))?;
    }
    let tok = Tok::from_file(&a.tokenizer).map_err(|x| e(format!("tokenizer: {x}")))?;
    let tpl = ChatTemplate::from_file(&a.template).map_err(|x| e(format!("template: {x}")))?;
    let opts = LoadOpts { budget: a.budget, max_pos: a.max_pos, n_slots: a.slots, large_pages: a.large_pages, qd: a.qd, verbose: a.verbose, spec_k: a.spec, ..LoadOpts::default() };
    eprintln!("loading {} ({} threads{}) ...", a.model, a.threads, if a.spec > 0 { format!(", --spec {}", a.spec) } else { String::new() });
    let model = Model::load_with(&a.model, a.threads, &opts).map_err(|x| e(format!("load {}: {x}", a.model)))?;
    eprintln!("loaded in {:.1} s: {} pinned, {} streamed layers; {:.3} GB of weights held", model.load_secs, model.plan.pinned, model.plan.streamed(), model.weight_bytes as f64 / 1e9);
    if a.spec > 0 && model.mtp.is_none() {
        return Err("--spec: this GGUF has no MTP block".into());
    }
    show_config(&a, &model.stop, None);
    let ropts = render_opts(&a);
    let vocab = model.vocab();
    let hidden = model.hidden();
    let new_state = |model: &Model| -> Result<State, String> {
        let mut s = if a.spec > 0 { model.new_state_spec(a.spec).map_err(|x| x.to_string())? } else { model.new_state() };
        s.reserve(a.max_pos);
        Ok(s)
    };
    let mut state = new_state(&model)?;
    let mut sampler = Sampler::new(a.sampler.clone(), vocab);
    let mut consumed: Vec<u32> = Vec::new();
    let mut messages: Vec<Message> = a.system.iter().map(|s| Message::new("system", s)).collect();
    let mut logits = vec![0f32; vocab];
    install_ctrl_c();
    println!("(type a message; /quit ends, /reset clears the conversation, /config shows the settings; Ctrl-C ends the turn)");
    let stdin = io::stdin();
    let mut out = io::stdout();
    loop {
        print!("\n> ");
        out.flush().ok();
        let mut line = String::new();
        let n = loop {
            match stdin.lock().read_line(&mut line) {
                Ok(n) => break n,
                Err(err) if INTERRUPT.swap(false, Ordering::SeqCst) || err.kind() == io::ErrorKind::Interrupted => {
                    line.clear();
                    continue;
                }
                Err(err) => return Err(format!("stdin: {err}")),
            }
        };
        if n == 0 {
            println!();
            break;
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        match line.as_str() {
            "/quit" | "/exit" => break,
            "/config" => {
                show_config(&a, &model.stop, None);
                continue;
            }
            "/reset" => {
                messages = a.system.iter().map(|s| Message::new("system", s)).collect();
                state = new_state(&model)?;
                consumed.clear();
                sampler.reseed(a.sampler.seed);
                println!("(conversation cleared)");
                continue;
            }
            _ => {}
        }
        messages.push(Message::new("user", &line));
        let rendered = tpl.render(&messages, &ropts).map_err(|x| e(format!("render: {x}")))?;
        let ids = tok.encode(&rendered).map_err(|x| e(format!("encode: {x}")))?;
        if ids.len() + a.max_tokens + a.spec + 2 > a.max_pos {
            messages.pop();
            println!("(the conversation is {} tokens; with --max-tokens {} it does not fit --max-pos {}: /reset, or restart with a larger --max-pos)", ids.len(), a.max_tokens, a.max_pos);
            continue;
        }
        let reuse = ids.len() > consumed.len() && ids[..consumed.len()] == consumed[..];
        if !reuse && !consumed.is_empty() {
            println!("(the rendered conversation is not an extension of what the model has consumed: re-encoding {} tokens)", ids.len());
            state = new_state(&model)?;
            consumed.clear();
        }
        let suffix: Vec<u32> = ids[consumed.len()..].to_vec();
        // ---- prefill the new tokens
        INTERRUPT.store(false, Ordering::SeqCst);
        let t0 = Instant::now();
        let first = if a.spec > 0 {
            model.spec_prefill(&suffix, &mut state, &mut sampler)
        } else {
            let hs = model.prefill(&suffix, &mut state);
            model.logits_of(&hs[(suffix.len() - 1) * hidden..], &mut logits);
            if sampler.cfg.is_greedy() {
                argmax(&logits)
            } else {
                sampler.sample(&logits)
            }
        };
        consumed.extend_from_slice(&suffix);
        let prefill_s = t0.elapsed().as_secs_f64();
        let spec_before = state.spec.as_ref().map(|s| s.stats.clone());
        // ---- decode, streaming the text
        let t1 = Instant::now();
        let mut generated: Vec<u32> = Vec::new();
        let mut printed = String::new();
        let mut pending = first;
        let mut interrupted = false;
        let mut poisoned = false;
        'turn: loop {
            if model.is_stop(pending) {
                break;
            }
            generated.push(pending);
            print_new(&tok, &generated, &mut printed, &mut out);
            if generated.len() >= a.max_tokens {
                break;
            }
            if INTERRUPT.swap(false, Ordering::SeqCst) {
                interrupted = true;
                break;
            }
            if a.spec > 0 {
                model.spec_round(&mut state, pending, &mut sampler);
                consumed.push(pending);
                let emitted: Vec<u32> = state.spec.as_ref().unwrap().emitted.clone();
                let m = emitted.len() - 1;
                for (i, &t) in emitted[..m].iter().enumerate() {
                    consumed.push(t);
                    if model.is_stop(t) {
                        // consumed past a stop: the state is ahead of the conversation
                        poisoned = i + 1 < m;
                        pending = t;
                        break 'turn;
                    }
                    generated.push(t);
                    print_new(&tok, &generated, &mut printed, &mut out);
                    if generated.len() >= a.max_tokens {
                        poisoned = i + 1 < m;
                        pending = t;
                        break 'turn;
                    }
                }
                pending = emitted[m];
            } else {
                model.forward_token(pending, &mut state, &mut logits);
                consumed.push(pending);
                pending = if sampler.cfg.is_greedy() { argmax(&logits) } else { sampler.sample(&logits) };
            }
        }
        let decode_s = t1.elapsed().as_secs_f64();
        let text = tok.decode(&generated).unwrap_or_default();
        // print whatever was held back (an incomplete UTF-8 sequence at the end)
        if text.len() > printed.len() && text.starts_with(&printed) {
            print!("{}", &text[printed.len()..]);
        }
        println!();
        if poisoned {
            consumed.clear();
        }
        let (reasoning, content) = split_output(&text, a.think);
        messages.push(Message { role: "assistant".into(), content, reasoning_content: reasoning });
        let mut line = format!(
            "[{} prompt tokens in {:.2} s; {} generated in {:.2} s = {:.2} tok/s{}{}]",
            suffix.len(),
            prefill_s,
            generated.len(),
            decode_s,
            generated.len() as f64 / decode_s.max(1e-9),
            if interrupted { "; interrupted" } else if model.is_stop(pending) { "; stopped" } else { "; max tokens" },
            if reuse || consumed.len() == suffix.len() { "" } else { "; re-encoded" }
        );
        if let (Some(b), Some(s)) = (spec_before, state.spec.as_ref().map(|s| &s.stats)) {
            let rounds = s.rounds - b.rounds;
            let drafted = s.drafted - b.drafted;
            let accepted = s.accepted - b.accepted;
            line.push_str(&format!(
                " [spec {}: {} rounds, {}/{} drafts accepted = {:.1} %, {:.2} accepted per round]",
                a.spec,
                rounds,
                accepted,
                drafted,
                100.0 * accepted as f64 / drafted.max(1) as f64,
                accepted as f64 / rounds.max(1) as f64
            ));
        }
        println!("{line}");
    }
    Ok(())
}

/// Print the part of the decoded text that is new since the last call, holding back a trailing replacement
/// character (an incomplete multi-byte sequence whose remaining bytes are in the next token).
fn print_new(tok: &Tok, generated: &[u32], printed: &mut String, out: &mut io::Stdout) {
    let text = tok.decode(generated).unwrap_or_default();
    let mut end = text.len();
    if text.ends_with('\u{fffd}') {
        end -= '\u{fffd}'.len_utf8();
    }
    if end > printed.len() && text[..end].starts_with(printed.as_str()) {
        print!("{}", &text[printed.len()..end]);
        out.flush().ok();
        *printed = text[..end].to_string();
    } else if !text[..end].starts_with(printed.as_str()) {
        // the decoded prefix changed (a merge across tokens): reprint from the divergence
        let common = text[..end].bytes().zip(printed.bytes()).take_while(|(a, b)| a == b).count();
        let common = text[..common].char_indices().map(|(i, _)| i).next_back().unwrap_or(0);
        print!("\n[...] {}", &text[common..end]);
        out.flush().ok();
        *printed = text[..end].to_string();
    }
}
