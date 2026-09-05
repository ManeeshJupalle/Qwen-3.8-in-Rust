//! `aqueduct run`: load the GGUF under a memory plan (everything resident, or `--budget` with the rest
//! streamed from disk), feed a prompt (raw ids or raw text, no chat template), decode greedily by default
//! or with sampling (`--sample`, or any of `--temperature --top-k --top-p --min-p --seed`), plainly or with
//! MTP speculative decoding (`--spec k`, Phase 5). Stats go to stderr; ids and text to stdout (`--ids-only`:
//! the generated ids alone, the test channel). `--stats <file>` writes one JSON object with every number the
//! ladder and the acceptance runs record.

use std::time::Instant;

use aqueduct_core::kernels::matvec::set_q8_fine;
use aqueduct_core::model::{argmax, LoadOpts, Model};
use aqueduct_core::os::{apply_job_memory_limit, job_peak_memory, large_page_note};
use aqueduct_core::sample::{Sampler, SamplerConfig};
use aqueduct_core::tier::parse_bytes;
use aqueduct_core::Tok;

use aqueduct_core::rss::peak_rss_bytes;

pub const DEFAULT_GGUF: &str = r"C:\models\Qwen3.8-27B-Q4_K_M.gguf";

/// The non-matvec constant of the cost model (docs/data/nonmatvec_profile.txt: 53 ms per token).
pub const NON_MATVEC_S: f64 = 0.053;

/// Drafts per round when `--spec` is given without a number: chosen from the acceptance measurement of
/// Phase 5.4 (`docs/data/spec_acceptance.txt`, `docs/phase5-report.md`).
pub const DEFAULT_SPEC_K: usize = 3;

pub struct RunArgs {
    pub model: String,
    pub tokenizer: String,
    pub threads: usize,
    pub max_tokens: usize,
    pub ids: Option<Vec<u32>>,
    pub prompt: Option<String>,
    pub ids_only: bool,
    /// Feed the prompt token by token instead of the batched prefill (the 3.2 path, for timing comparisons).
    pub sequential_prefill: bool,
    /// Print the per-stage decode breakdown (needs `--features profile`).
    pub profile: bool,
    pub verbose: bool,
    /// Phase 4: the memory budget (None = everything resident), the job-object cap, ring depth, read params.
    pub budget: Option<u64>,
    pub job_limit: Option<u64>,
    pub slots: usize,
    pub max_pos: Option<usize>,
    pub qd: usize,
    pub large_pages: bool,
    pub stats: Option<String>,
    /// For the predicted s/token line: membw and disk GB/s (from `aqueduct doctor`).
    pub membw: Option<f64>,
    pub diskbw: Option<f64>,
    /// Phase 5: MTP drafts per round (0 = plain decode), and the sampler (greedy unless asked).
    pub spec: usize,
    pub sampler: SamplerConfig,
    pub sampler_given: Vec<String>,
}

pub fn parse(args: &[String], default_tokenizer: &str) -> Result<RunArgs, String> {
    // physical cores, not hardware threads: the matvec kernels are memory-bound, so SMT siblings contend
    // rather than add bandwidth (`aqueduct_core::cpu`, finding 51). `--threads N` overrides.
    let all = aqueduct_core::physical_cores();
    let mut a = RunArgs {
        model: DEFAULT_GGUF.into(),
        tokenizer: default_tokenizer.into(),
        threads: all,
        max_tokens: 32,
        ids: None,
        prompt: None,
        ids_only: false,
        sequential_prefill: false,
        profile: false,
        verbose: false,
        budget: None,
        job_limit: None,
        slots: 2,
        max_pos: None,
        qd: 2,
        large_pages: true,
        stats: None,
        membw: None,
        diskbw: None,
        spec: 0,
        sampler: SamplerConfig::greedy(),
        sampler_given: Vec::new(),
    };
    let mut sample = false;
    let mut gen_config = crate::chat::DEFAULT_GEN_CONFIG.to_string();
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
            "--max-pos" => {
                a.max_pos = Some(next(i)?.parse().map_err(|e| format!("--max-pos: {e}"))?);
                i += 2;
            }
            "--qd" => {
                a.qd = next(i)?.parse().map_err(|e| format!("--qd: {e}"))?;
                i += 2;
            }
            "--stats" => {
                a.stats = Some(next(i)?.clone());
                i += 2;
            }
            "--membw" => {
                a.membw = Some(next(i)?.parse().map_err(|e| format!("--membw: {e}"))?);
                i += 2;
            }
            "--diskbw" => {
                a.diskbw = Some(next(i)?.parse().map_err(|e| format!("--diskbw: {e}"))?);
                i += 2;
            }
            "--spec" => match args.get(i + 1).and_then(|s| s.parse::<usize>().ok()) {
                Some(k) => {
                    a.spec = k;
                    i += 2;
                }
                None => {
                    a.spec = DEFAULT_SPEC_K;
                    i += 1;
                }
            },
            "--gen-config" => {
                gen_config = next(i)?.clone();
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
            "--sample" => {
                sample = true;
                i += 1;
            }
            "--no-large-pages" => {
                a.large_pages = false;
                i += 1;
            }
            "--ids-only" => {
                a.ids_only = true;
                i += 1;
            }
            "--q8-fine" => {
                set_q8_fine(true);
                i += 1;
            }
            "--profile" => {
                a.profile = true;
                i += 1;
            }
            "--sequential-prefill" => {
                a.sequential_prefill = true;
                i += 1;
            }
            "-v" | "--verbose" => {
                a.verbose = true;
                i += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.ids.is_none() == a.prompt.is_none() {
        return Err("give exactly one of --ids <csv> or --prompt <text>".into());
    }
    if sample || temperature.is_some() || top_k.is_some() || top_p.is_some() || min_p.is_some() || seed.is_some() {
        // the file's defaults, then the flags
        let (mut cfg, given) = match std::fs::read_to_string(&gen_config) {
            Ok(text) => SamplerConfig::from_generation_config_json(&text).map_err(|e| format!("{gen_config}: {e}"))?,
            Err(_) if !sample => (SamplerConfig { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, seed: 0 }, vec!["(no generation_config.json: flags only)".into()]),
            Err(e) => return Err(format!("--sample: read {gen_config}: {e}")),
        };
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
        a.sampler = cfg;
        a.sampler_given = given;
    }
    Ok(a)
}

fn json_list<T: std::fmt::Display>(v: &[T]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}

pub fn run(a: RunArgs) -> Result<(), String> {
    let e = |m: String| m;
    // the cap first, before anything of size is allocated: from here on the process cannot commit more
    if let Some(limit) = a.job_limit {
        apply_job_memory_limit(limit).map_err(|x| e(format!("job memory limit: {x}")))?;
        eprintln!("job object: committed memory capped at {limit} bytes ({:.2} GiB) for this process", limit as f64 / (1u64 << 30) as f64);
    }
    let tok = if a.prompt.is_some() || !a.ids_only { Some(Tok::from_file(&a.tokenizer).map_err(|x| e(format!("tokenizer: {x}")))?) } else { None };
    let prompt_ids: Vec<u32> = match (&a.ids, &a.prompt) {
        (Some(ids), _) => ids.clone(),
        (None, Some(text)) => tok.as_ref().unwrap().encode(text).map_err(|x| e(format!("encode: {x}")))?,
        _ => unreachable!(),
    };
    if prompt_ids.is_empty() {
        return Err("empty prompt".into());
    }
    let k = a.spec;
    let max_pos = a.max_pos.unwrap_or(prompt_ids.len() + a.max_tokens + k + 2);
    let opts = LoadOpts { budget: a.budget, max_pos, n_slots: a.slots, large_pages: a.large_pages, qd: a.qd, verbose: true, spec_k: k, ..LoadOpts::default() };
    eprintln!("loading {} with {} threads ...", a.model, a.threads);
    let model = Model::load_with(&a.model, a.threads, &opts).map_err(|x| e(format!("load: {x}")))?;
    let rss_after_load = peak_rss_bytes();
    eprintln!(
        "loaded in {:.1} s: {} layers ({} pinned, {} streamed), {:.3} GB of weights held, peak RSS {}",
        model.load_secs,
        model.layers.len(),
        model.plan.pinned,
        model.plan.streamed(),
        model.weight_bytes as f64 / 1e9,
        rss_after_load.map(|b| format!("{:.3} GB", b as f64 / 1e9)).unwrap_or_else(|| "n/a".into())
    );
    if k > 0 && model.mtp.is_none() {
        return Err("--spec: this GGUF has no MTP block".into());
    }
    eprintln!("sampling: {}{}", a.sampler, if a.sampler_given.is_empty() { String::new() } else { format!(" (generation_config gives: {})", a.sampler_given.join(", ")) });
    let mut sampler = Sampler::new(a.sampler.clone(), model.vocab());
    let greedy = sampler.cfg.is_greedy();

    let mut state = if k > 0 { model.new_state_spec(k).map_err(|x| x.to_string())? } else { model.new_state() };
    // preallocate the KV caches and the attention score buffers for the whole run: after this the decode
    // step allocates nothing (Phase 3.6, `tests/decode_alloc.rs`)
    state.reserve(max_pos);
    let state_bytes = state.bytes();
    let mut logits = vec![0f32; model.vocab()];
    // prefill: batched (3.3) unless asked for the token-by-token path; logits only for the last position
    let t0 = Instant::now();
    let hidden = model.hidden();
    let first: u32 = if k > 0 {
        model.spec_prefill(&prompt_ids, &mut state, &mut sampler)
    } else {
        let last: Vec<f32> = if a.sequential_prefill {
            let mut h = Vec::new();
            for &id in &prompt_ids {
                h = model.forward_hidden(id, &mut state);
            }
            h
        } else {
            let hs = model.prefill(&prompt_ids, &mut state);
            hs[(prompt_ids.len() - 1) * hidden..].to_vec()
        };
        model.logits_of(&last, &mut logits);
        if greedy {
            argmax(&logits)
        } else {
            sampler.sample(&logits)
        }
    };
    let prefill_s = t0.elapsed().as_secs_f64();
    eprintln!("prefill ({}): {} tokens in {:.2} s ({:.2} tok/s)", if a.sequential_prefill { "sequential" } else { "batched" }, prompt_ids.len(), prefill_s, prompt_ids.len() as f64 / prefill_s);

    let mut out: Vec<u32> = Vec::new();
    let mut next = first;
    aqueduct_core::prof::reset();
    let stream_before = model.stream_stats();
    let t1 = Instant::now();
    let mut stopped = None;
    let mut step_secs: Vec<f64> = Vec::new();
    let mut disk_per_step: Vec<u64> = Vec::new();
    let mut round_accepted: Vec<usize> = Vec::new();
    if k == 0 {
        for _ in 0..a.max_tokens {
            if model.is_stop(next) {
                stopped = Some(next);
                break;
            }
            out.push(next);
            if out.len() == a.max_tokens {
                break;
            }
            let ts = Instant::now();
            let c0 = model.stream_stats().map_or(0, |s| s.consumed_bytes);
            model.forward_token(next, &mut state, &mut logits);
            let c1 = model.stream_stats().map_or(0, |s| s.consumed_bytes);
            // rule 4, checked here as well as inside the model: exactly the streamed layers, once
            assert_eq!(c1 - c0, model.streamed_bytes_per_pass(), "disk bytes this token");
            disk_per_step.push(c1 - c0);
            step_secs.push(ts.elapsed().as_secs_f64());
            next = if greedy { argmax(&logits) } else { sampler.sample(&logits) };
            if a.verbose {
                eprintln!("  token {} -> {} ({:.3} s, {} bytes from disk)", out.len(), next, step_secs.last().unwrap(), c1 - c0);
            }
        }
    } else {
        // speculative: `next` is the pending token; each round consumes it and emits m + 1 more
        'outer: loop {
            if model.is_stop(next) {
                stopped = Some(next);
                break;
            }
            out.push(next);
            if out.len() >= a.max_tokens {
                break;
            }
            let ts = Instant::now();
            let c0 = model.stream_stats().map_or(0, |s| s.consumed_bytes);
            let m = model.spec_round(&mut state, next, &mut sampler);
            let c1 = model.stream_stats().map_or(0, |s| s.consumed_bytes);
            // rule 5: one disk pass per verification round
            assert_eq!(c1 - c0, model.streamed_bytes_per_pass(), "disk bytes this round");
            disk_per_step.push(c1 - c0);
            step_secs.push(ts.elapsed().as_secs_f64());
            round_accepted.push(m);
            let emitted: Vec<u32> = state.spec.as_ref().unwrap().emitted.clone();
            if a.verbose {
                eprintln!("  round {}: accepted {m} of {k}, emitted {emitted:?} ({:.3} s, {} bytes from disk)", round_accepted.len(), step_secs.last().unwrap(), c1 - c0);
            }
            for &t in &emitted[..emitted.len() - 1] {
                if model.is_stop(t) {
                    stopped = Some(t);
                    break 'outer;
                }
                out.push(t);
                if out.len() >= a.max_tokens {
                    break 'outer;
                }
            }
            next = *emitted.last().unwrap();
        }
        out.truncate(a.max_tokens);
    }
    let decode_s = t1.elapsed().as_secs_f64();
    let steps = out.len().saturating_sub(1).max(1);
    let s_per_tok = decode_s / steps as f64;
    let ram_gbps = model.decode_bytes_per_token() as f64 / s_per_tok / 1e9;
    eprintln!(
        "decode: {} tokens, {:.3} s/token ({:.2} tok/s){}, {} threads{}",
        out.len(),
        s_per_tok,
        1.0 / s_per_tok,
        if k == 0 { format!(" = {ram_gbps:.2} GB/s of weights through the CPU") } else { format!(" over {} rounds of --spec {k}", step_secs.len()) },
        model.threads,
        stopped.map(|s| format!(", stopped at id {s}")).unwrap_or_default()
    );
    let spec_stats = state.spec.as_ref().map(|s| s.stats.clone());
    if let Some(s) = &spec_stats {
        eprintln!(
            "spec: {} rounds, {} of {} drafts accepted ({:.1} %), mean {:.3} accepted per round, {:.3} tokens per round; accepted at position {:?}; histogram {:?}; per round: snapshot {:.1} ms, verify {:.1} ms, replay {:.1} ms, MTP re-feed {:.1} ms, chained drafts {:.1} ms each ({} steps)",
            s.rounds,
            s.accepted,
            s.drafted,
            100.0 * s.acceptance_rate(),
            s.mean_accepted(),
            s.emitted as f64 / s.rounds.max(1) as f64,
            s.accepted_at,
            s.hist,
            s.snapshot_ns as f64 / 1e6 / s.rounds.max(1) as f64,
            s.verify_ns as f64 / 1e6 / s.rounds.max(1) as f64,
            s.replay_ns as f64 / 1e6 / s.rounds.max(1) as f64,
            s.refeed_ns as f64 / 1e6 / s.refeeds.max(1) as f64,
            s.chain_ns as f64 / 1e6 / s.chain_steps.max(1) as f64,
            s.chain_steps
        );
    }
    // streaming numbers over the decode phase
    let stream_after = model.stream_stats();
    // Overlap accounting over the decode phase: the consumer is either waiting for a slot (`wait`) or
    // computing (`compute = decode - wait`); the I/O thread is reading for `read`. With no overlap at all
    // the phase would take compute + read; what it saved is the overlap, which hides a share of the read
    // time (the spec's number) and a share of the compute (the meaningful one when the disk is the
    // bottleneck, as it is at every streamed rung here).
    let (disk_bytes_per_token, hidden_of_read, hidden_of_compute, wait_s, read_s, compute_s, overlap_s, read_gbps) = match (stream_before, stream_after) {
        (Some(b), Some(af)) => {
            let consumed = af.consumed_bytes - b.consumed_bytes;
            let wait = (af.wait_ns - b.wait_ns) as f64 / 1e9;
            let read = (af.read_ns - b.read_ns) as f64 / 1e9;
            let read_bytes = af.read_bytes - b.read_bytes;
            let compute = (decode_s - wait).max(0.0);
            let overlap = (compute + read - decode_s).max(0.0);
            let of_read = if read > 0.0 { overlap / read } else { 1.0 };
            let of_compute = if compute > 0.0 { overlap / compute } else { 1.0 };
            let gbps = if read > 0.0 { read_bytes as f64 / read / 1e9 } else { 0.0 };
            eprintln!(
                "disk: {:.3} GB per token from disk ({} bytes per pass = streamed layers exactly, asserted every pass; {} passes); reads at {:.2} GB/s; over the decode phase: reading {read:.2} s, computing {compute:.2} s, waiting {wait:.2} s, wall {decode_s:.2} s; overlap {overlap:.2} s = {:.1}% of the read time hidden, {:.1}% of the compute hidden",
                consumed as f64 / steps as f64 / 1e9,
                model.streamed_bytes_per_pass(),
                step_secs.len(),
                gbps,
                100.0 * of_read,
                100.0 * of_compute
            );
            (consumed / steps as u64, of_read, of_compute, wait, read, compute, overlap, gbps)
        }
        _ => (0, 1.0, 1.0, 0.0, 0.0, decode_s, 0.0, 0.0),
    };
    let expected = model.weight_bytes + state_bytes as u64 + (model.vocab() * 4) as u64;
    let peak = peak_rss_bytes();
    if let Some(rss) = peak {
        eprintln!(
            "peak RSS {:.3} GB vs planned {:.3} GB (weights held {:.3} GB + state {:.3} GB + logits); plan total {:.3} GB{}",
            rss as f64 / 1e9,
            expected as f64 / 1e9,
            model.weight_bytes as f64 / 1e9,
            state_bytes as f64 / 1e9,
            model.plan.total as f64 / 1e9,
            a.budget.map(|b| format!("; budget {:.3} GB: {}", b as f64 / 1e9, if rss <= b { "UNDER" } else { "OVER" })).unwrap_or_default()
        );
    }
    let job_peak = job_peak_memory();
    if let Some((pp, jp)) = job_peak {
        eprintln!("job object peak: process {:.3} GB, job {:.3} GB (committed memory, as the cap counts it)", pp as f64 / 1e9, jp as f64 / 1e9);
    }
    // the cost model, when the doctor numbers were given (the plain-token model; the spec round model is
    // fitted by the acceptance script from the numbers filed below)
    let bytes_disk = model.streamed_bytes_per_pass();
    let bytes_ram = model.decode_bytes_per_token() - bytes_disk;
    let predicted = match (a.membw, a.diskbw) {
        (Some(m), Some(d)) if m > 0.0 && d > 0.0 => {
            let p = bytes_ram as f64 / 1e9 / m + bytes_disk as f64 / 1e9 / d + NON_MATVEC_S;
            eprintln!("cost model (plain token): {:.3} GB/{m} GB/s + {:.3} GB/{d} GB/s + {NON_MATVEC_S} = {p:.3} s/token predicted vs {s_per_tok:.3} measured, ratio {:.2}", bytes_ram as f64 / 1e9, bytes_disk as f64 / 1e9, s_per_tok / p);
            Some(p)
        }
        _ => None,
    };
    if a.profile {
        eprintln!("# decode stage breakdown, {} threads, {steps} steps ({} s/token)", model.threads, format_args!("{s_per_tok:.3}"));
        eprint!("{}", aqueduct_core::prof::report(steps));
    }
    let ids_line = out.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    if let Some(path) = &a.stats {
        let spec_json = match &spec_stats {
            Some(s) => format!(
                ",\"spec_k\":{k},\"spec_rounds\":{},\"spec_drafted\":{},\"spec_accepted\":{},\"spec_emitted\":{},\"spec_accepted_at\":{},\"spec_hist\":{},\"spec_mean_accepted\":{:.4},\"spec_acceptance_rate\":{:.4},\"spec_snapshot_s\":{:.4},\"spec_verify_s\":{:.4},\"spec_replay_s\":{:.4},\"spec_refeed_s\":{:.4},\"spec_refeeds\":{},\"spec_chain_s\":{:.4},\"spec_chain_steps\":{},\"round_accepted\":{}",
                s.rounds,
                s.drafted,
                s.accepted,
                s.emitted,
                json_list(&s.accepted_at),
                json_list(&s.hist),
                s.mean_accepted(),
                s.acceptance_rate(),
                s.snapshot_ns as f64 / 1e9,
                s.verify_ns as f64 / 1e9,
                s.replay_ns as f64 / 1e9,
                s.refeed_ns as f64 / 1e9,
                s.refeeds,
                s.chain_ns as f64 / 1e9,
                s.chain_steps,
                json_list(&round_accepted)
            ),
            None => format!(",\"spec_k\":{k}"),
        };
        let js = format!(
            "{{\"budget\":{},\"job_limit\":{},\"pinned\":{},\"streamed\":{},\"n_slots\":{},\"slot_bytes\":{},\"plan_total\":{},\"plan_resident\":{},\"sector\":{},\"large_pages\":{},\"large_page_note\":{:?},\"threads\":{},\"qd\":{},\"load_s\":{:.3},\"load_read_bytes\":{},\"load_read_s\":{:.3},\"prompt_len\":{},\"n_generated\":{},\"steps\":{},\"prefill_s\":{:.3},\"decode_s\":{:.3},\"s_per_token\":{:.4},\"step_secs\":{},\"decode_bytes_per_token\":{},\"ram_gbps\":{:.3},\"disk_bytes_per_token\":{},\"streamed_bytes_per_pass\":{},\"disk_per_step\":{},\"read_gbps\":{:.3},\"prefetch_hidden\":{:.4},\"hidden_of_compute\":{:.4},\"wait_s\":{:.3},\"read_s\":{:.3},\"compute_s\":{:.3},\"overlap_s\":{:.3},\"peak_rss\":{},\"peak_rss_after_load\":{},\"job_peak_process\":{},\"job_peak_job\":{},\"planned_rss\":{},\"state_bytes\":{},\"weight_bytes\":{},\"membw\":{},\"diskbw\":{},\"predicted_s_per_token\":{},\"sampling\":{:?},\"seed\":{}{},\"ids\":{}}}\n",
            a.budget.map_or("null".to_string(), |b| b.to_string()),
            a.job_limit.map_or("null".to_string(), |b| b.to_string()),
            model.plan.pinned,
            model.plan.streamed(),
            model.plan.n_slots,
            model.plan.slot_bytes,
            model.plan.total,
            model.plan.resident_bytes,
            model.sector,
            model.large_pages_used,
            large_page_note().unwrap_or(""),
            model.threads,
            a.qd,
            model.load_secs,
            model.load_read_bytes,
            model.load_read_secs,
            prompt_ids.len(),
            out.len(),
            steps,
            prefill_s,
            decode_s,
            s_per_tok,
            json_list(&step_secs.iter().map(|s| format!("{s:.4}")).collect::<Vec<_>>()),
            model.decode_bytes_per_token(),
            ram_gbps,
            disk_bytes_per_token,
            model.streamed_bytes_per_pass(),
            json_list(&disk_per_step),
            read_gbps,
            hidden_of_read,
            hidden_of_compute,
            wait_s,
            read_s,
            compute_s,
            overlap_s,
            peak.map_or("null".to_string(), |b| b.to_string()),
            rss_after_load.map_or("null".to_string(), |b| b.to_string()),
            job_peak.map_or("null".to_string(), |(p, _)| p.to_string()),
            job_peak.map_or("null".to_string(), |(_, j)| j.to_string()),
            expected,
            state_bytes,
            model.weight_bytes,
            a.membw.map_or("null".to_string(), |m| m.to_string()),
            a.diskbw.map_or("null".to_string(), |d| d.to_string()),
            predicted.map_or("null".to_string(), |p| format!("{p:.4}")),
            a.sampler.to_string(),
            a.sampler.seed,
            spec_json,
            json_list(&out),
        );
        std::fs::write(path, js).map_err(|x| e(format!("write {path}: {x}")))?;
    }
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
