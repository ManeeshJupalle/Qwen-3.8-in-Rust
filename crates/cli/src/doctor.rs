//! `aqueduct doctor` (Phase 4.3): measure this machine the way the engine uses it, then print the memory
//! plan and the predicted s/token for it.
//!
//! RAM and cores from the OS; memory bandwidth as the short form of `bench membw` (1 GiB, best of 3, at the
//! physical-core thread count the engine runs with); the disk holding the model file measured with the
//! engine's own read path: unbuffered (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`) sequential reads of
//! `largest_layer_bytes`, at queue depth 1, 2 and 4 (overlapped 32 MiB requests), best of 5 with every run
//! printed, each of the 5 at a different layer's offset so a drive's own cache cannot replay the same
//! range; then one "cold-ish" read after 2 GiB of other data has been pulled through the drive. No buffered
//! throughput anywhere: it reports the page cache, not the drive. Then the plan (`--budget`, default the
//! free RAM the OS reports) and the cost model
//! `t = bytes_ram / membw + bytes_disk / diskbw + 0.053` at the ladder budgets. Output is filed to
//! `docs/data/doctor_<machine>.txt` (`--out`).

use std::time::Instant;

use aqueduct_core::os::{align_down, align_up, arena_bytes, mem_status, AlignedBuf, DirectFile, DEFAULT_CHUNK};
use aqueduct_core::tier::{group, parse_bytes, MemoryPlan, PlanInput, PlanParams};
use aqueduct_core::{logical_cores, physical_cores, Gguf, ModelConfig};

use crate::bench::membw_best;
use crate::run::{DEFAULT_GGUF, NON_MATVEC_S};

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1))
}

fn machine_name() -> String {
    std::env::var("COMPUTERNAME").or_else(|_| std::env::var("HOSTNAME")).unwrap_or_else(|_| "unknown".into()).to_lowercase()
}

struct Out(String);
impl Out {
    fn line(&mut self, s: String) {
        println!("{s}");
        self.0.push_str(&s);
        self.0.push('\n');
    }
}

/// Time `n` reads of `len` bytes at the given offsets with a fresh handle at `qd`; returns GB/s per read.
fn time_reads(path: &std::path::Path, qd: usize, offsets: &[u64], len: usize, buf: &AlignedBuf) -> Result<Vec<f64>, String> {
    let mut f = DirectFile::open(path, DEFAULT_CHUNK, qd).map_err(|e| format!("open unbuffered: {e}"))?;
    let mut out = Vec::with_capacity(offsets.len());
    for &off in offsets {
        let t0 = Instant::now();
        let got = f.read_at(off, buf, len).map_err(|e| format!("read at {off}: {e}"))?;
        let s = t0.elapsed().as_secs_f64();
        out.push(got as f64 / s / 1e9);
    }
    Ok(out)
}

pub fn doctor(args: &[String]) -> Result<(), String> {
    let model_path = arg(args, "--model").cloned().unwrap_or_else(|| DEFAULT_GGUF.to_string());
    let path = std::path::Path::new(&model_path);
    let runs: usize = arg(args, "--runs").map(|s| s.parse().map_err(|e| format!("--runs: {e}"))).transpose()?.unwrap_or(5);
    let max_pos: u64 = arg(args, "--max-pos").map(|s| s.parse().map_err(|e| format!("--max-pos: {e}"))).transpose()?.unwrap_or(4096);
    let n_slots: usize = arg(args, "--slots").map(|s| s.parse().map_err(|e| format!("--slots: {e}"))).transpose()?.unwrap_or(2);
    let budget_arg = arg(args, "--budget").map(|s| parse_bytes(s)).transpose()?;
    let machine = machine_name();
    let out_path = arg(args, "--out").cloned().unwrap_or_else(|| format!("docs/data/doctor_{machine}.txt"));
    let mut o = Out(String::new());

    o.line(format!("# aqueduct doctor: {machine}, {} {}, {}", std::env::consts::OS, std::env::consts::ARCH, chrono_like_now()));
    o.line(format!("# model: {model_path}"));
    let (physical, logical) = (physical_cores(), logical_cores());
    let (total, avail) = mem_status().ok_or("memory status unavailable")?;
    o.line(format!("cores      : {physical} physical, {logical} logical (the engine runs {physical} threads)"));
    o.line(format!("RAM        : {:.2} GiB total, {:.2} GiB free now ({} / {} bytes)", total as f64 / (1u64 << 30) as f64, avail as f64 / (1u64 << 30) as f64, group(total), group(avail)));

    // ---- membw, short form
    let (membw, membw_runs) = membw_best(1, 3, physical)?;
    o.line(format!("membw      : {membw:.2} GB/s at {physical} threads (1 GiB sum-reduce, best of 3: {})", membw_runs.iter().map(|g| format!("{g:.2}")).collect::<Vec<_>>().join(" ")));

    // ---- the model's layout
    let g = Gguf::open(path).map_err(|e| format!("open {model_path}: {e}"))?;
    let cfg = ModelConfig::from_gguf(&g).map_err(|e| format!("config: {e}"))?;
    let input = PlanInput::new(&g, &cfg);
    let probe = DirectFile::open(path, DEFAULT_CHUNK, 1).map_err(|e| format!("open unbuffered: {e}"))?;
    let sector = probe.sector as u64;
    let file_size = probe.file_size;
    drop(probe);
    let (largest_layer, largest_bytes) = (0..cfg.n_layer).map(|i| (i, input.layer_bytes(i as usize))).max_by_key(|&(_, b)| b).unwrap();
    let read_len = align_up(largest_bytes, sector) as usize;
    let drive = path.to_string_lossy().chars().take(2).collect::<String>();
    o.line(format!("disk       : {drive} sector {sector} bytes (queried); largest layer {largest_layer}: {} bytes = {:.2} MiB; read size {} bytes; chunk {} MiB", group(largest_bytes), largest_bytes as f64 / 1048576.0, group(read_len as u64), DEFAULT_CHUNK >> 20));
    // five offsets, each a different layer's start (spaced through the file), all inside the file
    let mut offsets: Vec<u64> = Vec::new();
    let step = (cfg.n_layer as usize / runs).max(1);
    for j in (0..cfg.n_layer as usize).step_by(step).take(runs) {
        let off = align_down(input.layer_span[j].0, sector);
        if off + read_len as u64 <= align_up(file_size, sector) {
            offsets.push(off);
        }
    }
    while offsets.len() < runs {
        offsets.push(0);
    }
    let buf = AlignedBuf::new(arena_bytes(largest_bytes, sector) as usize, false).map_err(|e| format!("buffer: {e}"))?;
    let mut best: Vec<(usize, f64)> = Vec::new();
    for qd in [1usize, 2, 4] {
        let gbs = time_reads(path, qd, &offsets, read_len, &buf)?;
        let b = gbs.iter().cloned().fold(0f64, f64::max);
        let w = gbs.iter().cloned().fold(f64::INFINITY, f64::min);
        o.line(format!("disk qd{qd}   : best {b:.2} GB/s, worst {w:.2}; runs {} (unbuffered, {} reads of {:.1} MiB at layers {:?})", gbs.iter().map(|g| format!("{g:.2}")).collect::<Vec<_>>().join(" "), gbs.len(), read_len as f64 / 1048576.0, (0..cfg.n_layer as usize).step_by(step).take(runs).collect::<Vec<_>>()));
        best.push((qd, b));
    }
    // cold-ish: pull 2 GiB of other data through the drive, then one read of layer 0 at the engine's qd
    {
        let other_start = align_down(file_size / 2, sector);
        let mut pulled = 0u64;
        let mut f = DirectFile::open(path, DEFAULT_CHUNK, 2).map_err(|e| format!("open unbuffered: {e}"))?;
        let t0 = Instant::now();
        while pulled < (2u64 << 30) && other_start + pulled + read_len as u64 <= align_up(file_size, sector) {
            f.read_at(other_start + pulled, &buf, read_len).map_err(|e| format!("read: {e}"))?;
            pulled += read_len as u64;
        }
        let other_s = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let got = f.read_at(offsets[0], &buf, read_len).map_err(|e| format!("read: {e}"))?;
        let s = t0.elapsed().as_secs_f64();
        o.line(format!("disk cold  : {:.2} GB/s at qd2 for layer {} after {:.2} GiB of other data ({:.2} GB/s while pulling it)", got as f64 / s / 1e9, (0..cfg.n_layer as usize).step_by(step).next().unwrap_or(0), pulled as f64 / (1u64 << 30) as f64, pulled as f64 / other_s / 1e9));
    }
    let diskbw_qd1 = best[0].1;
    let diskbw_qd2 = best[1].1;
    o.line(format!("large pages: {}", match aqueduct_core::os::try_enable_lock_memory_privilege() {
        Ok(()) => format!("privilege held, page {} bytes", aqueduct_core::os::large_page_minimum()),
        Err(r) => format!("unavailable: {r}"),
    }));

    // ---- the plan for this machine and the cost model at the ladder budgets
    let budget = budget_arg.unwrap_or(avail);
    o.line(format!("plan for this machine (budget {}: {}):", if budget_arg.is_some() { "--budget" } else { "free RAM now" }, group(budget)));
    let plan = MemoryPlan::compute(&input, &PlanParams::new(Some(budget), max_pos, n_slots, sector));
    for l in plan.table().lines() {
        o.line(format!("  {l}"));
    }
    if let Err(e) = plan.check() {
        o.line(format!("  REFUSED: {e}"));
    }
    let layer_total: u64 = (0..cfg.n_layer as usize).map(|i| input.layer_bytes(i)).sum();
    let head: u64 = input.non_layer.iter().filter(|(n, _)| n == "output.weight" || n == "output_norm.weight").map(|(_, b)| *b).sum();
    let embed_row = input.non_layer.iter().find(|(n, _)| n == "token_embd.weight").map(|(_, b)| b / input.vocab).unwrap_or(0);
    let decode_bytes = layer_total + head + embed_row;
    o.line(format!("cost model : t = bytes_ram / {membw:.2} GB/s + bytes_disk / {diskbw_qd2:.2} GB/s (qd2; qd1 {diskbw_qd1:.2}) + {NON_MATVEC_S} s; bytes per token through the CPU {:.3} GB", decode_bytes as f64 / 1e9));
    o.line(format!("  {:>10} {:>7} {:>9} {:>14} {:>14} {:>12} {:>12}", "budget", "pinned", "streamed", "GB disk/token", "GB ram/token", "pred s/tok", "pred tok/s"));
    let mut budgets: Vec<(String, Option<u64>)> = [6u64, 8, 12, 16, 32].iter().map(|g| (format!("{g} GiB"), Some(g << 30))).collect();
    budgets.push((format!("{:.1} GiB free", avail as f64 / (1u64 << 30) as f64), Some(avail)));
    budgets.push(("resident".into(), None));
    for (name, b) in budgets {
        let p = MemoryPlan::compute(&input, &PlanParams::new(b, max_pos, n_slots, sector));
        if p.check().is_err() {
            o.line(format!("  {name:>10} refused: needs {} bytes", group(p.minimum)));
            continue;
        }
        let disk = p.streamed_bytes_per_pass;
        let ram = decode_bytes - disk;
        let t = ram as f64 / 1e9 / membw + disk as f64 / 1e9 / diskbw_qd2 + NON_MATVEC_S;
        o.line(format!("  {name:>10} {:>7} {:>9} {:>14.3} {:>14.3} {:>12.3} {:>12.2}", p.pinned, p.streamed(), disk as f64 / 1e9, ram as f64 / 1e9, t, 1.0 / t));
    }
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&out_path, &o.0).map_err(|e| format!("write {out_path}: {e}"))?;
    eprintln!("filed to {out_path}");
    Ok(())
}

/// Date and time without a crate: seconds since the epoch, good enough to order files.
fn chrono_like_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // civil date from days (Howard Hinnant's algorithm)
    let days = (secs / 86400) as i64;
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    let t = secs % 86400;
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", t / 3600, (t % 3600) / 60)
}
