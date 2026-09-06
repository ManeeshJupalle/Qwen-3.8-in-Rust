//! `aqueduct doctor` (Phase 4.3, rewritten as the front door in Phase 6): measure this machine the way the
//! engine uses it, plan the model into its free RAM, predict the token time from the cost model, and say in
//! one line whether to download the 17.8 GB file.
//!
//! Runs with or without the model. What it measures: physical cores and AVX2/F16C, RAM and free RAM,
//! memory bandwidth (the short form of `bench membw`: 1 GiB, best of 3, at the physical-core thread count
//! the engine runs with), the drive under the model path (its bus and seek penalty queried from the OS,
//! `disk.rs`; then unbuffered sequential reads at queue depth 1 and 2 with the engine's own read path: of the
//! model's layers when the file is there, of a 1 GiB scratch file on the same volume when it is not), and
//! whether large pages are available. Then the memory plan for the free RAM (from the file's header when
//! present, from the built-in layout table of the supported file when not, `known.rs`), the predicted
//! s/token plain and with `--spec 3` at the ladder budgets, a verdict, and the exact next commands.
//!
//! Cost model (docs/ladder.md, docs/data/spec_cost_model.txt):
//! `t_plain = bytes_ram / membw + bytes_disk / diskbw + 0.053`,
//! `t_round = t_plain + c_mtp x k + c_verify x (k + 1)`, `t_spec = t_round / tokens_per_round`.
//! The plain model is within 6 % where the disk dominates and 1.2 to 1.9 x optimistic where RAM does; the
//! round model overstates streamed rounds by 10 to 25 %. Nothing here is a measurement of the model running.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use aqueduct_core::kernels::simd::avx2_detected;
use aqueduct_core::os::{align_down, align_up, arena_bytes, mem_status, AlignedBuf, DirectFile, DEFAULT_CHUNK};
use aqueduct_core::tier::{group, parse_bytes, MemoryPlan, PlanInput, PlanParams};
use aqueduct_core::{logical_cores, physical_cores, Gguf, ModelConfig};

use crate::bench::membw_best;
use crate::disk::{disk_info, nearest_existing, DiskClass, DiskInfo};
use crate::known;
use crate::run::{DEFAULT_GGUF, DEFAULT_SPEC_K, NON_MATVEC_S};

/// The speculative round model fitted on the Phase 5.5 ladder (`docs/data/spec_cost_model.txt`,
/// `tools/spec_cost_model.py`): the chained MTP draft step and the marginal verify row, both resident.
pub const SPEC_C_MTP_S: f64 = 0.085;
pub const SPEC_C_VERIFY_S: f64 = 0.527;
/// Tokens a `--spec 3` round emits on the ladder prompts (63 % acceptance), same file.
pub const SPEC_TOKENS_PER_ROUND: f64 = 2.74;

/// Unbuffered read speed assumed for a drive class when the measurement could not run (GB/s); each is
/// labelled "assumed" wherever it is used. NVMe is this laptop's qd1 number (docs/data/doctor_maneesh-msi.txt).
fn assumed_gbps(class: DiskClass) -> f64 {
    match class {
        DiskClass::Nvme => 2.95,
        DiskClass::Ssd => 0.5,
        DiskClass::Hdd => 0.12,
        DiskClass::Unknown => 0.5,
    }
}

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

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// Time `offsets.len()` reads of `len` bytes with a fresh handle at `qd`; returns GB/s per read.
fn time_reads(path: &Path, qd: usize, offsets: &[u64], len: usize, buf: &AlignedBuf) -> Result<Vec<f64>, String> {
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

fn fmt_runs(v: &[f64]) -> String {
    v.iter().map(|g| format!("{g:.2}")).collect::<Vec<_>>().join(" ")
}

/// A scratch file that is removed when dropped (also on an error path).
struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Write `bytes` of pseudo-random data to `path` (buffered), flushed to the device.
fn write_scratch(path: &Path, bytes: u64) -> Result<(), String> {
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let chunk = 32usize << 20;
    let mut buf = vec![0u8; chunk];
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut left = bytes;
    while left > 0 {
        for w in buf.chunks_exact_mut(8) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            w.copy_from_slice(&x.to_le_bytes());
        }
        let n = (left as usize).min(chunk);
        f.write_all(&buf[..n]).map_err(|e| format!("write {}: {e}", path.display()))?;
        left -= n as u64;
    }
    f.sync_all().map_err(|e| format!("flush {}: {e}", path.display()))?;
    Ok(())
}

/// Where a scratch file for the disk measurement can go: the model's directory, its nearest existing
/// ancestor, and the temp directory if it is on the same volume (first that accepts a write wins).
fn scratch_candidates(model: &Path, volume: &str) -> Vec<PathBuf> {
    let mut v = Vec::new();
    let anc = nearest_existing(model);
    let dir = if anc.is_dir() { anc } else { anc.parent().map(Path::to_path_buf).unwrap_or(anc) };
    v.push(dir);
    let tmp = std::env::temp_dir();
    if disk_info(&tmp).volume == volume {
        v.push(tmp);
    }
    v
}

/// The plan and the two predictions (plain, `--spec 3`) at one budget.
struct Prediction {
    pinned: u32,
    pinned_spec: u32,
    streamed: u32,
    disk_gb: f64,
    ram_gb: f64,
    t_plain: f64,
    t_spec: f64,
    fits: bool,
    minimum: u64,
}

/// What every prediction shares: the plan parameters and the measured rates.
struct CostEnv {
    max_pos: u64,
    n_slots: usize,
    sector: u64,
    /// Bytes a plain token moves through the CPU (every layer, the head, one embedding row).
    decode_bytes: u64,
    membw: f64,
    diskbw: f64,
}

fn predict(input: &PlanInput, budget: Option<u64>, env: &CostEnv) -> Prediction {
    let CostEnv { max_pos, n_slots, sector, decode_bytes, membw, diskbw } = *env;
    let p = MemoryPlan::compute(input, &PlanParams::new(budget, max_pos, n_slots, sector));
    let mut params = PlanParams::new(budget, max_pos, n_slots, sector);
    params.spec_k = DEFAULT_SPEC_K as u64;
    let ps = MemoryPlan::compute(input, &params);
    let fits = p.check().is_ok();
    let disk = p.streamed_bytes_per_pass;
    let ram = decode_bytes - disk;
    let t_plain = ram as f64 / 1e9 / membw + disk as f64 / 1e9 / diskbw + NON_MATVEC_S;
    // the spec plan pins one layer fewer, so its disk pass is a little longer
    let disk_s = ps.streamed_bytes_per_pass;
    let ram_s = decode_bytes - disk_s;
    let t_plain_s = ram_s as f64 / 1e9 / membw + disk_s as f64 / 1e9 / diskbw + NON_MATVEC_S;
    let k = DEFAULT_SPEC_K as f64;
    let t_round = t_plain_s + SPEC_C_MTP_S * k + SPEC_C_VERIFY_S * (k + 1.0);
    Prediction { pinned: p.pinned, pinned_spec: ps.pinned, streamed: p.streamed(), disk_gb: disk as f64 / 1e9, ram_gb: ram as f64 / 1e9, t_plain, t_spec: t_round / SPEC_TOKENS_PER_ROUND, fits, minimum: p.minimum.max(ps.minimum) }
}

/// sha256 of a file (buffered reads), lower-case hex.
fn sha256_file(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 32 << 20];
    loop {
        let n = std::io::Read::read(&mut f, &mut buf).map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The download command: `curl.exe` on Windows, because in Windows PowerShell 5.1 `curl` is an alias of
/// `Invoke-WebRequest` and rejects these flags; `curl` everywhere else.
fn shell_download(url: &str, dest: &str) -> String {
    format!("{} -L --create-dirs -o {} {url}", if cfg!(windows) { "curl.exe" } else { "curl" }, quote(dest))
}

/// A byte count as a `--budget` argument: `17G`, or `17.5G` for a half.
fn size_flag(b: u64) -> String {
    let g = b as f64 / (1u64 << 30) as f64;
    if (g - g.floor()).abs() < 1e-9 {
        format!("{}G", g as u64)
    } else {
        format!("{g:.1}G")
    }
}

fn quote(p: &str) -> String {
    if p.contains(' ') {
        format!("\"{p}\"")
    } else {
        p.to_string()
    }
}

pub fn doctor(args: &[String]) -> Result<(), String> {
    let model_path = arg(args, "--model").cloned().unwrap_or_else(|| DEFAULT_GGUF.to_string());
    let path = Path::new(&model_path);
    let runs: usize = arg(args, "--runs").map(|s| s.parse().map_err(|e| format!("--runs: {e}"))).transpose()?.unwrap_or(5).max(1);
    let max_pos: u64 = arg(args, "--max-pos").map(|s| s.parse().map_err(|e| format!("--max-pos: {e}"))).transpose()?.unwrap_or(4096);
    let n_slots: usize = arg(args, "--slots").map(|s| s.parse().map_err(|e| format!("--slots: {e}"))).transpose()?.unwrap_or(2);
    let budget_arg = arg(args, "--budget").map(|s| parse_bytes(s)).transpose()?;
    let out_path = arg(args, "--out").cloned();
    let want_sha = args.iter().any(|a| a == "--sha256");
    let tokenizer = arg(args, "--tokenizer").cloned().unwrap_or_else(|| crate::DEFAULT_TOKENIZER.to_string());
    let template = arg(args, "--template").cloned().unwrap_or_else(|| crate::chat::DEFAULT_TEMPLATE.to_string());
    let gen_config = arg(args, "--gen-config").cloned().unwrap_or_else(|| crate::chat::DEFAULT_GEN_CONFIG.to_string());
    let machine = machine_name();
    let mut o = Out(String::new());

    o.line(format!("# aqueduct doctor {}: {machine}, {} {}, {}", env!("CARGO_PKG_VERSION"), std::env::consts::OS, std::env::consts::ARCH, chrono_like_now()));
    o.line(format!("# model path: {model_path}"));
    o.line(String::new());
    o.line("== this machine".into());

    // ---- CPU
    let (physical, logical) = (physical_cores(), logical_cores());
    let avx2 = avx2_detected();
    o.line(format!("cpu        : {physical} physical cores, {logical} hardware threads (the engine runs {physical} threads for the matvec, {logical} for batches); AVX2+F16C: {}", if avx2 { "yes" } else { "NO (the kernels need it)" }));

    // ---- RAM
    let (total, avail) = mem_status().ok_or("memory status unavailable")?;
    o.line(format!("RAM        : {:.2} GiB total, {:.2} GiB free now ({} / {} bytes)", gib(total), gib(avail), group(total), group(avail)));

    // ---- membw
    let (membw, membw_runs) = membw_best(1, 3, physical)?;
    o.line(format!("membw      : {membw:.2} GB/s at {physical} threads (1 GiB sum-reduce, best of 3: {})", fmt_runs(&membw_runs)));

    // ---- the model file
    let present = path.is_file();
    let file_size = if present { std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) } else { 0 };
    let is_known_name = path.file_name().map(|n| n == known::FILE_NAME).unwrap_or(false);
    let mut model_ok = false;
    let mut sha_line: Option<String> = None;
    let (input, sector_hint): (PlanInput, Option<u64>) = if present {
        let size_ok = file_size == known::FILE_SIZE;
        model_ok = size_ok && is_known_name;
        o.line(format!(
            "model      : present, {} bytes ({:.2} GiB){}",
            group(file_size),
            gib(file_size),
            if size_ok { "; the size of the supported file".to_string() } else { format!("; NOT the supported file's size ({} bytes): a different GGUF", group(known::FILE_SIZE)) }
        ));
        if want_sha {
            let t0 = Instant::now();
            match sha256_file(path) {
                Ok(h) => {
                    let ok = h == known::SHA256;
                    model_ok = ok;
                    let what = if ok { "the supported file, verified".to_string() } else { format!("MISMATCH: expected {}", known::SHA256) };
                    sha_line = Some(format!("sha256     : {h} = {what} ({:.0} s)", t0.elapsed().as_secs_f64()));
                }
                Err(e) => sha_line = Some(format!("sha256     : could not hash: {e}")),
            }
            o.line(sha_line.clone().unwrap());
        } else {
            o.line(format!("sha256     : not checked (aqueduct doctor --sha256 reads the whole file; expected {})", known::SHA256));
        }
        let g = Gguf::open(path).map_err(|e| format!("open {model_path}: {e}"))?;
        let cfg = ModelConfig::from_gguf(&g).map_err(|e| format!("config: {e}"))?;
        (PlanInput::new(&g, &cfg), None)
    } else {
        o.line(format!("model      : absent at {model_path}; planning with the built-in layout of {} ({} bytes = {:.2} GiB)", known::FILE_NAME, group(known::FILE_SIZE), gib(known::FILE_SIZE)));
        (known::plan_input(), None)
    };
    let _ = sector_hint;

    // ---- the drive under the model path
    let info: DiskInfo = disk_info(path);
    o.line(format!("disk       : {}", info.describe()));
    let mut sector: u64 = 4096;
    let mut sector_queried = false;
    let mut diskbw_qd1: Option<f64> = None;
    let mut diskbw_qd2: Option<f64> = None;
    if present {
        let probe = DirectFile::open(path, DEFAULT_CHUNK, 1).map_err(|e| format!("open unbuffered: {e}"))?;
        sector = probe.sector as u64;
        sector_queried = true;
        drop(probe);
        let (largest_layer, largest_bytes) = (0..input.n_layer).map(|i| (i, input.layer_bytes(i as usize))).max_by_key(|&(_, b)| b).unwrap();
        let read_len = align_up(largest_bytes, sector) as usize;
        o.line(format!("disk read  : sector {sector} bytes (queried); unbuffered reads of the largest layer ({largest_layer}: {} bytes = {:.1} MiB), {} MiB requests", group(largest_bytes), largest_bytes as f64 / 1048576.0, DEFAULT_CHUNK >> 20));
        let mut offsets: Vec<u64> = Vec::new();
        let step = (input.n_layer as usize / runs).max(1);
        let at: Vec<usize> = (0..input.n_layer as usize).step_by(step).take(runs).collect();
        for &j in &at {
            let off = align_down(input.layer_span[j].0, sector);
            if off + read_len as u64 <= align_up(file_size, sector) {
                offsets.push(off);
            }
        }
        while offsets.len() < runs {
            offsets.push(0);
        }
        let buf = AlignedBuf::new(arena_bytes(largest_bytes, sector) as usize, false).map_err(|e| format!("buffer: {e}"))?;
        for qd in [1usize, 2] {
            let gbs = time_reads(path, qd, &offsets, read_len, &buf)?;
            let b = gbs.iter().cloned().fold(0f64, f64::max);
            let w = gbs.iter().cloned().fold(f64::INFINITY, f64::min);
            o.line(format!("disk qd{qd}   : best {b:.2} GB/s, worst {w:.2}; runs {} ({} reads at layers {at:?})", fmt_runs(&gbs), gbs.len()));
            if qd == 1 {
                diskbw_qd1 = Some(b);
            } else {
                diskbw_qd2 = Some(b);
            }
        }
        // cold-ish: 2 GiB of other data through the drive, then layer 0 at the engine's queue depth
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
        o.line(format!("disk cold  : {:.2} GB/s at qd2 for layer {} after {:.2} GiB of other data ({:.2} GB/s while pulling it)", got as f64 / s / 1e9, at[0], gib(pulled), pulled as f64 / other_s / 1e9));
    } else {
        // the same reads on a 1 GiB scratch file on the volume the model would land on
        let scratch_bytes = 1u64 << 30;
        let read_len = 256usize << 20;
        let mut measured = false;
        for dir in scratch_candidates(path, &info.volume) {
            let file = dir.join("aqueduct-doctor-scratch.bin");
            let guard = Scratch(file.clone());
            if let Err(e) = write_scratch(&file, scratch_bytes) {
                o.line(format!("disk read  : could not write a 1 GiB scratch file at {}: {e}", file.display()));
                drop(guard);
                continue;
            }
            let probe = match DirectFile::open(&file, DEFAULT_CHUNK, 1) {
                Ok(p) => p,
                Err(e) => {
                    o.line(format!("disk read  : could not open {} unbuffered: {e}", file.display()));
                    continue;
                }
            };
            sector = probe.sector as u64;
            sector_queried = true;
            drop(probe);
            let offsets: Vec<u64> = (0..4).map(|i| i * read_len as u64).collect();
            let buf = AlignedBuf::new(read_len + sector as usize, false).map_err(|e| format!("buffer: {e}"))?;
            o.line(format!("disk read  : sector {sector} bytes (queried); unbuffered reads of 256 MiB from a 1 GiB scratch file at {} ({} MiB requests)", file.display(), DEFAULT_CHUNK >> 20));
            let mut ok = true;
            for qd in [1usize, 2] {
                match time_reads(&file, qd, &offsets, read_len, &buf) {
                    Ok(gbs) => {
                        let b = gbs.iter().cloned().fold(0f64, f64::max);
                        let w = gbs.iter().cloned().fold(f64::INFINITY, f64::min);
                        o.line(format!("disk qd{qd}   : best {b:.2} GB/s, worst {w:.2}; runs {}", fmt_runs(&gbs)));
                        if qd == 1 {
                            diskbw_qd1 = Some(b);
                        } else {
                            diskbw_qd2 = Some(b);
                        }
                    }
                    Err(e) => {
                        o.line(format!("disk qd{qd}   : failed: {e}"));
                        ok = false;
                    }
                }
            }
            drop(guard);
            if ok {
                measured = true;
                break;
            }
        }
        if !measured {
            o.line(format!("disk read  : NOT measured; assuming {:.2} GB/s for a {:?} drive (the prediction below is only as good as that guess)", assumed_gbps(info.class), info.class));
        }
    }
    if !sector_queried {
        o.line(format!("disk sector: assumed {sector} bytes (nothing could be opened unbuffered on that volume)"));
    }
    let diskbw = diskbw_qd2.or(diskbw_qd1).unwrap_or_else(|| assumed_gbps(info.class));
    let disk_assumed = diskbw_qd2.is_none() && diskbw_qd1.is_none();

    // ---- large pages
    o.line(format!("large pages: {}", match aqueduct_core::os::try_enable_lock_memory_privilege() {
        Ok(()) => format!("privilege held, page {} bytes", aqueduct_core::os::large_page_minimum()),
        Err(r) => format!("unavailable: {r}"),
    }));

    // ---- the plan for this machine: --budget as given, or the free RAM rounded down to 0.5 GiB, which is
    // exactly the budget the next command will name, so the verdict and the command agree
    let budget = match budget_arg {
        Some(b) => b,
        None => (avail / (1u64 << 29)).max(1) * (1u64 << 29),
    };
    let model_bytes = if present { file_size } else { known::FILE_SIZE };
    o.line(String::new());
    o.line(match budget_arg {
        Some(_) => format!("== the plan for --budget {} ({} bytes = {:.2} GiB)", size_flag(budget), group(budget), gib(budget)),
        None => format!("== the plan for this machine's free RAM now, {:.2} GiB, rounded down to --budget {} ({} bytes)", gib(avail), size_flag(budget), group(budget)),
    });
    let plan = MemoryPlan::compute(&input, &PlanParams::new(Some(budget), max_pos, n_slots, sector));
    for l in plan.table().lines() {
        o.line(format!("  {l}"));
    }
    let plan_err = plan.check().err();
    if let Some(e) = &plan_err {
        o.line(format!("  REFUSED: {e}"));
    }

    // ---- the cost model at the ladder budgets
    let layer_total: u64 = (0..input.n_layer as usize).map(|i| input.layer_bytes(i)).sum();
    let head: u64 = input.non_layer.iter().filter(|(n, _)| n == "output.weight" || n == "output_norm.weight").map(|(_, b)| *b).sum();
    let embed_row = input.non_layer.iter().find(|(n, _)| n == "token_embd.weight").map(|(_, b)| b / input.vocab).unwrap_or(0);
    let decode_bytes = layer_total + head + embed_row;
    o.line(String::new());
    o.line(format!(
        "== predicted seconds per token (cost model, not a measurement): plain t = bytes_ram / {membw:.2} GB/s + bytes_disk / {diskbw:.2} GB/s{} + {NON_MATVEC_S} s; --spec {DEFAULT_SPEC_K}: (t + {SPEC_C_MTP_S} x {DEFAULT_SPEC_K} + {SPEC_C_VERIFY_S} x {}) / {SPEC_TOKENS_PER_ROUND} tokens per round",
        if disk_assumed { " (assumed)" } else { "" },
        DEFAULT_SPEC_K + 1
    ));
    o.line(format!("   bytes per token through the CPU: {:.3} GB; on the reference laptop the plain model was within 6 % where the disk dominates and 1.2 to 1.9 x optimistic where RAM does, and the --spec model 10 to 25 % pessimistic on streamed budgets (docs/ladder.md)", decode_bytes as f64 / 1e9));
    o.line(format!("  {:>13} {:>13} {:>8} {:>13} {:>12} {:>10} {:>10} {:>12} {:>10}", "budget", "pinned (spec)", "streamed", "GB disk/token", "GB ram/token", "plain s/tok", "tok/s", "spec3 s/tok", "tok/s"));
    let mut budgets: Vec<(String, Option<u64>)> = [6u64, 8, 12, 16].iter().map(|g| (format!("{g} GiB"), Some(g << 30))).collect();
    budgets.push((format!("{} (here)", size_flag(budget)), Some(budget)));
    budgets.push(("resident".into(), None));
    let env = CostEnv { max_pos, n_slots, sector, decode_bytes, membw, diskbw };
    let mut here: Option<Prediction> = None;
    for (name, b) in budgets {
        let p = predict(&input, b, &env);
        if !p.fits {
            o.line(format!("  {name:>13} refused: needs {} bytes = {:.2} GiB", group(p.minimum), gib(p.minimum)));
        } else {
            o.line(format!("  {name:>13} {:>13} {:>8} {:>13.3} {:>12.3} {:>10.2} {:>10.2} {:>12.2} {:>10.2}", format!("{} ({})", p.pinned, p.pinned_spec), p.streamed, p.disk_gb, p.ram_gb, p.t_plain, 1.0 / p.t_plain, p.t_spec, 1.0 / p.t_spec));
        }
        if b == Some(budget) {
            here = Some(p);
        }
    }
    let here = here.expect("the budget row");

    // ---- verdict
    o.line(String::new());
    let budget_flag = format!("--budget {}", size_flag(budget));
    // --spec pays where the disk is the bottleneck (docs/ladder.md); recommend it only where the model says it wins
    let use_spec = here.t_spec < here.t_plain;
    let speed = |p: &Prediction| {
        if p.t_spec < p.t_plain {
            format!("~{:.1} s/token with --spec {DEFAULT_SPEC_K} (~{:.1} plain)", p.t_spec, p.t_plain)
        } else {
            format!("~{:.1} s/token (--spec {DEFAULT_SPEC_K} would not help here: ~{:.1})", p.t_plain, p.t_spec)
        }
    };
    let verdict = if !avx2 {
        "Cannot run: AVX2 missing.".to_string()
    } else if let Some(e) = &plan_err {
        format!("Cannot run at this free RAM: {e}. Close other programs or run on a machine with more free RAM (an 8 GB laptop usually has ~5 GiB free, a 16 GB one ~11 GiB).")
    } else if here.streamed == 0 {
        format!("Good: everything fits in RAM; expect {} (the drive only affects the load: about {:.0} s for {:.1} GB at {diskbw:.2} GB/s).", speed(&here), model_bytes as f64 / 1e9 / diskbw, model_bytes as f64 / 1e9)
    } else {
        match info.class {
            DiskClass::Nvme => format!("Good: expect {}.", speed(&here)),
            DiskClass::Hdd => format!("Not recommended: spinning HDD, expect {}.", speed(&here)),
            DiskClass::Ssd => format!("Slow: a {} SSD will give {}; still works.", info.bus, speed(&here)),
            DiskClass::Unknown => {
                if diskbw >= 2.0 && !disk_assumed {
                    format!("Good: an unclassified drive reading at {diskbw:.2} GB/s; expect {}.", speed(&here))
                } else {
                    format!("Slow: an unclassified drive reading at {diskbw:.2} GB/s{} will give {}; still works.", if disk_assumed { " (assumed)" } else { "" }, speed(&here))
                }
            }
        }
    };
    o.line(format!("== verdict: {verdict}"));
    if plan_err.is_none() && here.streamed > 0 {
        o.line(format!("   ({} of {} layers pinned in RAM, {} streamed from disk each token, {:.3} GB per token; --spec {DEFAULT_SPEC_K} pins {} and drafts {DEFAULT_SPEC_K} tokens per round)", here.pinned, input.n_layer, here.streamed, here.disk_gb, here.pinned_spec));
    }

    // ---- next
    o.line(String::new());
    o.line("== next".into());
    let sha_tool = if cfg!(windows) { format!("certutil -hashfile {} SHA256", quote(&model_path)) } else { format!("sha256sum {}", quote(&model_path)) };
    let mut step = 1;
    if !present {
        o.line(format!("  {step}. download the model ({:.1} GB, {}/{}, Qwen's Apache-2.0 license) to the path above:", known::FILE_SIZE as f64 / 1e9, known::REPO, known::FILE_NAME));
        o.line(format!("       {}", shell_download(known::URL, &model_path)));
        step += 1;
        o.line(format!("  {step}. verify it (expect {}):", known::SHA256));
        o.line(format!("       {sha_tool}"));
        o.line(format!("       or: aqueduct doctor --model {} --sha256", quote(&model_path)));
        step += 1;
    } else if !model_ok {
        if sha_line.is_some() {
            o.line(format!("  {step}. the file at {model_path} is NOT the supported one; download {} from {} (see: aqueduct doctor --model <new path>)", known::FILE_NAME, known::REPO));
        } else if !is_known_name || file_size != known::FILE_SIZE {
            o.line(format!("  {step}. the file at {model_path} is not {} at {} bytes; only that file is supported (other recipes have tensor types this engine has no kernels for):", known::FILE_NAME, group(known::FILE_SIZE)));
            o.line(format!("       {}", shell_download(known::URL, &model_path)));
        }
        step += 1;
    } else if sha_line.is_none() {
        o.line(format!("  {step}. (optional) verify the file once: {sha_tool}   expect {}", known::SHA256));
        step += 1;
    }
    let assets = [("tokenizer.json", tokenizer.as_str()), ("chat_template.jinja", template.as_str()), ("generation_config.json", gen_config.as_str())];
    let missing: Vec<&(&str, &str)> = assets.iter().filter(|(_, p)| !Path::new(p).is_file()).collect();
    if missing.is_empty() {
        o.line(format!("  {step}. tokenizer, chat template and generation config: present ({}, {}, {})", tokenizer, template, gen_config));
    } else {
        o.line(format!("  {step}. the tokenizer, chat template and generation config from Qwen/Qwen3.8-27B (Apache-2.0; {} of 3 missing):", missing.len()));
        for (name, dest) in &missing {
            o.line(format!("       {}", shell_download(&format!("https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/{name}"), dest)));
        }
    }
    step += 1;
    if avx2 && plan_err.is_none() {
        o.line(format!("  {step}. run (from this directory, so the files above are found):"));
        o.line(format!("       aqueduct chat --model {} {budget_flag}{}", quote(&model_path), if use_spec { format!(" --spec {DEFAULT_SPEC_K}") } else { String::new() }));
        o.line(format!("       ({budget_flag} is the plan above{}; the smallest budget this model accepts is {:.2} GiB)", if budget_arg.is_none() { format!(": your free RAM right now, {:.2} GiB, rounded down", gib(avail)) } else { String::new() }, gib(here.minimum)));
    }

    if let Some(out_path) = out_path {
        if let Some(parent) = Path::new(&out_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&out_path, &o.0).map_err(|e| format!("write {out_path}: {e}"))?;
        eprintln!("filed to {out_path}");
    }
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
