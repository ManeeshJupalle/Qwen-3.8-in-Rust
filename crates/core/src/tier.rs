//! Tiers (Phase 4): the memory plan, the arenas layers live in, and the ring that streams the layers that do
//! not fit.
//!
//! **One loader for both tiers.** Every layer's projection matrices are `WeightMat` views into an arena
//! holding the layer's whole contiguous byte span from the GGUF (`docs/data/gguf_layout.txt`: every layer is
//! contiguous). A tier-1 (pinned) layer has its own arena, filled once at load; a tier-2 (streamed) layer is
//! a view into a ring slot that the I/O thread refills with that layer once per pass. The small per-layer
//! f32 vectors (norms, the conv kernel, the DeltaNet constants, `RESIDENT_SMALL`) are resident copies for
//! every layer, so a decoder layer struct never changes after load, whichever tier it is in.
//!
//! **The ring.** `n_slots` arenas sized for the largest streamed layer; streamed layer `L` always lands in
//! slot `(L - first_streamed) % n_slots`, so its views are fixed at load and nothing is re-pointed per token.
//! The I/O thread walks the streamed layers cyclically (`first..n_layer`, again and again), waits for the
//! target slot to be free, reads the layer's aligned superset with one unbuffered read, marks the slot
//! ready; the decode thread waits for `Ready(L)` before running layer `L` and marks the slot free after it.
//! Both orders are the layer order, so a slot's next occupant is always the layer `n_slots` further on and
//! the prefetch of `L + 1` overlaps the compute of `L`. (At the pass boundary the last streamed layer and
//! the first may share a slot when the streamed count is not a multiple of `n_slots`; the refill then starts
//! when the last layer finishes and overlaps the head and the pinned prefix instead.)
//!
//! **Rule 4.** Every fill is exactly one read of one layer span; the consumer counts the span bytes it takes
//! (`consumed_bytes`), the I/O thread counts what it read (`read_bytes`, plus `io_bytes` with the sector
//! slack), and `release` asserts that a layer's read count never exceeds its consume count by more than the
//! one prefetch in flight. `Model` asserts per token that the consumed delta equals the sum of the streamed
//! layer sizes exactly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::config::ModelConfig;
use crate::gguf::{layer_of, Gguf, TensorInfo};
use crate::kernels::matvec::WeightMat;
use crate::layers::{LayerError, TensorSource};
use crate::os::{align_down, align_up, arena_bytes, AlignedBuf, DirectFile};

/// Layer tensors kept as resident f32 copies for every layer (loaded through `TensorSource::vec`): the two
/// norms, the DeltaNet per-head constants, the conv kernel, the gated-norm weight, the q/k norms. Everything
/// else in a layer is a projection matrix and is a view into the layer's arena or ring slot. The plan sums
/// these per layer and `SlotSource` asserts the split, so the plan and the loader cannot disagree.
pub const RESIDENT_SMALL: &[&str] = &["attn_norm.weight", "post_attention_norm.weight", "ssm_a", "ssm_dt.bias", "ssm_conv1d.weight", "ssm_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight"];

pub fn is_resident_small(name: &str) -> bool {
    layer_of(name).is_some() && RESIDENT_SMALL.iter().any(|s| name.ends_with(s))
}

/// Bytes of `ActBuf::with_capacity(n)`: a Q8_0 row (`n` codes, per-32 f16 scale, f32 scale, `dxs`, 8 i32
/// `off6`) plus a Q8_K row (`n` codes, per-256 f32 scale, 16 i16 `bsums`, 8 i16 `q8s`). `n` is a multiple
/// of 256 everywhere it is used, so this is exact.
pub fn act_buf_bytes(n: u64) -> u64 {
    let q8 = n + n / 32 * (2 + 4 + 4 + 32);
    let q8k = n + n / 256 * (4 + 32 + 16);
    q8 + q8k
}

/// The numbers the plan needs, taken from the GGUF header and the config: no tensor data is read.
#[derive(Debug, Clone)]
pub struct PlanInput {
    /// Non-layer tensors in file order: (name, bytes).
    pub non_layer: Vec<(String, u64)>,
    /// The contiguous file range holding them (`start`, `end`), if they are contiguous enough to be one arena.
    pub non_layer_region: Option<(u64, u64)>,
    /// MTP blocks: (layer index, span bytes, span start).
    pub mtp: Vec<(u32, u64, u64)>,
    /// Span (start, end) of layers `0..n_layer`.
    pub layer_span: Vec<(u64, u64)>,
    /// Resident small-tensor bytes per layer (`RESIDENT_SMALL`).
    pub layer_small: Vec<u64>,
    pub n_layer: u32,
    pub n_deltanet: u32,
    pub n_attention: u32,
    pub hidden: u64,
    pub inter: u64,
    pub vocab: u64,
    /// DeltaNet: conv channels, V heads, k dim, v dim, conv kernel.
    pub dn_conv_dim: u64,
    pub dn_n_v: u64,
    pub dn_dk: u64,
    pub dn_dv: u64,
    pub dn_kernel: u64,
    /// Attention: heads, kv heads, head dim, rope dim.
    pub n_head: u64,
    pub n_head_kv: u64,
    pub head_dim: u64,
    pub rope_dim: u64,
}

impl PlanInput {
    pub fn new(g: &Gguf, cfg: &ModelConfig) -> PlanInput {
        let nl = g.non_layer_tensors();
        let non_layer: Vec<(String, u64)> = nl.iter().map(|t| (t.name.clone(), t.byte_size)).collect();
        let non_layer_region = if nl.is_empty() {
            None
        } else {
            let start = nl[0].absolute_file_offset;
            let end = nl.iter().map(|t| t.end_offset()).max().unwrap();
            let sum: u64 = nl.iter().map(|t| t.byte_size).sum();
            // one arena only when the region is not mostly other tensors (the file has them adjacent)
            if end - start <= sum + sum / 8 {
                Some((start, end))
            } else {
                None
            }
        };
        let mtp = cfg.mtp_layers.iter().map(|&m| {
            let s = g.layer_span(m).expect("mtp span");
            (m, s.end - s.start, s.start)
        }).collect();
        let mut layer_span = Vec::with_capacity(cfg.n_layer as usize);
        let mut layer_small = Vec::with_capacity(cfg.n_layer as usize);
        for i in 0..cfg.n_layer {
            let s = g.layer_span(i).expect("layer span");
            layer_span.push((s.start, s.end));
            layer_small.push(g.layer_tensors(i).iter().filter(|t| is_resident_small(&t.name)).map(|t| t.byte_size).sum());
        }
        PlanInput {
            non_layer,
            non_layer_region,
            mtp,
            layer_span,
            layer_small,
            n_layer: cfg.n_layer,
            n_deltanet: cfg.deltanet_layers.len() as u32,
            n_attention: cfg.full_attention_layers.len() as u32,
            hidden: cfg.hidden_size as u64,
            inter: cfg.intermediate_size as u64,
            vocab: cfg.vocab_size as u64,
            dn_conv_dim: cfg.dn_qkv_dim as u64,
            dn_n_v: cfg.dn_n_v_heads as u64,
            dn_dk: cfg.dn_head_dim_k as u64,
            dn_dv: cfg.dn_head_dim_v as u64,
            dn_kernel: cfg.dn_conv_kernel as u64,
            n_head: cfg.n_head as u64,
            n_head_kv: cfg.n_head_kv as u64,
            head_dim: cfg.head_dim_k as u64,
            rope_dim: cfg.rope_dim as u64,
        }
    }

    pub fn layer_bytes(&self, i: usize) -> u64 {
        self.layer_span[i].1 - self.layer_span[i].0
    }

    /// Bytes of the non-layer arena (or of the per-tensor copies when there is no single region).
    pub fn non_layer_arena_bytes(&self, sector: u64) -> u64 {
        match self.non_layer_region {
            Some((s, e)) => arena_bytes(e - s, sector),
            None => self.non_layer.iter().map(|(_, b)| *b).sum(),
        }
    }

    /// KV cache bytes per position over the attention layers (k and v, f32).
    pub fn kv_bytes_per_pos(&self) -> u64 {
        self.n_attention as u64 * 2 * self.n_head_kv * self.head_dim * 4
    }

    /// DeltaNet carried state: recurrent matrices plus the conv window, f32.
    pub fn deltanet_state_bytes(&self) -> u64 {
        self.n_deltanet as u64 * (self.dn_n_v * self.dn_dk * self.dn_dv + self.dn_conv_dim * (self.dn_kernel - 1)) * 4
    }

    /// Every per-token buffer `model::State` holds (mirrors `Scratch::for_layers`, `DeltaScratch::new`,
    /// `AttnScratch::new` + `reserve(max_pos)`, `MlpScratch::new`, the residual ping-pong, the head input),
    /// plus the caller's logits vector. `State::bytes` measures the live structs; `tests/tier_plan.rs`
    /// checks the two agree on the tiny model.
    pub fn scratch_bytes(&self, max_pos: u64) -> u64 {
        let h = self.hidden;
        let mut b = 0u64;
        // State: h0, h1, lm_normed, lm_act
        b += 3 * h * 4 + act_buf_bytes(h);
        // Scratch: normed, mixed, resid, act; mlp
        b += 3 * h * 4 + act_buf_bytes(h);
        b += 3 * self.inter * 4 + act_buf_bytes(self.inter);
        if self.n_deltanet > 0 {
            let (cd, nv, dk, dv) = (self.dn_conv_dim, self.dn_n_v, self.dn_dk, self.dn_dv);
            let vd = nv * dv;
            b += (cd + vd + nv + nv + cd + vd + nv * dk + nv * dk + vd) * 4 + act_buf_bytes(vd);
        }
        if self.n_attention > 0 {
            let (nh, nkv, hd, rd) = (self.n_head, self.n_head_kv, self.head_dim, self.rope_dim);
            b += (2 * nh * hd + nkv * hd + nkv * hd + rd + rd + nh * hd + nh * hd + hd + nh * hd) * 4 + act_buf_bytes(nh * hd);
            b += 2 * nh * max_pos * 4; // scores, probs
        }
        // logits
        b += self.vocab * 4;
        b
    }
}

/// What the plan is computed for.
#[derive(Debug, Clone)]
pub struct PlanParams {
    /// `None`: everything resident (the ladder's "resident" rung), no ring.
    pub budget: Option<u64>,
    /// Positions the KV cache and the score buffers are sized for.
    pub max_pos: u64,
    pub n_slots: usize,
    /// Alignment of the arenas and reads (the drive's sector size, queried).
    pub sector: u64,
    /// Reserve for everything that is not a planned buffer: the binary, the parsed GGUF header (the
    /// vocabulary arrays), thread stacks, allocator slack. `aqueduct plan` prints the measured baseline
    /// next to it.
    pub baseline: u64,
}

impl PlanParams {
    pub const DEFAULT_BASELINE: u64 = 256 << 20;
    pub fn new(budget: Option<u64>, max_pos: u64, n_slots: usize, sector: u64) -> PlanParams {
        PlanParams { budget, max_pos, n_slots: n_slots.max(1), sector, baseline: Self::DEFAULT_BASELINE }
    }
}

/// One row of the plan table.
#[derive(Debug, Clone)]
pub struct PlanLine {
    pub name: String,
    pub bytes: u64,
}

/// The memory plan: what is always resident, how many layers are pinned, the ring, the total against the
/// budget. Computed before a single byte is allocated; `check()` refuses a plan that does not fit.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub params: PlanParams,
    pub lines: Vec<PlanLine>,
    /// Sum of the always-resident lines (everything except the ring and tier 1).
    pub resident_bytes: u64,
    pub n_layer: u32,
    /// Layers `0..pinned` are tier 1.
    pub pinned: u32,
    pub pinned_bytes: u64,
    /// 0 when nothing is streamed.
    pub n_slots: usize,
    pub slot_bytes: u64,
    pub ring_bytes: u64,
    /// The largest streamed layer (index, span bytes), if any.
    pub largest_streamed: Option<(u32, u64)>,
    /// Sum of the streamed layers' spans: what one pass (one token) reads from disk.
    pub streamed_bytes_per_pass: u64,
    pub total: u64,
    /// Bytes the always-resident set plus a full ring need: the minimum feasible budget.
    pub minimum: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("memory plan does not fit: the always-resident set ({resident} bytes) plus a {n_slots}-slot ring ({ring} bytes) needs {minimum} bytes = {minimum_gib:.2} GiB, the budget is {budget} bytes = {budget_gib:.2} GiB: short by {shortfall} bytes = {shortfall_gib:.2} GiB")]
    DoesNotFit { resident: u64, n_slots: usize, ring: u64, minimum: u64, minimum_gib: f64, budget: u64, budget_gib: f64, shortfall: u64, shortfall_gib: f64 },
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

impl MemoryPlan {
    /// Pure arithmetic over `input` and `params`. Tier 1 is the longest prefix of whole layers `0..k` such
    /// that the always-resident set, a ring sized for the largest layer of the streamed suffix `k..`, and
    /// the prefix's arenas fit the budget (every `k` is tried from the top down: pinning one more layer can
    /// shrink the ring, so the feasible set is not a simple prefix of `k`). No ring when everything fits.
    pub fn compute(input: &PlanInput, params: &PlanParams) -> MemoryPlan {
        let sector = params.sector.max(1);
        let n_layer = input.n_layer;
        let mut lines = Vec::new();
        let non_layer_names: Vec<String> = input.non_layer.iter().map(|(n, b)| format!("{n} {b}")).collect();
        lines.push(PlanLine { name: format!("non-layer arena [{}]", non_layer_names.join(", ")), bytes: input.non_layer_arena_bytes(sector) });
        for (m, span, _) in &input.mtp {
            lines.push(PlanLine { name: format!("MTP block blk.{m} arena (loaded, unused)"), bytes: arena_bytes(*span, sector) });
        }
        lines.push(PlanLine { name: format!("per-layer small tensors, {n_layer} layers (norms, conv, dt/a; resident copies)"), bytes: input.layer_small.iter().sum() });
        lines.push(PlanLine { name: format!("DeltaNet carried state ({} layers)", input.n_deltanet), bytes: input.deltanet_state_bytes() });
        lines.push(PlanLine { name: format!("KV cache ({} layers x {} positions)", input.n_attention, params.max_pos), bytes: input.kv_bytes_per_pos() * params.max_pos });
        lines.push(PlanLine { name: "scratch buffers, activation rows, logits".into(), bytes: input.scratch_bytes(params.max_pos) });
        lines.push(PlanLine { name: "process baseline reserve (binary, GGUF header index, thread stacks, allocator)".into(), bytes: params.baseline });
        let resident_bytes: u64 = lines.iter().map(|l| l.bytes).sum();

        let layer_total: u64 = (0..n_layer as usize).map(|i| input.layer_bytes(i)).sum();
        let arena_total: u64 = (0..n_layer as usize).map(|i| arena_bytes(input.layer_bytes(i), sector)).sum();
        let largest_from = |from: u32| -> Option<(u32, u64)> {
            let mut best: Option<(u32, u64)> = None;
            for i in from..n_layer {
                let b = input.layer_bytes(i as usize);
                if best.is_none_or(|(_, bb)| b > bb) {
                    best = Some((i, b));
                }
            }
            best
        };
        // prefix arena sums: prefix[k] = bytes of the arenas of layers 0..k
        let mut prefix = Vec::with_capacity(n_layer as usize + 1);
        prefix.push(0u64);
        for i in 0..n_layer as usize {
            let p = prefix[i] + arena_bytes(input.layer_bytes(i), sector);
            prefix.push(p);
        }
        // the ring a suffix k.. needs (none when the suffix is empty)
        let ring_for = |k: u32| -> (u64, u64, Option<(u32, u64)>) {
            if k >= n_layer {
                return (0, 0, None);
            }
            let largest = largest_from(k);
            let slot = arena_bytes(largest.map(|(_, b)| b).unwrap_or(0), sector);
            (slot, slot * params.n_slots as u64, largest)
        };
        let (pinned, n_slots, slot_bytes, ring_bytes, largest_streamed) = match params.budget {
            None => (n_layer, 0usize, 0u64, 0u64, None),
            Some(budget) if budget >= resident_bytes + arena_total => (n_layer, 0, 0, 0, None),
            Some(budget) => {
                let mut chosen = None;
                for k in (0..n_layer).rev() {
                    let (slot, ring, largest) = ring_for(k);
                    if resident_bytes + ring + prefix[k as usize] <= budget {
                        chosen = Some((k, params.n_slots, slot, ring, largest));
                        break;
                    }
                }
                // nothing fits: report the k = 0 plan, which `check` refuses
                chosen.unwrap_or_else(|| {
                    let (slot, ring, largest) = ring_for(0);
                    (0, params.n_slots, slot, ring, largest)
                })
            }
        };
        let pinned_bytes = prefix[pinned as usize];
        let streamed_bytes_per_pass: u64 = (pinned as usize..n_layer as usize).map(|i| input.layer_bytes(i)).sum();
        let total = resident_bytes + ring_bytes + pinned_bytes;
        // the smallest budget at which any plan is feasible: the resident set plus a ring for the largest layer
        let minimum = resident_bytes + ring_for(0).1;
        let _ = layer_total;
        MemoryPlan { params: params.clone(), lines, resident_bytes, n_layer, pinned, pinned_bytes, n_slots, slot_bytes, ring_bytes, largest_streamed, streamed_bytes_per_pass, total, minimum }
    }

    pub fn streamed(&self) -> u32 {
        self.n_layer - self.pinned
    }

    pub fn fits(&self) -> bool {
        match self.params.budget {
            None => true,
            Some(b) => self.total <= b && (self.streamed() == 0 || self.n_slots > 0),
        }
    }

    /// Bytes below the budget (negative when it does not fit).
    pub fn headroom(&self) -> i64 {
        match self.params.budget {
            None => 0,
            Some(b) => b as i64 - self.total as i64,
        }
    }

    /// `Err` with the shortfall when the plan cannot fit its budget.
    pub fn check(&self) -> Result<(), PlanError> {
        match self.params.budget {
            Some(budget) if !self.fits() || self.minimum > budget => {
                let ring = self.minimum - self.resident_bytes;
                Err(PlanError::DoesNotFit {
                    resident: self.resident_bytes,
                    n_slots: self.params.n_slots,
                    ring,
                    minimum: self.minimum,
                    minimum_gib: gib(self.minimum),
                    budget,
                    budget_gib: gib(budget),
                    shortfall: self.minimum.saturating_sub(budget).max(1),
                    shortfall_gib: gib(self.minimum.saturating_sub(budget)),
                })
            }
            _ => Ok(()),
        }
    }

    /// The table: category, bytes, running total; then budget and headroom, and the tier-2 line.
    pub fn table(&self) -> String {
        let mut s = String::new();
        let budget = match self.params.budget {
            Some(b) => format!("{} bytes = {:.2} GiB", group(b), gib(b)),
            None => "none (everything resident)".into(),
        };
        s.push_str(&format!("memory plan: budget {budget}; max_pos {}; sector {}; ring {} slots\n", self.params.max_pos, self.params.sector, self.params.n_slots));
        s.push_str(&format!("  {:<88} {:>18} {:>18}\n", "category", "bytes", "running total"));
        let mut run = 0u64;
        let mut line = |name: &str, bytes: u64, run: &mut u64| {
            *run += bytes;
            s.push_str(&format!("  {:<88} {:>18} {:>18}\n", trunc(name, 88), group(bytes), group(*run)));
        };
        for l in &self.lines {
            line(&l.name, l.bytes, &mut run);
        }
        if self.n_slots > 0 {
            let (li, lb) = self.largest_streamed.unwrap_or((0, 0));
            line(&format!("ring: {} slots x {} (largest streamed layer {li}: {} bytes)", self.n_slots, group(self.slot_bytes), group(lb)), self.ring_bytes, &mut run);
        }
        line(&format!("tier 1 (pinned): layers 0..{} ({} layers), aligned arenas", self.pinned, self.pinned), self.pinned_bytes, &mut run);
        s.push_str(&format!("  {:<88} {:>18}\n", "total", group(self.total)));
        if let Some(b) = self.params.budget {
            let head = self.headroom();
            s.push_str(&format!("  {:<88} {:>18}\n", "budget", group(b)));
            s.push_str(&format!("  {:<88} {:>18}   {}\n", "headroom", if head >= 0 { group(head as u64) } else { format!("-{}", group((-head) as u64)) }, if self.fits() { "fits" } else { "DOES NOT FIT" }));
        }
        s.push_str(&format!(
            "  tier 2 (streamed): layers {}..{} ({} layers), {} bytes = {:.3} GB per token from disk\n",
            self.pinned,
            self.n_layer,
            self.streamed(),
            group(self.streamed_bytes_per_pass),
            self.streamed_bytes_per_pass as f64 / 1e9
        ));
        s
    }
}

/// Thousands separators.
pub fn group(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let t: String = s.chars().take(n - 3).collect();
        format!("{t}...")
    }
}

/// Parse `8G`, `512M`, `8.5G`, `12345` (bytes). `G` and `M` are binary (GiB, MiB), which is how RAM is sold
/// and how Windows reports it.
pub fn parse_bytes(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (num, mul) = match s.chars().last() {
        Some('G') | Some('g') => (&s[..s.len() - 1], 1u64 << 30),
        Some('M') | Some('m') => (&s[..s.len() - 1], 1u64 << 20),
        Some('K') | Some('k') => (&s[..s.len() - 1], 1u64 << 10),
        _ => (s, 1u64),
    };
    let v: f64 = num.parse().map_err(|_| format!("bad size {s:?} (use e.g. 8G, 512M, or bytes)"))?;
    if v < 0.0 {
        return Err(format!("negative size {s:?}"));
    }
    Ok((v * mul as f64).round() as u64)
}

// ================================================================================================ sources

/// A `TensorSource` over one arena holding a contiguous file range: matrices become views, small vectors
/// become resident copies read from the GGUF the ordinary way.
pub struct ArenaSource<'a> {
    pub g: &'a Gguf,
    /// File offset of the arena's first byte (sector-aligned).
    pub file_start: u64,
    pub buf: Arc<AlignedBuf>,
}

impl ArenaSource<'_> {
    fn window(&self, t: &TensorInfo) -> Result<(usize, usize), LayerError> {
        let off = t.absolute_file_offset.checked_sub(self.file_start).ok_or_else(|| LayerError::Missing(format!("{} lies before its arena", t.name)))?;
        if off + t.byte_size > self.buf.len() as u64 {
            return Err(LayerError::Missing(format!("{} lies past its arena", t.name)));
        }
        Ok((off as usize, t.byte_size as usize))
    }
}

impl TensorSource for ArenaSource<'_> {
    fn mat(&self, name: &str, rows: usize, cols: usize) -> Result<WeightMat, LayerError> {
        assert!(!is_resident_small(name), "ArenaSource::mat: {name} is a resident small tensor, not a matrix");
        let t = self.g.tensor(name).map_err(|_| LayerError::Missing(name.to_string()))?;
        let found: Vec<usize> = t.shape_numpy_order().iter().map(|&d| d as usize).collect();
        if found != vec![rows, cols] {
            return Err(LayerError::Shape { name: name.into(), expected: vec![rows, cols], found });
        }
        let (off, len) = self.window(t)?;
        Ok(WeightMat::view(t.ggml_type, rows, cols, Arc::clone(&self.buf), off, len))
    }

    fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, LayerError> {
        assert!(layer_of(name).is_none() || is_resident_small(name), "ArenaSource::vec: {name} is not in RESIDENT_SMALL");
        self.g.vec(name, len)
    }
}

// ================================================================================================ the ring

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    Ready(u32),
}

struct Slot {
    buf: Arc<AlignedBuf>,
    state: Mutex<SlotState>,
    cv: Condvar,
}

/// Counters of the streaming tier (all monotone; the I/O thread and the consumer each own their side).
#[derive(Default)]
pub struct StreamCounters {
    /// Span bytes the consumer took (one span per streamed layer per pass).
    pub consumed_bytes: AtomicU64,
    pub consumed_layers: AtomicU64,
    /// Span bytes the I/O thread read (useful bytes) and the bytes it asked the OS for (sector slack included).
    pub read_bytes: AtomicU64,
    pub io_bytes: AtomicU64,
    pub reads: AtomicU64,
    /// Time the consumer spent waiting for a slot, and time the I/O thread spent inside reads.
    pub wait_ns: AtomicU64,
    pub read_ns: AtomicU64,
    /// Per layer: reads issued and consumes taken (rule 4: reads - consumes is 0 or 1).
    pub reads_per_layer: Vec<AtomicU32>,
    pub consumes_per_layer: Vec<AtomicU32>,
}

/// A snapshot of the counters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StreamStats {
    pub consumed_bytes: u64,
    pub consumed_layers: u64,
    pub read_bytes: u64,
    pub io_bytes: u64,
    pub reads: u64,
    pub wait_ns: u64,
    pub read_ns: u64,
}

impl StreamCounters {
    pub fn snapshot(&self) -> StreamStats {
        StreamStats {
            consumed_bytes: self.consumed_bytes.load(Ordering::Acquire),
            consumed_layers: self.consumed_layers.load(Ordering::Acquire),
            read_bytes: self.read_bytes.load(Ordering::Acquire),
            io_bytes: self.io_bytes.load(Ordering::Acquire),
            reads: self.reads.load(Ordering::Acquire),
            wait_ns: self.wait_ns.load(Ordering::Acquire),
            read_ns: self.read_ns.load(Ordering::Acquire),
        }
    }
}

/// Where a streamed layer's bytes sit: the aligned file range read for it and the slack before the span.
#[derive(Debug, Clone, Copy)]
pub struct LayerRead {
    pub layer: u32,
    pub span_start: u64,
    pub span_bytes: u64,
    /// Sector-aligned read range.
    pub read_start: u64,
    pub read_bytes: u64,
    /// `span_start - read_start`: where the span begins inside the slot.
    pub head: usize,
}

impl LayerRead {
    pub fn new(layer: u32, span: (u64, u64), sector: u64) -> LayerRead {
        let read_start = align_down(span.0, sector);
        let read_end = align_up(span.1, sector);
        LayerRead { layer, span_start: span.0, span_bytes: span.1 - span.0, read_start, read_bytes: read_end - read_start, head: (span.0 - read_start) as usize }
    }
}

/// The streaming tier: ring slots, the I/O thread, the counters.
pub struct Ring {
    slots: Vec<Arc<Slot>>,
    pub first: u32,
    pub n_layer: u32,
    pub counters: Arc<StreamCounters>,
    pub reads: Vec<LayerRead>,
    stop: Arc<AtomicBool>,
    io: Option<std::thread::JoinHandle<()>>,
    pub large_pages: bool,
}

impl Ring {
    /// Slot of streamed layer `layer` (fixed for the run).
    pub fn slot_of(&self, layer: u32) -> usize {
        ((layer - self.first) as usize) % self.slots.len()
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn slot_buf(&self, s: usize) -> Arc<AlignedBuf> {
        Arc::clone(&self.slots[s].buf)
    }

    /// Allocate `n_slots` slots of `slot_bytes` and start the I/O thread, which begins prefetching layer
    /// `first` at once. `file` is the unbuffered handle the thread reads with; `spans` are the layer spans of
    /// `0..n_layer`.
    pub fn start(file: DirectFile, spans: &[(u64, u64)], first: u32, n_slots: usize, slot_bytes: u64, large_pages: bool, counters: Arc<StreamCounters>) -> std::io::Result<Ring> {
        let n_layer = spans.len() as u32;
        assert!(first < n_layer && n_slots >= 1);
        let sector = file.sector as u64;
        let reads: Vec<LayerRead> = (0..n_layer).map(|l| LayerRead::new(l, spans[l as usize], sector)).collect();
        let mut slots = Vec::with_capacity(n_slots);
        let mut any_large = false;
        for _ in 0..n_slots {
            let buf = AlignedBuf::new(slot_bytes as usize, large_pages)?;
            for r in &reads[first as usize..] {
                assert!(r.read_bytes <= buf.len() as u64, "ring slot {} bytes cannot hold layer {} read of {} bytes", buf.len(), r.layer, r.read_bytes);
            }
            any_large |= buf.large_pages;
            slots.push(Arc::new(Slot { buf: Arc::new(buf), state: Mutex::new(SlotState::Free), cv: Condvar::new() }));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let io = {
            let slots = slots.clone();
            let reads = reads.clone();
            let stop = Arc::clone(&stop);
            let counters = Arc::clone(&counters);
            let mut file = file;
            std::thread::Builder::new().name("aqueduct-io".into()).spawn(move || {
                let n_streamed = (n_layer - first) as u64;
                let mut idx = 0u64;
                loop {
                    let layer = first + (idx % n_streamed) as u32;
                    let s = ((layer - first) as usize) % slots.len();
                    let slot = &slots[s];
                    // wait for the slot to be free (or for shutdown)
                    {
                        let mut st = slot.state.lock().unwrap();
                        while *st != SlotState::Free {
                            if stop.load(Ordering::Acquire) {
                                return;
                            }
                            st = slot.cv.wait(st).unwrap();
                        }
                    }
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    let r = reads[layer as usize];
                    let t0 = Instant::now();
                    let got = file.read_at(r.read_start, &slot.buf, r.read_bytes as usize).unwrap_or_else(|e| panic!("aqueduct-io: read of layer {} at {} ({} bytes) failed: {e}", r.layer, r.read_start, r.read_bytes));
                    assert!(got as u64 >= r.head as u64 + r.span_bytes, "aqueduct-io: short read for layer {}: {got} of {} bytes", r.layer, r.read_bytes);
                    counters.read_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    counters.read_bytes.fetch_add(r.span_bytes, Ordering::Relaxed);
                    counters.io_bytes.fetch_add(got as u64, Ordering::Relaxed);
                    counters.reads.fetch_add(1, Ordering::Relaxed);
                    counters.reads_per_layer[layer as usize].fetch_add(1, Ordering::Relaxed);
                    {
                        let mut st = slot.state.lock().unwrap();
                        *st = SlotState::Ready(layer);
                        slot.cv.notify_all();
                    }
                    idx += 1;
                }
            })?
        };
        Ok(Ring { slots, first, n_layer, counters, reads, stop, io: Some(io), large_pages: any_large })
    }

    /// Block until streamed layer `layer` is in its slot. Counts the span as consumed.
    pub fn acquire(&self, layer: u32) {
        let slot = &self.slots[self.slot_of(layer)];
        let t0 = Instant::now();
        let mut st = slot.state.lock().unwrap();
        while *st != SlotState::Ready(layer) {
            st = slot.cv.wait(st).unwrap();
        }
        drop(st);
        let c = &self.counters;
        c.wait_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        c.consumed_bytes.fetch_add(self.reads[layer as usize].span_bytes, Ordering::Relaxed);
        c.consumed_layers.fetch_add(1, Ordering::Relaxed);
        c.consumes_per_layer[layer as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// Give the slot back so the I/O thread can refill it. Asserts rule 4 for this layer: its reads never
    /// run more than one ahead of its consumes (that one is the prefetch for the next pass).
    pub fn release(&self, layer: u32) {
        let c = &self.counters;
        let reads = c.reads_per_layer[layer as usize].load(Ordering::Acquire);
        let consumes = c.consumes_per_layer[layer as usize].load(Ordering::Acquire);
        assert!(reads == consumes || reads == consumes + 1, "layer {layer}: {reads} reads for {consumes} consumes (a byte was read more than once per pass)");
        let slot = &self.slots[self.slot_of(layer)];
        let mut st = slot.state.lock().unwrap();
        assert_eq!(*st, SlotState::Ready(layer), "release of layer {layer} but its slot holds {:?}", *st);
        *st = SlotState::Free;
        slot.cv.notify_all();
    }

    /// Wait for the I/O thread's current read to land and stop it. After this the counters are final.
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        for s in &self.slots {
            // wake the thread whether it is waiting for a free slot or about to
            let _g = s.state.lock().unwrap();
            s.cv.notify_all();
        }
        if let Some(h) = self.io.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Per-layer read ranges, keyed by layer, for `info`-style printing.
pub fn layer_reads(spans: &[(u64, u64)], sector: u64) -> BTreeMap<u32, LayerRead> {
    spans.iter().enumerate().map(|(i, &s)| (i as u32, LayerRead::new(i as u32, s, sector))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> PlanInput {
        // a toy model: 6 layers of 1000 / 2000 / 1500 / 1000 / 1000 / 1000 bytes, everything else zero-sized
        PlanInput {
            non_layer: vec![("token_embd.weight".into(), 100), ("output.weight".into(), 100)],
            non_layer_region: Some((0, 200)),
            mtp: vec![],
            layer_span: vec![(1000, 2000), (2000, 4000), (4000, 5500), (5500, 6500), (6500, 7500), (7500, 8500)],
            layer_small: vec![10, 10, 10, 10, 10, 10],
            n_layer: 6,
            n_deltanet: 0,
            n_attention: 0,
            hidden: 0,
            inter: 0,
            vocab: 0,
            dn_conv_dim: 0,
            dn_n_v: 0,
            dn_dk: 0,
            dn_dv: 0,
            dn_kernel: 1,
            n_head: 0,
            n_head_kv: 0,
            head_dim: 0,
            rope_dim: 0,
        }
    }

    #[test]
    fn pins_from_zero_and_sizes_the_ring_for_the_streamed_suffix() {
        let inp = input();
        let mut p = PlanParams::new(Some(0), 0, 2, 1);
        p.baseline = 0;
        // resident = 200 + 1 (arena sector) + 60 small = 261; arenas are span + 1
        let resident = 200 + 1 + 60;
        // budget just for resident + ring(2 x (2000+1)): nothing pinned
        p.budget = Some(resident + 2 * 2001);
        let plan = MemoryPlan::compute(&inp, &p);
        assert_eq!((plan.pinned, plan.n_slots, plan.slot_bytes), (0, 2, 2001));
        assert!(plan.fits() && plan.check().is_ok());
        assert_eq!(plan.streamed_bytes_per_pass, 7500);
        // one byte less: refused, naming the shortfall
        p.budget = Some(resident + 2 * 2001 - 1);
        let plan = MemoryPlan::compute(&inp, &p);
        let e = plan.check().unwrap_err().to_string();
        assert!(e.contains("short by 1 bytes"), "{e}");
        // layers 0 and 1 pinned: the ring shrinks to the largest of layers 2.. (1500); the same budget could
        // not hold layer 1 next to a ring sized for it, which is why the search runs from the top down
        p.budget = Some(resident + 2 * 1501 + 1001 + 2001);
        let plan = MemoryPlan::compute(&inp, &p);
        assert_eq!((plan.pinned, plan.slot_bytes), (2, 1501));
        assert_eq!(plan.streamed_bytes_per_pass, 4500);
        assert_eq!(plan.total, plan.params.budget.unwrap());
        // one byte less than that: only layer 0 fits next to a ring for layer 1
        p.budget = Some(resident + 2 * 1501 + 1001 + 2001 - 1);
        let plan = MemoryPlan::compute(&inp, &p);
        assert_eq!((plan.pinned, plan.slot_bytes), (1, 2001));
        // everything: no ring
        p.budget = Some(resident + 1001 + 2001 + 1501 + 3 * 1001);
        let plan = MemoryPlan::compute(&inp, &p);
        assert_eq!((plan.pinned, plan.n_slots, plan.ring_bytes), (6, 0, 0));
        p.budget = None;
        let plan = MemoryPlan::compute(&inp, &p);
        assert_eq!((plan.pinned, plan.n_slots), (6, 0));
        assert!(plan.table().contains("tier 2 (streamed): layers 6..6 (0 layers)"));
    }

    #[test]
    fn parse_sizes() {
        assert_eq!(parse_bytes("8G").unwrap(), 8 << 30);
        assert_eq!(parse_bytes("1.5G").unwrap(), 3 << 29);
        assert_eq!(parse_bytes("512M").unwrap(), 512 << 20);
        assert_eq!(parse_bytes("12345").unwrap(), 12345);
        assert!(parse_bytes("abc").is_err());
    }

    #[test]
    fn act_buf_formula_matches_the_rows() {
        use crate::kernels::matvec::ActBuf;
        for n in [5120u64, 6144, 8192, 17408] {
            let b = ActBuf::with_capacity(n as usize);
            assert_eq!(b.bytes() as u64, act_buf_bytes(n), "n {n}");
        }
    }

    #[test]
    fn layer_read_alignment() {
        let r = LayerRead::new(3, (1769121376, 2038802912), 4096);
        assert_eq!(r.read_start % 4096, 0);
        assert_eq!(r.read_bytes % 4096, 0);
        assert_eq!(r.head as u64, 1769121376 - r.read_start);
        assert!(r.read_bytes >= r.head as u64 + r.span_bytes);
        assert!(r.read_bytes <= arena_bytes(r.span_bytes, 4096));
    }
}
