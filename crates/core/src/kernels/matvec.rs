//! Matrix-vector products: rows of weights (ggml block layout or f32) times one activation vector, or
//! (`matmul`) times a batch of activation vectors with every weight byte read once. Rows are partitioned
//! into contiguous chunks over the persistent pool (`pool.rs`); every row is computed by exactly one
//! participant with the same kernel, so results are bit-identical for any thread count.
//!
//! Activations (Phase 3.5 decision, `docs/kquant-dot.md`): K-quant weights consume ggml's Q8_K rows
//! (`q8k.rs`, `kdot.rs` / `avx2.rs`), the production path since its 64-layer measurement met the promotion
//! criterion (argmax 3/3, top-10 >= 9/10, logits 0.28 to 0.36 from the f32 reference against the 0.52 to
//! 0.68 bf16 noise floor) and is gated by its own frozen ceilings; Q8_0-grain rows (`q8.rs`; `dot.rs` /
//! `avx2.rs` with the same lane structure, 2 to 4 x less rounding) are the `--q8-fine` opt-in for K-quants
//! (`set_q8_fine`) and the only form for Q8_0 / Q4_0 weights; F32 weights consume f32. `ActVec` holds one
//! vector with each quantised form computed at most once, so the projections that share an input share the
//! quantisation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use super::dot::{dot_f32, dot_q8};
use super::kdot::dot_q8k;
use super::pool;
use super::q8::Q8Row;
use super::q8k::Q8KRow;
use crate::gguf::GgmlType;
use crate::os::AlignedBuf;

/// Where a `WeightMat`'s bytes live: its own allocation, or a window of an arena / ring slot (Phase 4).
/// A view keeps the arena alive through the `Arc`; the ring protocol (`tier.rs`) guarantees nobody writes a
/// slot while a view of it is being read.
#[derive(Debug, Clone)]
pub enum WeightBytes {
    Owned(Vec<u8>),
    View { buf: Arc<AlignedBuf>, off: usize, len: usize },
}

/// A weight matrix of `rows` rows, each `cols` elements, stored as ggml rows (`row_bytes` each).
#[derive(Debug, Clone)]
pub struct WeightMat {
    pub ggml_type: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
    /// Raw ggml bytes (`rows * row_bytes`), also for F32 (little-endian f32).
    pub data: WeightBytes,
}

impl WeightMat {
    pub fn new(ggml_type: GgmlType, rows: usize, cols: usize, data: Vec<u8>) -> WeightMat {
        let row_bytes = Self::row_bytes_of(ggml_type, cols);
        assert_eq!(data.len(), rows * row_bytes, "WeightMat: data length");
        WeightMat { ggml_type, rows, cols, row_bytes, data: WeightBytes::Owned(data) }
    }

    /// A matrix whose bytes are `len` bytes at `off` inside `buf` (an arena or a ring slot).
    pub fn view(ggml_type: GgmlType, rows: usize, cols: usize, buf: Arc<AlignedBuf>, off: usize, len: usize) -> WeightMat {
        let row_bytes = Self::row_bytes_of(ggml_type, cols);
        assert_eq!(len, rows * row_bytes, "WeightMat::view: data length");
        assert!(off + len <= buf.len(), "WeightMat::view: window {off}..{} outside the {}-byte buffer", off + len, buf.len());
        WeightMat { ggml_type, rows, cols, row_bytes, data: WeightBytes::View { buf, off, len } }
    }

    fn row_bytes_of(ggml_type: GgmlType, cols: usize) -> usize {
        let (bs, ts) = ggml_type.block_layout();
        assert!((cols as u64).is_multiple_of(bs), "WeightMat: cols {cols} not a multiple of block size {bs} for {ggml_type:?}");
        (cols as u64 / bs * ts) as usize
    }

    /// All the bytes, whichever way they are held.
    #[inline]
    pub fn data(&self) -> &[u8] {
        match &self.data {
            WeightBytes::Owned(v) => v,
            // SAFETY: the window was checked at construction; the ring protocol keeps the slot stable while
            // any reader (a matvec on this matrix) is inside it.
            WeightBytes::View { buf, off, len } => unsafe { buf.slice(*off, *len) },
        }
    }

    /// Is this matrix a window of an arena rather than its own allocation?
    pub fn is_view(&self) -> bool {
        matches!(self.data, WeightBytes::View { .. })
    }

    pub fn from_f32(rows: usize, cols: usize, w: &[f32]) -> WeightMat {
        assert_eq!(w.len(), rows * cols);
        let mut data = Vec::with_capacity(w.len() * 4);
        for v in w {
            data.extend_from_slice(&v.to_le_bytes());
        }
        WeightMat::new(GgmlType::F32, rows, cols, data)
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[u8] {
        &self.data()[r * self.row_bytes..(r + 1) * self.row_bytes]
    }

    /// Row `r` as f32 (only for F32 matrices).
    pub fn row_f32(&self, r: usize) -> Vec<f32> {
        assert_eq!(self.ggml_type, GgmlType::F32);
        self.row(r).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// Weight bytes streamed by one matvec (what decode is bounded by).
    pub fn bytes(&self) -> usize {
        self.rows * self.row_bytes
    }
}

/// The activation form a weight type consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActKind {
    F32,
    Q8,
    Q8K,
}

static Q8_FINE: AtomicBool = AtomicBool::new(false);

/// Give K-quant weights Q8_0-grain activations (per-32 scales, `aqueduct run --q8-fine`) instead of the
/// production Q8_K rows. Off by default (the Phase 3.5 decision, see the module doc).
pub fn set_q8_fine(on: bool) {
    Q8_FINE.store(on, Ordering::Relaxed);
}

/// Are K-quant weights taking Q8_0-grain activations (`--q8-fine`)?
pub fn q8_fine() -> bool {
    Q8_FINE.load(Ordering::Relaxed)
}

/// Are K-quant weights taking ggml's Q8_K activations (the production default)?
pub fn q8k_activations() -> bool {
    !q8_fine()
}

pub fn act_kind(t: GgmlType) -> ActKind {
    match t {
        GgmlType::F32 => ActKind::F32,
        GgmlType::Q8_0 | GgmlType::Q4_0 => ActKind::Q8,
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => {
            if q8k_activations() {
                ActKind::Q8K
            } else {
                ActKind::Q8
            }
        }
        other => panic!("act_kind: no kernel for {other:?} weights"),
    }
}

/// The activation vector in one of its forms.
#[derive(Debug, Clone, Copy)]
pub enum Act<'a> {
    Q8(&'a Q8Row),
    Q8K(&'a Q8KRow),
    F32(&'a [f32]),
}

/// One activation vector with its quantised forms, each computed at most once (`OnceLock`), so every
/// projection that consumes the same input reuses the same quantisation.
pub struct ActVec<'a> {
    f: &'a [f32],
    q8: OnceLock<Q8Row>,
    q8k: OnceLock<Q8KRow>,
}

impl<'a> ActVec<'a> {
    pub fn new(f: &'a [f32]) -> ActVec<'a> {
        ActVec { f, q8: OnceLock::new(), q8k: OnceLock::new() }
    }
    pub fn len(&self) -> usize {
        self.f.len()
    }
    pub fn is_empty(&self) -> bool {
        self.f.is_empty()
    }
    pub fn f32(&self) -> &[f32] {
        self.f
    }
    pub fn q8(&self) -> &Q8Row {
        self.q8.get_or_init(|| Q8Row::quantize(self.f))
    }
    pub fn q8k(&self) -> &Q8KRow {
        self.q8k.get_or_init(|| Q8KRow::quantize(self.f))
    }
    /// The form `t` weights consume.
    pub fn act_for(&self, t: GgmlType) -> Act<'_> {
        match act_kind(t) {
            ActKind::F32 => Act::F32(self.f),
            ActKind::Q8 => Act::Q8(self.q8()),
            ActKind::Q8K => Act::Q8K(self.q8k()),
        }
    }
}

/// The forms `t` weights consume, for a batch of activations (`matmul`).
pub fn acts_for<'a>(acts: &'a [ActVec<'_>], t: GgmlType) -> Vec<Act<'a>> {
    acts.iter().map(|a| a.act_for(t)).collect()
}

/// A reusable activation buffer: the same thing `ActVec` holds, but owning its quantised rows so a decode
/// step can refill them without allocating (Phase 3.6). `fill` quantises into the forms the given weight
/// types consume; `act_for` then hands each projection its form. The f32 form is the caller's own slice,
/// which is why `act_for` takes it back.
pub struct ActBuf {
    q8: Q8Row,
    q8k: Q8KRow,
    has_q8: bool,
    has_q8k: bool,
}

impl ActBuf {
    /// Room for a row of `n` values in either form; `fill` on a row of that length never allocates.
    pub fn with_capacity(n: usize) -> ActBuf {
        ActBuf { q8: Q8Row::with_capacity(n), q8k: Q8KRow::with_capacity(n), has_q8: false, has_q8k: false }
    }

    /// Bytes held by both rows (capacities), for the memory plan (`tier::act_buf_bytes` is the formula).
    pub fn bytes(&self) -> usize {
        self.q8.bytes() + self.q8k.bytes()
    }

    /// Quantise `x` into every form the weight types in `types` consume, each at most once.
    pub fn fill(&mut self, x: &[f32], types: &[GgmlType]) {
        self.has_q8 = false;
        self.has_q8k = false;
        for &t in types {
            match act_kind(t) {
                ActKind::Q8 if !self.has_q8 => {
                    self.q8.quantize_into(x);
                    self.has_q8 = true;
                }
                ActKind::Q8K if !self.has_q8k => {
                    self.q8k.quantize_into(x);
                    self.has_q8k = true;
                }
                _ => {}
            }
        }
    }

    /// The activation `t` weights consume. `x` must be the slice the last `fill` was given.
    pub fn act_for<'a>(&'a self, t: GgmlType, x: &'a [f32]) -> Act<'a> {
        match act_kind(t) {
            ActKind::F32 => Act::F32(x),
            ActKind::Q8 => {
                assert!(self.has_q8, "ActBuf: fill() was not told about a {t:?} weight");
                Act::Q8(&self.q8)
            }
            ActKind::Q8K => {
                assert!(self.has_q8k, "ActBuf: fill() was not told about a {t:?} weight");
                Act::Q8K(&self.q8k)
            }
        }
    }
}

/// f32 weight bytes times f32 activations without an intermediate copy when the row is 4-byte aligned.
fn dot_f32_bytes(row: &[u8], x: &[f32]) -> f32 {
    // SAFETY: every bit pattern is a valid f32; `align_to` checks the alignment.
    let (pre, mid, post) = unsafe { row.align_to::<f32>() };
    if pre.is_empty() && post.is_empty() {
        dot_f32(mid, x)
    } else {
        let v: Vec<f32> = row.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        dot_f32(&v, x)
    }
}

#[inline]
fn one_row(w: &WeightMat, x: Act<'_>, r: usize) -> f32 {
    let row = w.row(r);
    match (w.ggml_type, x) {
        (GgmlType::F32, Act::F32(xf)) => dot_f32_bytes(row, xf),
        (GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K, Act::Q8K(q)) => dot_q8k(w.ggml_type, row, q),
        // every quantised type with the default Q8_0 activation, and F32 weights with a quantised input
        (_, Act::Q8(q)) => dot_q8(w.ggml_type, row, q),
        (t, Act::Q8K(_)) => panic!("matvec: {t:?} weights do not take a Q8_K activation row"),
        (t, Act::F32(_)) => panic!("matvec: {t:?} weights need a quantised activation row"),
    }
}

fn check_act(w: &WeightMat, x: Act<'_>) {
    let n = match x {
        Act::Q8(q) => q.n,
        Act::Q8K(q) => q.n,
        Act::F32(f) => f.len(),
    };
    assert_eq!(n, w.cols, "matvec: activation length");
}

/// `*mut f32` that may be shared with the pool: participants write disjoint index ranges.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// `y[i] = v`. Caller guarantees `i` is inside the slice and owned by this participant.
    #[inline]
    unsafe fn set(&self, i: usize, v: f32) {
        *self.0.add(i) = v;
    }
}

/// Contiguous row range of participant `tid` of `n` over `rows` rows.
#[inline]
pub fn row_range(rows: usize, tid: usize, n: usize) -> (usize, usize) {
    let chunk = rows.div_ceil(n);
    let start = (tid * chunk).min(rows);
    (start, ((tid + 1) * chunk).min(rows))
}

/// `y[r] = w[r] . x` for `r` in `a..b`, the participant's contiguous range. (Phase 3.5 tried two rows per
/// pass here, sharing the activation loads between a row pair; measured interleaved against this loop it was
/// slower on every kernel, `docs/data/two_row_ab.log`, so rows stay one at a time.)
#[inline]
fn rows_into(w: &WeightMat, x: Act<'_>, a: usize, b: usize, y: SendPtr) {
    for r in a..b {
        // SAFETY: `a..b` is this participant's disjoint range inside `y`.
        unsafe { y.set(r, one_row(w, x, r)) };
    }
}

/// `y[r] = w[r] . x` for every row, using up to `threads` participants over contiguous row ranges.
pub fn matvec(w: &WeightMat, x: Act<'_>, y: &mut [f32], threads: usize) {
    crate::prof_scope!(crate::prof::Stage::Matvec);
    assert_eq!(y.len(), w.rows, "matvec: output length");
    check_act(w, x);
    let yp = SendPtr(y.as_mut_ptr());
    pool::global().run(threads.min(w.rows.max(1)), &|tid, n| {
        let (a, b) = row_range(w.rows, tid, n);
        rows_into(w, x, a, b, yp);
    });
}

/// Two matrices with the same shape over the same input in one dispatch (the MLP's gate and up): each
/// participant streams its rows of the first, then of the second, so the two outputs cost one dispatch.
pub fn matvec2(w1: &WeightMat, w2: &WeightMat, x1: Act<'_>, x2: Act<'_>, y1: &mut [f32], y2: &mut [f32], threads: usize) {
    crate::prof_scope!(crate::prof::Stage::Matvec);
    assert_eq!((w1.rows, w1.cols), (w2.rows, w2.cols), "matvec2: shapes differ");
    assert_eq!(y1.len(), w1.rows, "matvec2: output length");
    assert_eq!(y2.len(), w2.rows, "matvec2: output length");
    check_act(w1, x1);
    check_act(w2, x2);
    let (p1, p2) = (SendPtr(y1.as_mut_ptr()), SendPtr(y2.as_mut_ptr()));
    pool::global().run(threads.min(w1.rows.max(1)), &|tid, n| {
        let (a, b) = row_range(w1.rows, tid, n);
        rows_into(w1, x1, a, b, p1);
        rows_into(w2, x2, a, b, p2);
    });
}

/// Batched form (prefill): `y[t * rows + r] = w[r] . xs[t]` for `t` in `0..xs.len()`. Each participant walks
/// its rows once and applies every activation to the row while it is in cache, so the weights are read
/// from memory once per batch instead of once per token. Row `r` of token `t` is the same arithmetic as
/// `matvec` on `xs[t]` alone.
pub fn matmul(w: &WeightMat, xs: &[Act<'_>], y: &mut [f32], threads: usize) {
    crate::prof_scope!(crate::prof::Stage::Matvec);
    let t_len = xs.len();
    assert_eq!(y.len(), t_len * w.rows, "matmul: output length");
    for x in xs {
        check_act(w, *x);
    }
    let yp = SendPtr(y.as_mut_ptr());
    pool::global().run(threads.min(w.rows.max(1)), &|tid, n| {
        let (a, b) = row_range(w.rows, tid, n);
        for r in a..b {
            for (t, x) in xs.iter().enumerate() {
                // SAFETY: as in `matvec`; row ranges are disjoint for every `t`.
                unsafe { yp.set(t * w.rows + r, one_row(w, *x, r)) };
            }
        }
    });
}

/// `matmul` over activations produced by a closure (`act(t)` for `t` in `0..n`), so a caller holding `n`
/// preallocated `ActBuf`s can run the batched form without building a `Vec<Act>` (Phase 5: the verification
/// batch and the MTP re-feed are allocation-free). Same arithmetic as `matmul`, row for row.
pub fn matmul_fn<'a>(w: &WeightMat, n: usize, act: &(dyn Fn(usize) -> Act<'a> + Sync + 'a), y: &mut [f32], threads: usize) {
    crate::prof_scope!(crate::prof::Stage::Matvec);
    assert_eq!(y.len(), n * w.rows, "matmul_fn: output length");
    for t in 0..n {
        check_act(w, act(t));
    }
    let yp = SendPtr(y.as_mut_ptr());
    pool::global().run(threads.min(w.rows.max(1)), &|tid, nthr| {
        let (a, b) = row_range(w.rows, tid, nthr);
        for r in a..b {
            for t in 0..n {
                // SAFETY: as in `matvec`; row ranges are disjoint for every `t`.
                unsafe { yp.set(t * w.rows + r, one_row(w, act(t), r)) };
            }
        }
    });
}

/// Convenience: quantise `x` once (in the form `w` consumes) and run the matvec.
pub fn matvec_f32_in(w: &WeightMat, x: &[f32], y: &mut [f32], threads: usize) {
    let a = ActVec::new(x);
    matvec(w, a.act_for(w.ggml_type), y, threads);
}

/// `matvec` on an already-wrapped activation.
pub fn matvec_act(w: &WeightMat, x: &ActVec<'_>, y: &mut [f32], threads: usize) {
    matvec(w, x.act_for(w.ggml_type), y, threads);
}
