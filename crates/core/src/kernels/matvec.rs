//! Matrix-vector products: rows of weights (ggml block layout or f32) times one activation vector, or
//! (`matmul`) times a batch of activation vectors with every weight byte read once. Rows are partitioned
//! into contiguous chunks over the persistent pool (`pool.rs`); every row is computed by exactly one
//! participant with the same kernel, so results are bit-identical for any thread count.
//!
//! Activations (Phase 3): every quantised weight type consumes Q8_0 rows (`q8.rs`; K-quant kernels in
//! `dot.rs` / `avx2.rs` with the Phase 3 lane structure), F32 weights consume f32. ggml's Q8_K rows
//! (`q8k.rs`, `kdot.rs`) are an opt-in for K-quant weights (`set_q8k_activations`): faster to quantise and
//! a little faster to multiply, but one scale per 256 values exceeds the frozen Phase 2b error ceilings
//! (`docs/kquant-dot.md`). `ActVec` holds one vector with each quantised form computed at most once, so the
//! projections that share an input share the quantisation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use super::dot::{dot_f32, dot_q8};
use super::kdot::dot_q8k;
use super::pool;
use super::q8::Q8Row;
use super::q8k::Q8KRow;
use crate::gguf::GgmlType;

/// A weight matrix of `rows` rows, each `cols` elements, stored as ggml rows (`row_bytes` each).
#[derive(Debug, Clone)]
pub struct WeightMat {
    pub ggml_type: GgmlType,
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
    /// Raw ggml bytes (`rows * row_bytes`), also for F32 (little-endian f32).
    pub data: Vec<u8>,
}

impl WeightMat {
    pub fn new(ggml_type: GgmlType, rows: usize, cols: usize, data: Vec<u8>) -> WeightMat {
        let (bs, ts) = ggml_type.block_layout();
        assert!((cols as u64).is_multiple_of(bs), "WeightMat: cols {cols} not a multiple of block size {bs} for {ggml_type:?}");
        let row_bytes = (cols as u64 / bs * ts) as usize;
        assert_eq!(data.len(), rows * row_bytes, "WeightMat: data length");
        WeightMat { ggml_type, rows, cols, row_bytes, data }
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
        &self.data[r * self.row_bytes..(r + 1) * self.row_bytes]
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

static Q8K_ACTS: AtomicBool = AtomicBool::new(false);

/// Opt in to (or out of) Q8_K activations for K-quant weights. Off by default (see the module doc).
pub fn set_q8k_activations(on: bool) {
    Q8K_ACTS.store(on, Ordering::Relaxed);
}

pub fn q8k_activations() -> bool {
    Q8K_ACTS.load(Ordering::Relaxed)
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

/// `y[r] = w[r] . x` for every row, using up to `threads` participants over contiguous row ranges.
pub fn matvec(w: &WeightMat, x: Act<'_>, y: &mut [f32], threads: usize) {
    assert_eq!(y.len(), w.rows, "matvec: output length");
    check_act(w, x);
    let yp = SendPtr(y.as_mut_ptr());
    pool::global().run(threads.min(w.rows.max(1)), &|tid, n| {
        let (a, b) = row_range(w.rows, tid, n);
        for r in a..b {
            // SAFETY: ranges of different participants are disjoint and inside `y`.
            unsafe { yp.set(r, one_row(w, x, r)) };
        }
    });
}

/// Two matrices with the same shape over the same input in one pass (the MLP's gate and up): each
/// participant streams its rows of both, so the two outputs cost one dispatch and one walk of the input.
pub fn matvec2(w1: &WeightMat, w2: &WeightMat, x: &ActVec<'_>, y1: &mut [f32], y2: &mut [f32], threads: usize) {
    assert_eq!((w1.rows, w1.cols), (w2.rows, w2.cols), "matvec2: shapes differ");
    assert_eq!(y1.len(), w1.rows, "matvec2: output length");
    assert_eq!(y2.len(), w2.rows, "matvec2: output length");
    let x1 = x.act_for(w1.ggml_type);
    let x2 = x.act_for(w2.ggml_type);
    check_act(w1, x1);
    check_act(w2, x2);
    let (p1, p2) = (SendPtr(y1.as_mut_ptr()), SendPtr(y2.as_mut_ptr()));
    pool::global().run(threads.min(w1.rows.max(1)), &|tid, n| {
        let (a, b) = row_range(w1.rows, tid, n);
        for r in a..b {
            // SAFETY: as in `matvec`.
            unsafe {
                p1.set(r, one_row(w1, x1, r));
                p2.set(r, one_row(w2, x2, r));
            }
        }
    });
}

/// Batched form (prefill): `y[t * rows + r] = w[r] . xs[t]` for `t` in `0..xs.len()`. Each participant walks
/// its rows once and applies every activation to the row while it is in cache, so the weights are read
/// from memory once per batch instead of once per token. Row `r` of token `t` is the same arithmetic as
/// `matvec` on `xs[t]` alone.
pub fn matmul(w: &WeightMat, xs: &[Act<'_>], y: &mut [f32], threads: usize) {
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

/// Convenience: quantise `x` once (in the form `w` consumes) and run the matvec.
pub fn matvec_f32_in(w: &WeightMat, x: &[f32], y: &mut [f32], threads: usize) {
    let a = ActVec::new(x);
    matvec(w, a.act_for(w.ggml_type), y, threads);
}

/// `matvec` on an already-wrapped activation.
pub fn matvec_act(w: &WeightMat, x: &ActVec<'_>, y: &mut [f32], threads: usize) {
    matvec(w, x.act_for(w.ggml_type), y, threads);
}
