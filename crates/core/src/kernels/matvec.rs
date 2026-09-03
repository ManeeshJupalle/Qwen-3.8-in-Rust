//! Matrix-vector product: rows of weights (ggml block layout or f32) times one activation vector.
//! This is the shape every projection uses. Parallel over rows with `std::thread::scope`; every row is
//! computed by exactly one thread with the same scalar kernel, so the result is bit-identical for any
//! thread count.

use super::dot::{dot_f32, dot_q8};
use super::q8::Q8Row;
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
}

/// The activation vector: Q8_0 blocks for quantised weights, or plain f32 (accepted by F32 weights only).
#[derive(Debug, Clone, Copy)]
pub enum Act<'a> {
    Q8(&'a Q8Row),
    F32(&'a [f32]),
}

fn one_row(w: &WeightMat, x: Act<'_>, r: usize) -> f32 {
    match (w.ggml_type, x) {
        (GgmlType::F32, Act::F32(xf)) => {
            let row = w.row_f32(r);
            dot_f32(&row, xf)
        }
        (_, Act::Q8(q)) => {
            assert_eq!(q.n, w.cols, "matvec: activation length");
            dot_q8(w.ggml_type, w.row(r), q)
        }
        (t, Act::F32(_)) => panic!("matvec: {t:?} weights need a Q8 activation row"),
    }
}

/// `y[r] = w[r] . x` for every row, using up to `threads` threads over contiguous row ranges.
pub fn matvec(w: &WeightMat, x: Act<'_>, y: &mut [f32], threads: usize) {
    assert_eq!(y.len(), w.rows, "matvec: output length");
    let threads = threads.max(1).min(w.rows.max(1));
    if threads == 1 || w.rows < 2 {
        for (r, out) in y.iter_mut().enumerate() {
            *out = one_row(w, x, r);
        }
        return;
    }
    let chunk = w.rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (ci, ys) in y.chunks_mut(chunk).enumerate() {
            let start = ci * chunk;
            s.spawn(move || {
                for (i, out) in ys.iter_mut().enumerate() {
                    *out = one_row(w, x, start + i);
                }
            });
        }
    });
}

/// Convenience: quantise `x` once and run the matvec (what quantised layers do).
pub fn matvec_f32_in(w: &WeightMat, x: &[f32], y: &mut [f32], threads: usize) {
    if w.ggml_type == GgmlType::F32 {
        matvec(w, Act::F32(x), y, threads);
    } else {
        let q = Q8Row::quantize(x);
        matvec(w, Act::Q8(&q), y, threads);
    }
}
