//! Activation quantisation to Q8_0 blocks (32 values, f16 scale, int8 codes), bit-exact with ggml's
//! `quantize_row_q8_0_ref` and gguf-py's `Q8_0.quantize_blocks`:
//! `amax = max |x|` (left to right), `d = amax / 127` (f32), `id = d != 0 ? 1/d : 0` (from the unrounded d),
//! `q = round_half_away_from_zero(x * id)`, stored `d` is `f16(d)`.
//!
//! Phase 3: the row also carries what the K-quant kernels (`docs/kquant-dot.md`) read per block, derived once
//! per quantisation: `d32` (the scale as f32), `dxs = d32 * sum(q)` (exact: 11 x 12 significant bits) for the
//! Q4_K / Q5_K mins, and `off6` (`[32 * sum(q[0..16]), 0, 0, 0, 32 * sum(q[16..32]), 0, 0, 0]`) for the
//! Q6_K offset. None of it is part of the ggml layout.

use crate::quant::f16_to_f32;

/// One row of Q8_0 blocks. `d` holds the f16 bits of each block scale; `qs` the int8 codes.
#[derive(Debug, Clone, PartialEq)]
pub struct Q8Row {
    pub n: usize,
    pub d: Vec<u16>,
    pub qs: Vec<i8>,
    /// Derived: block scales as f32.
    pub d32: Vec<f32>,
    /// Derived: `d32 * sum(q)` per block.
    pub dxs: Vec<f32>,
    /// Derived: 8 i32 per block, the Q6_K offset lanes.
    pub off6: Vec<i32>,
}

impl Q8Row {
    pub const BLOCK: usize = 32;

    /// An empty row with room for `n` values, so `quantize_into` never allocates (Phase 3.6).
    pub fn with_capacity(n: usize) -> Q8Row {
        let nb = n / Self::BLOCK;
        Q8Row {
            n: 0,
            d: Vec::with_capacity(nb),
            qs: Vec::with_capacity(n),
            d32: Vec::with_capacity(nb),
            dxs: Vec::with_capacity(nb),
            off6: Vec::with_capacity(nb * 8),
        }
    }

    /// Bytes this row's buffers hold (capacities, i.e. what `with_capacity` reserved), for the memory plan.
    pub fn bytes(&self) -> usize {
        self.d.capacity() * 2 + self.qs.capacity() + self.d32.capacity() * 4 + self.dxs.capacity() * 4 + self.off6.capacity() * 4
    }

    /// Quantise `x` (length a multiple of 32). AVX2 when available (bit-identical, `avx2.rs`), else scalar.
    pub fn quantize(x: &[f32]) -> Q8Row {
        let mut r = Q8Row::with_capacity(x.len());
        r.quantize_into(x);
        r
    }

    /// The scalar reference quantiser.
    pub fn quantize_scalar(x: &[f32]) -> Q8Row {
        let mut r = Q8Row::with_capacity(x.len());
        r.quantize_scalar_into(x);
        r
    }

    /// Quantise `x` into this row, reusing its buffers: no allocation once the capacity is right
    /// (`with_capacity`), which is what makes the decode path allocation-free (Phase 3.6, finding 52).
    /// Same bits as `quantize`.
    pub fn quantize_into(&mut self, x: &[f32]) {
        crate::prof_scope!(crate::prof::Stage::Quantise);
        #[cfg(target_arch = "x86_64")]
        if super::simd::use_avx2() {
            assert!(x.len().is_multiple_of(Self::BLOCK), "Q8Row::quantize: length {} is not a multiple of 32", x.len());
            self.d.clear();
            self.qs.clear();
            // SAFETY: `use_avx2` is only true when the CPU reports AVX2 and F16C.
            unsafe { super::avx2::quantize_row(x, &mut self.d, &mut self.qs) };
            self.derive(x.len());
            return;
        }
        self.quantize_scalar_into(x);
    }

    /// The scalar quantiser, in place.
    pub fn quantize_scalar_into(&mut self, x: &[f32]) {
        assert!(x.len().is_multiple_of(Self::BLOCK), "Q8Row::quantize: length {} is not a multiple of 32", x.len());
        let nb = x.len() / Self::BLOCK;
        self.d.clear();
        self.qs.clear();
        for b in 0..nb {
            let blk = &x[b * Self::BLOCK..(b + 1) * Self::BLOCK];
            let mut amax = 0f32;
            for &v in blk {
                let a = v.abs();
                if a > amax {
                    amax = a;
                }
            }
            let dd = amax / 127.0;
            let id = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            self.d.push(half::f16::from_f32(dd).to_bits());
            for &v in blk {
                self.qs.push((v * id).round() as i8);
            }
        }
        self.derive(x.len());
    }

    /// Recompute the derived per-block data (`d32`, `dxs`, `off6`) from `d` and `qs`.
    fn derive(&mut self, n: usize) {
        let nb = self.d.len();
        assert_eq!(self.qs.len(), nb * Self::BLOCK);
        self.n = n;
        self.d32.clear();
        self.dxs.clear();
        self.off6.clear();
        for b in 0..nb {
            self.d32.push(f16_to_f32(self.d[b]));
            let q = &self.qs[b * Self::BLOCK..(b + 1) * Self::BLOCK];
            let lo: i32 = q[..16].iter().map(|&v| v as i32).sum();
            let hi: i32 = q[16..].iter().map(|&v| v as i32).sum();
            self.dxs.push(self.d32[b] * (lo + hi) as f32);
            self.off6.extend_from_slice(&[32 * lo, 0, 0, 0, 32 * hi, 0, 0, 0]);
        }
    }

    /// Assemble a row from its scales and codes and compute the derived per-block data.
    pub fn from_parts(n: usize, d: Vec<u16>, qs: Vec<i8>) -> Q8Row {
        let mut r = Q8Row { n, d, qs, d32: Vec::new(), dxs: Vec::new(), off6: Vec::new() };
        r.derive(n);
        r
    }

    /// Parse a ggml Q8_0 row (`n/32` blocks of `f16 d` + 32 bytes), e.g. from a fixture.
    pub fn from_ggml_bytes(bytes: &[u8], n: usize) -> Q8Row {
        assert!(n.is_multiple_of(Self::BLOCK));
        let nb = n / Self::BLOCK;
        assert_eq!(bytes.len(), nb * 34, "Q8Row::from_ggml_bytes: byte length");
        let mut d = Vec::with_capacity(nb);
        let mut qs = Vec::with_capacity(n);
        for b in 0..nb {
            let blk = &bytes[b * 34..(b + 1) * 34];
            d.push(u16::from_le_bytes([blk[0], blk[1]]));
            qs.extend(blk[2..34].iter().map(|&x| x as i8));
        }
        Q8Row::from_parts(n, d, qs)
    }

    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }

    /// Scale of block `b` as f32 (exact f16 -> f32).
    #[inline]
    pub fn scale(&self, b: usize) -> f32 {
        f16_to_f32(self.d[b])
    }

    /// Codes of block `b`.
    #[inline]
    pub fn block(&self, b: usize) -> &[i8] {
        &self.qs[b * Self::BLOCK..(b + 1) * Self::BLOCK]
    }

    /// ggml on-disk layout: per block `f16 d` then 32 bytes.
    pub fn to_ggml_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.n_blocks() * 34);
        for b in 0..self.n_blocks() {
            out.extend_from_slice(&self.d[b].to_le_bytes());
            out.extend(self.block(b).iter().map(|&q| q as u8));
        }
        out
    }

    /// `q * d` per value (gguf-py dequantize order).
    pub fn dequantize(&self) -> Vec<f32> {
        let mut y = Vec::with_capacity(self.n);
        for b in 0..self.n_blocks() {
            let d = self.scale(b);
            for &q in self.block(b) {
                y.push(q as f32 * d);
            }
        }
        y
    }
}
