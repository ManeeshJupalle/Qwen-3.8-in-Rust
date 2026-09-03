//! Activation quantisation to Q8_0 blocks (32 values, f16 scale, int8 codes), bit-exact with ggml's
//! `quantize_row_q8_0_ref` and gguf-py's `Q8_0.quantize_blocks`:
//! `amax = max |x|` (left to right), `d = amax / 127` (f32), `id = d != 0 ? 1/d : 0` (from the unrounded d),
//! `q = round_half_away_from_zero(x * id)`, stored `d` is `f16(d)`.

use crate::quant::f16_to_f32;

/// One row of Q8_0 blocks. `d` holds the f16 bits of each block scale; `qs` the int8 codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q8Row {
    pub n: usize,
    pub d: Vec<u16>,
    pub qs: Vec<i8>,
}

impl Q8Row {
    pub const BLOCK: usize = 32;

    /// Quantise `x` (length a multiple of 32). AVX2 when available (bit-identical, `avx2.rs`), else scalar.
    pub fn quantize(x: &[f32]) -> Q8Row {
        #[cfg(target_arch = "x86_64")]
        if super::simd::use_avx2() {
            assert!(x.len().is_multiple_of(Self::BLOCK), "Q8Row::quantize: length {} is not a multiple of 32", x.len());
            let mut d = Vec::with_capacity(x.len() / Self::BLOCK);
            let mut qs = Vec::with_capacity(x.len());
            // SAFETY: `use_avx2` is only true when the CPU reports AVX2 and F16C.
            unsafe { super::avx2::quantize_row(x, &mut d, &mut qs) };
            return Q8Row { n: x.len(), d, qs };
        }
        Self::quantize_scalar(x)
    }

    /// The scalar reference quantiser.
    pub fn quantize_scalar(x: &[f32]) -> Q8Row {
        assert!(x.len().is_multiple_of(Self::BLOCK), "Q8Row::quantize: length {} is not a multiple of 32", x.len());
        let nb = x.len() / Self::BLOCK;
        let mut d = Vec::with_capacity(nb);
        let mut qs = Vec::with_capacity(x.len());
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
            d.push(half::f16::from_f32(dd).to_bits());
            for &v in blk {
                qs.push((v * id).round() as i8);
            }
        }
        Q8Row { n: x.len(), d, qs }
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
        Q8Row { n, d, qs }
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
