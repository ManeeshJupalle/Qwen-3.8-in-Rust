//! Q8_K activation blocks (ggml `block_q8_K`, Phase 3): 256 values per block, one f32 scale `d`, int8 codes,
//! and the sixteen 16-element code sums `bsums` that the Q4_K / Q5_K kernels use to fold the sub-block mins
//! and the Q6_K kernel uses to fold the -32 offset (`docs/kquant-dot.md`).
//!
//! Bit-exact with ggml's `quantize_row_q8_K_ref` (ggml-quants.c) on finite input:
//! `max` = the signed element with the largest |x| (the first one on ties, strict `>` scan);
//! `iscale = -127 / max` (f32 divide); `q = min(127, nearest_even(iscale * x))` (one f32 multiply, then
//! ggml's `nearest_int`, which is round-half-to-even for |v| < 2^22, i.e. `f32::round_ties_even`);
//! `d = 1 / iscale` (f32 divide, so `d` carries the sign of `-max`); `bsums[g] = sum(q[16g..16g+16])`.
//! An all-zero block stores `d = 0` and zero codes. Unlike Q8_0 the scale is f32, not f16.
//! `q8s[s] = bsums[2s] + bsums[2s+1]` (the 8 sub-block sums) is derived once here so the Q4_K / Q5_K min
//! term is one `madd` per super-block; it is not part of the ggml layout.

/// One row of Q8_K blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct Q8KRow {
    pub n: usize,
    pub d: Vec<f32>,
    pub qs: Vec<i8>,
    pub bsums: Vec<i16>,
    /// Derived: `bsums[2s] + bsums[2s+1]`, 8 per block.
    pub q8s: Vec<i16>,
}

impl Q8KRow {
    pub const BLOCK: usize = 256;
    /// ggml layout per block: f32 d, 256 x i8, 16 x i16.
    pub const BLOCK_BYTES: usize = 4 + 256 + 32;

    /// An empty row with room for `n` values, so `quantize_into` never allocates (Phase 3.6).
    pub fn with_capacity(n: usize) -> Q8KRow {
        let nb = n / Self::BLOCK;
        Q8KRow { n: 0, d: Vec::with_capacity(nb), qs: Vec::with_capacity(n), bsums: Vec::with_capacity(nb * 16), q8s: Vec::with_capacity(nb * 8) }
    }

    /// Quantise `x` (length a multiple of 256). AVX2 when available (bit-identical, `avx2.rs`), else scalar.
    pub fn quantize(x: &[f32]) -> Q8KRow {
        let mut r = Q8KRow::with_capacity(x.len());
        r.quantize_into(x);
        r
    }

    /// Quantise `x` into this row, reusing its buffers: no allocation once the capacity is right
    /// (`with_capacity`), which is what makes the decode path allocation-free (Phase 3.6, finding 52).
    /// Same bits as `quantize`.
    pub fn quantize_into(&mut self, x: &[f32]) {
        crate::prof_scope!(crate::prof::Stage::Quantise);
        #[cfg(target_arch = "x86_64")]
        if super::simd::use_avx2() {
            assert!(x.len().is_multiple_of(Self::BLOCK), "Q8KRow::quantize: length {} is not a multiple of 256", x.len());
            self.d.clear();
            self.qs.clear();
            self.bsums.clear();
            // SAFETY: `use_avx2` is only true when the CPU reports AVX2 and F16C.
            unsafe { super::avx2::quantize_row_q8k(x, &mut self.d, &mut self.qs, &mut self.bsums) };
            self.n = x.len();
            pair_sums_into(&self.bsums, &mut self.q8s);
            return;
        }
        self.quantize_scalar_into(x);
    }

    /// The scalar reference quantiser (the port of `quantize_row_q8_K_ref`).
    pub fn quantize_scalar(x: &[f32]) -> Q8KRow {
        let mut r = Q8KRow::with_capacity(x.len());
        r.quantize_scalar_into(x);
        r
    }

    /// The scalar quantiser, in place.
    pub fn quantize_scalar_into(&mut self, x: &[f32]) {
        assert!(x.len().is_multiple_of(Self::BLOCK), "Q8KRow::quantize: length {} is not a multiple of 256", x.len());
        let nb = x.len() / Self::BLOCK;
        self.d.clear();
        self.qs.clear();
        self.bsums.clear();
        let (d, qs, bsums) = (&mut self.d, &mut self.qs, &mut self.bsums);
        for b in 0..nb {
            let blk = &x[b * Self::BLOCK..(b + 1) * Self::BLOCK];
            let mut amax = 0f32;
            let mut max = 0f32;
            for &v in blk {
                let a = v.abs();
                if a > amax {
                    amax = a;
                    max = v;
                }
            }
            if amax == 0.0 {
                d.push(0.0);
                qs.extend(std::iter::repeat_n(0i8, Self::BLOCK));
                bsums.extend(std::iter::repeat_n(0i16, 16));
                continue;
            }
            let iscale = -127.0f32 / max;
            let base = qs.len();
            for &v in blk {
                let q = (iscale * v).round_ties_even() as i32;
                qs.push(q.min(127) as i8);
            }
            for g in 0..16 {
                let s: i32 = qs[base + 16 * g..base + 16 * g + 16].iter().map(|&q| q as i32).sum();
                bsums.push(s as i16);
            }
            d.push(1.0 / iscale);
        }
        self.n = x.len();
        pair_sums_into(&self.bsums, &mut self.q8s);
    }

    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }

    /// Codes of block `b`.
    #[inline]
    pub fn block(&self, b: usize) -> &[i8] {
        &self.qs[b * Self::BLOCK..(b + 1) * Self::BLOCK]
    }

    /// The 16 group sums of block `b`.
    #[inline]
    pub fn bsums(&self, b: usize) -> &[i16] {
        &self.bsums[b * 16..(b + 1) * 16]
    }

    /// The 8 sub-block sums of block `b`.
    #[inline]
    pub fn q8s(&self, b: usize) -> &[i16] {
        &self.q8s[b * 8..(b + 1) * 8]
    }

    /// ggml `block_q8_K` layout: per block `f32 d`, 256 bytes, 16 x `i16 bsums` (little-endian).
    pub fn to_ggml_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.n_blocks() * Self::BLOCK_BYTES);
        for b in 0..self.n_blocks() {
            out.extend_from_slice(&self.d[b].to_le_bytes());
            out.extend(self.block(b).iter().map(|&q| q as u8));
            for &s in self.bsums(b) {
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
        out
    }

    /// Parse the ggml layout (e.g. a fixture written by the Python port).
    pub fn from_ggml_bytes(bytes: &[u8], n: usize) -> Q8KRow {
        assert!(n.is_multiple_of(Self::BLOCK));
        let nb = n / Self::BLOCK;
        assert_eq!(bytes.len(), nb * Self::BLOCK_BYTES, "Q8KRow::from_ggml_bytes: byte length");
        let mut d = Vec::with_capacity(nb);
        let mut qs = Vec::with_capacity(n);
        let mut bsums = Vec::with_capacity(nb * 16);
        for b in 0..nb {
            let blk = &bytes[b * Self::BLOCK_BYTES..(b + 1) * Self::BLOCK_BYTES];
            d.push(f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]));
            qs.extend(blk[4..260].iter().map(|&x| x as i8));
            for g in 0..16 {
                bsums.push(i16::from_le_bytes([blk[260 + 2 * g], blk[261 + 2 * g]]));
            }
        }
        let q8s = pair_sums(&bsums);
        Q8KRow { n, d, qs, bsums, q8s }
    }

    /// `d * q` per value (ggml `dequantize_row_q8_K` order).
    pub fn dequantize(&self) -> Vec<f32> {
        let mut y = Vec::with_capacity(self.n);
        for b in 0..self.n_blocks() {
            let d = self.d[b];
            for &q in self.block(b) {
                y.push(d * q as f32);
            }
        }
        y
    }
}

/// `bsums[2s] + bsums[2s+1]` for every block (exact: |bsums| <= 2032).
fn pair_sums(bsums: &[i16]) -> Vec<i16> {
    bsums.chunks_exact(2).map(|p| p[0] + p[1]).collect()
}

/// `pair_sums` into an existing buffer (reuses its capacity).
fn pair_sums_into(bsums: &[i16], out: &mut Vec<i16>) {
    out.clear();
    out.extend(bsums.chunks_exact(2).map(|p| p[0] + p[1]));
}
