//! AVX2 versions of the hot kernels (x86_64 only), selected at runtime by `super::simd::use_avx2()`.
//! The scalar kernels stay the reference; `tests/avx2.rs` compares the two on every kernel fixture and on
//! random rows.
//!
//! Every function here is bit-identical to its scalar counterpart on finite inputs (Phase 3: a property the
//! tests check, no longer a contract).
//! - `dot_q8_0_q8`, `dot_q4_0_q8`: the per-block integer sums are exact in both paths (i8 x i8 products widened
//!   through `maddubs`/`madd` with no saturation); the f32 expression per block is then evaluated with the same
//!   scalar operations, in the same order, without FMA, and block partials are accumulated left to right
//!   exactly as `dot.rs` documents.
//! - `dot_q4_k_q8`, `dot_q5_k_q8`, `dot_q6_k_q8` (Phase 3, `docs/kquant-dot.md`): exact i32 lane sums per
//!   sub-block, one f32 factor per sub-block into 8 f32 lanes (plus 8 min lanes), one reduction per row; the
//!   scalar reference in `dot.rs` computes the same lanes in the same order. Phase 3.5: the per-sub-block
//!   factors are computed once per super-block into a small stack table and read back with broadcast loads
//!   (no shuffles in the sub-block loop) and Q5_K consumes its high bits with a fixed shift; the arithmetic
//!   is unchanged, so the bits are. (Two-row forms were tried and measured slower, `docs/kquant-dot.md`.)
//! - `quantize_row` (Q8_0): the block max is order-independent, `d` and `id` are computed as in `q8.rs`,
//!   rounding is half-away-from-zero (ties fixed up after `round_ps`), the f16 scale comes from `half`.
//! - `inv_rms` and `softmax` sums: `rmsnorm.rs` / `softmax.rs` define their summation order as eight
//!   interleaved lanes reduced as `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))` then a scalar tail; the vector code
//!   is that order, so the results match bit for bit. `exp` stays scalar (libm) in both paths.
//!
//! F16C is required alongside AVX2 (every AVX2 CPU has it); `_mm_cvtph_ps` converts scales exactly like `half`.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::missing_safety_doc)]

use std::arch::x86_64::*;

use super::q8::Q8Row;
use super::q8k::Q8KRow;
use crate::gguf::GgmlType;
use crate::quant::{Q4_0_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES};

#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn f16_at(b: *const u8) -> f32 {
    let bits = u16::from_le_bytes([*b, *b.add(1)]);
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
}

#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn f16_bits(bits: u16) -> f32 {
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load(p: *const u8) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_x(x: &Q8Row, b: usize) -> __m256i {
    _mm256_loadu_si256(x.block(b).as_ptr() as *const __m256i)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum128_i32(v: __m128i) -> i32 {
    let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    hsum128_i32(_mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)))
}

/// Exact `sum(w[j] * x[j])` over 32 lanes for signed `w` (|w| <= 128) and signed `x` (|x| <= 127):
/// `maddubs(|w|, x * sign(w))` pairs are at most 2 * 128 * 127 = 32512 < 32767, so nothing saturates.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot32_i8(w: __m256i, x: __m256i) -> __m256i {
    let ax = _mm256_sign_epi8(x, w);
    let aw = _mm256_abs_epi8(w);
    _mm256_madd_epi16(_mm256_maddubs_epi16(aw, ax), _mm256_set1_epi16(1))
}

/// Q8_0 weights: per block `(d_w * d_x) * sum(q_w * q_x)`.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q8_0_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), nb * Q8_0_BLOCK_BYTES, "dot_q8_0_q8: weight row bytes");
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = w.as_ptr().add(b * Q8_0_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        let dw = f16_at(wb);
        let isum = hsum_i32(dot32_i8(load(wb.add(2)), load_x(x, b)));
        acc += (dw * f16_bits(x.d[b])) * isum as f32;
    }
    acc
}

/// Q4_0 weights: per block `(d_w * d_x) * sum((q_w - 8) * q_x)`; low nibbles are values 0..16, high 16..32.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q4_0_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), nb * Q4_0_BLOCK_BYTES, "dot_q4_0_q8: weight row bytes");
    let m4 = _mm_set1_epi8(0x0F);
    let eight = _mm256_set1_epi8(8);
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = w.as_ptr().add(b * Q4_0_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        let dw = f16_at(wb);
        let q = _mm_loadu_si128(wb.add(2) as *const __m128i);
        let lo = _mm_and_si128(q, m4);
        let hi = _mm_and_si128(_mm_srli_epi16(q, 4), m4);
        let wv = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), eight);
        let isum = hsum_i32(dot32_i8(wv, load_x(x, b)));
        acc += (dw * f16_bits(x.d[b])) * isum as f32;
    }
    acc
}

/// 8 lanes of `v[s]` (a lane permute; the Q8_0-grain Q5_K kernel keeps this form, see `docs/kquant-dot.md`).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lane_bcast(v: __m256, s: usize) -> __m256 {
    _mm256_permutevar8x32_ps(v, _mm256_set1_epi32(s as i32))
}

/// Factor `s` of a per-super-block table in all 8 lanes: one load-port uop (`vbroadcastss`) per sub-block
/// instead of a shuffle (Phase 3.5; the tables are written once per super-block with `storeu_ps`).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn bcast(tbl: *const f32, s: usize) -> __m256 {
    _mm256_broadcast_ss(&*tbl.add(s))
}

/// `[lo x4, hi x4]` from two table entries (Q6_K: the two 16-group factors of a sub-block).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn bcast_pair(lo: *const f32, hi: *const f32) -> __m256 {
    _mm256_blend_ps(_mm256_broadcast_ss(&*lo), _mm256_broadcast_ss(&*hi), 0xF0)
}

/// Exact i32 lane sums of a 32-element sub-block: lane `i` = elements `4i..4i+4` (unsigned codes <= 63).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lanes_u8(w: __m256i, x: __m256i) -> __m256i {
    _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1))
}

/// Shared Q4_K / Q5_K super-block header: `(d, dsc8 = d_x * (d * sc), accm + ((m * -dmin) * dxs))`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn k45_header(wb: *const u8, x: &Q8Row, sb: usize, accm: __m256) -> (__m256, __m256) {
    let d = f16_at(wb);
    let dmin = f16_at(wb.add(2));
    let sm = scales_mins_k4(wb.add(4));
    let sc8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sm)));
    let m8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sm, 1)));
    let dx8 = _mm256_loadu_ps(x.d32.as_ptr().add(sb * 8));
    let dsc8 = _mm256_mul_ps(dx8, _mm256_mul_ps(_mm256_set1_ps(d), sc8));
    let dxs8 = _mm256_loadu_ps(x.dxs.as_ptr().add(sb * 8));
    let accm = _mm256_add_ps(accm, _mm256_mul_ps(_mm256_mul_ps(m8, _mm256_set1_ps(-dmin)), dxs8));
    (dsc8, accm)
}

/// Q4_K weights x Q8_0 activations (Phase 3 structure, bit-identical to `dot::dot_q4_k_q8`).
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q4_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q4_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * Q4_K_BLOCK_BYTES, "dot_q4_k_q8: weight row bytes");
    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    let mut accm = _mm256_setzero_ps();
    let mut tbl = [0f32; 8];
    for sb in 0..nb / 8 {
        let wb = w.as_ptr().add(sb * Q4_K_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        prefetch(wb.add(PF_DIST + 48));
        prefetch(wb.add(PF_DIST + 96));
        let (dsc8, am) = k45_header(wb, x, sb, accm);
        accm = am;
        _mm256_storeu_ps(tbl.as_mut_ptr(), dsc8);
        let q4 = wb.add(16);
        let q8 = x.qs.as_ptr().add(sb * 256) as *const u8;
        for j in 0..4 {
            let q4bits = load(q4.add(32 * j));
            let pl = lanes_u8(_mm256_and_si256(q4bits, m4), load(q8.add(64 * j)));
            let ph = lanes_u8(_mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4), load(q8.add(64 * j + 32)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(pl), bcast(tbl.as_ptr(), 2 * j)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(ph), bcast(tbl.as_ptr(), 2 * j + 1)));
        }
    }
    reduce8(acc) + reduce8(accm)
}

/// Q5_K weights x Q8_0 activations: as Q4_K plus the fifth bit from `qh` (bit `s` of every byte). Kept in
/// its Phase 3 form (lane permute, constant shifts): the 3.5 table + bit-serial form measured 30 % slower
/// single-threaded for this kernel while the same change made `dot_q5_k_q8k` 60 % faster (`docs/kquant-dot.md`).
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q5_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q5_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * Q5_K_BLOCK_BYTES, "dot_q5_k_q8: weight row bytes");
    let m4 = _mm256_set1_epi8(0x0F);
    let mone = _mm256_set1_epi8(1);
    let mut acc = _mm256_setzero_ps();
    let mut accm = _mm256_setzero_ps();
    for sb in 0..nb / 8 {
        let wb = w.as_ptr().add(sb * Q5_K_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        prefetch(wb.add(PF_DIST + 60));
        prefetch(wb.add(PF_DIST + 120));
        let (dsc8, am) = k45_header(wb, x, sb, accm);
        accm = am;
        let hbits = load(wb.add(16));
        let q5 = wb.add(48);
        let q8 = x.qs.as_ptr().add(sb * 256) as *const u8;
        for j in 0..4 {
            let q5bits = load(q5.add(32 * j));
            let h0 = _mm256_slli_epi16(_mm256_and_si256(shift_right_epi16(hbits, (2 * j) as i32), mone), 4);
            let h1 = _mm256_slli_epi16(_mm256_and_si256(shift_right_epi16(hbits, (2 * j + 1) as i32), mone), 4);
            let q5_0 = _mm256_add_epi8(_mm256_and_si256(q5bits, m4), h0);
            let q5_1 = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4), h1);
            let p0 = lanes_u8(q5_0, load(q8.add(64 * j)));
            let p1 = lanes_u8(q5_1, load(q8.add(64 * j + 32)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(p0), lane_bcast(dsc8, 2 * j)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(p1), lane_bcast(dsc8, 2 * j + 1)));
        }
    }
    reduce8(acc) + reduce8(accm)
}

/// The 16 factors of a Q6_K super-block for Q8_0-grain rows, into `tbl`: `tbl[s] = d_x[s] * (d * sc[2s])`
/// (lanes 0..4 of sub-block `s`, its first 16-group) and `tbl[8 + s] = d_x[s] * (d * sc[2s + 1])` (lanes 4..8).
/// The int8 scales are de-interleaved with one byte shuffle (even groups, then odd groups).
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn q6_factors(wb: *const u8, x: &Q8Row, sb: usize, deint: __m128i, tbl: *mut f32) {
    let d = _mm256_set1_ps(f16_at(wb.add(208)));
    let sc2 = _mm_shuffle_epi8(_mm_loadu_si128(wb.add(192) as *const __m128i), deint);
    let dsc_even = _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sc2)));
    let dsc_odd = _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(sc2, 8))));
    let dx8 = _mm256_loadu_ps(x.d32.as_ptr().add(sb * 8));
    _mm256_storeu_ps(tbl, _mm256_mul_ps(dx8, dsc_even));
    _mm256_storeu_ps(tbl.add(8), _mm256_mul_ps(dx8, dsc_odd));
}

/// The four unsigned 6-bit code vectors of one 128-code half of a Q6_K super-block (sub-blocks `4j..4j+4`):
/// low nibbles of `ql` (two 32-byte loads) plus bit pairs of `qh` moved to bits 4..6 (one mask, `0x30`).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q6_codes(wb: *const u8, j: usize, m15: __m256i, m48: __m256i) -> [__m256i; 4] {
    let ql = wb.add(64 * j);
    let q4bits1 = load(ql);
    let q4bits2 = load(ql.add(32));
    let qh = load(wb.add(128 + 32 * j));
    [
        _mm256_or_si256(_mm256_and_si256(q4bits1, m15), _mm256_and_si256(_mm256_slli_epi16(qh, 4), m48)),
        _mm256_or_si256(_mm256_and_si256(q4bits2, m15), _mm256_and_si256(_mm256_slli_epi16(qh, 2), m48)),
        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), _mm256_and_si256(qh, m48)),
        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), _mm256_and_si256(_mm256_srli_epi16(qh, 2), m48)),
    ]
}

/// Q6_K weights x Q8_0 activations: int8 group scales (two per sub-block), the -32 offset from `off6`.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q6_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q6_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * Q6_K_BLOCK_BYTES, "dot_q6_k_q8: weight row bytes");
    let m48 = _mm256_set1_epi8(0x30);
    let m15 = _mm256_set1_epi8(15);
    let deint = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    let mut acc = _mm256_setzero_ps();
    let mut tbl = [0f32; 16];
    for sb in 0..nb / 8 {
        let wb = w.as_ptr().add(sb * Q6_K_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        prefetch(wb.add(PF_DIST + 56));
        prefetch(wb.add(PF_DIST + 112));
        prefetch(wb.add(PF_DIST + 168));
        q6_factors(wb, x, sb, deint, tbl.as_mut_ptr());
        for j in 0..2 {
            let q6 = q6_codes(wb, j, m15, m48);
            for (k, q6k) in q6.iter().enumerate() {
                let s = 4 * j + k;
                let xb = sb * 8 + s;
                let p = lanes_u8(*q6k, load(x.qs.as_ptr().add(xb * 32) as *const u8));
                let p = _mm256_sub_epi32(p, _mm256_loadu_si256(x.off6.as_ptr().add(xb * 8) as *const __m256i));
                let f = bcast_pair(tbl.as_ptr().add(s), tbl.as_ptr().add(8 + s));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(p), f));
            }
        }
    }
    reduce8(acc)
}

/// `_mm256_srli_epi16` needs an immediate; the shift is 0..7 here.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shift_right_epi16(v: __m256i, n: i32) -> __m256i {
    match n {
        0 => v,
        1 => _mm256_srli_epi16(v, 1),
        2 => _mm256_srli_epi16(v, 2),
        3 => _mm256_srli_epi16(v, 3),
        4 => _mm256_srli_epi16(v, 4),
        5 => _mm256_srli_epi16(v, 5),
        6 => _mm256_srli_epi16(v, 6),
        _ => _mm256_srli_epi16(v, 7),
    }
}

/// Dispatch by weight type (quantised types only; F32 rows use the scalar f32 kernel).
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q8(t: GgmlType, w: &[u8], x: &Q8Row) -> Option<f32> {
    Some(match t {
        GgmlType::Q8_0 => dot_q8_0_q8(w, x),
        GgmlType::Q4_0 => dot_q4_0_q8(w, x),
        GgmlType::Q4_K => dot_q4_k_q8(w, x),
        GgmlType::Q5_K => dot_q5_k_q8(w, x),
        GgmlType::Q6_K => dot_q6_k_q8(w, x),
        _ => return None,
    })
}

/// Round half away from zero (what `f32::round` and ggml's `roundf` do): `round_ps` rounds ties to even, so
/// ties (|t - trunc(t)| == 0.5 exactly) are replaced by `trunc(t) + copysign(1, t)`. Exact for |t| < 2^23.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn round_half_away(t: __m256) -> __m256 {
    let signbit = _mm256_set1_ps(-0.0);
    let even = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let trunc = _mm256_round_ps(t, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    let frac = _mm256_andnot_ps(signbit, _mm256_sub_ps(t, trunc));
    let tie = _mm256_cmp_ps(frac, _mm256_set1_ps(0.5), _CMP_EQ_OQ);
    let away = _mm256_add_ps(trunc, _mm256_or_ps(_mm256_and_ps(t, signbit), _mm256_set1_ps(1.0)));
    _mm256_blendv_ps(even, away, tie)
}

/// Q8_0 quantisation of one row (length a multiple of 32) into `d` (f16 bits) and `qs`, bit-identical to
/// `Q8Row::quantize_scalar` on finite input.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn quantize_row(x: &[f32], d: &mut Vec<u16>, qs: &mut Vec<i8>) {
    assert!(x.len().is_multiple_of(32));
    let signbit = _mm256_set1_ps(-0.0);
    let lo_clamp = _mm256_set1_ps(-128.0);
    let hi_clamp = _mm256_set1_ps(127.0);
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let base = qs.len();
    qs.resize(base + x.len(), 0);
    for b in 0..x.len() / 32 {
        let p = x.as_ptr().add(b * 32);
        let v0 = _mm256_loadu_ps(p);
        let v1 = _mm256_loadu_ps(p.add(8));
        let v2 = _mm256_loadu_ps(p.add(16));
        let v3 = _mm256_loadu_ps(p.add(24));
        let a = _mm256_max_ps(
            _mm256_max_ps(_mm256_andnot_ps(signbit, v0), _mm256_andnot_ps(signbit, v1)),
            _mm256_max_ps(_mm256_andnot_ps(signbit, v2), _mm256_andnot_ps(signbit, v3)),
        );
        let m4 = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let amax = _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1)));
        let dd = amax / 127.0;
        let id = if dd != 0.0 { 1.0 / dd } else { 0.0 };
        d.push(half::f16::from_f32(dd).to_bits());
        let idv = _mm256_set1_ps(id);
        // `as i8` semantics of the scalar path: saturate infinities, NaN (0 * inf when the block scale
        // underflows and `id` overflows) becomes 0
        let q = |v: __m256| -> __m256i {
            let r = round_half_away(_mm256_mul_ps(v, idv));
            let clamped = _mm256_min_ps(_mm256_max_ps(r, lo_clamp), hi_clamp);
            let nan = _mm256_cmp_ps(r, r, _CMP_UNORD_Q);
            _mm256_cvtps_epi32(_mm256_andnot_ps(nan, clamped))
        };
        let i0 = _mm256_packs_epi32(q(v0), q(v1));
        let i2 = _mm256_packs_epi32(q(v2), q(v3));
        let i8s = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(i0, i2), perm);
        _mm256_storeu_si256(qs.as_mut_ptr().add(base + b * 32) as *mut __m256i, i8s);
    }
}

/// `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))` of an 8-lane accumulator: the reduction order `rmsnorm.rs` and
/// `softmax.rs` define for their scalar sums.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce8(acc: __m256) -> f32 {
    let s4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)))
}

/// `1 / sqrt(mean(x^2) + eps)` with the documented 8-lane summation order.
#[target_feature(enable = "avx2")]
pub unsafe fn inv_rms(x: &[f32], eps: f32) -> f32 {
    let n8 = x.len() / 8 * 8;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < n8 {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
        i += 8;
    }
    let mut ss = reduce8(acc);
    for &v in &x[n8..] {
        ss += v * v;
    }
    let mean = ss / x.len() as f32;
    1.0 / (mean + eps).sqrt()
}

/// `y = (x * inv) * w` (zero-centered RMSNorm with the GGUF weight).
#[target_feature(enable = "avx2")]
pub unsafe fn rmsnorm(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) {
    let n = x.len();
    assert_eq!(w.len(), n);
    assert_eq!(y.len(), n);
    let inv = inv_rms(x, eps);
    let invv = _mm256_set1_ps(inv);
    let n8 = n / 8 * 8;
    let mut i = 0;
    while i < n8 {
        let v = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), invv), _mm256_loadu_ps(w.as_ptr().add(i)));
        _mm256_storeu_ps(y.as_mut_ptr().add(i), v);
        i += 8;
    }
    for j in n8..n {
        y[j] = (x[j] * inv) * w[j];
    }
}

/// Softmax with max subtraction: vector max (exact), scalar `exp`, 8-lane sum in the documented order,
/// elementwise divide.
#[target_feature(enable = "avx2")]
pub unsafe fn softmax(x: &[f32], y: &mut [f32]) {
    let n = x.len();
    assert_eq!(y.len(), n);
    let n8 = n / 8 * 8;
    let mut mv = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i < n8 {
        mv = _mm256_max_ps(mv, _mm256_loadu_ps(x.as_ptr().add(i)));
        i += 8;
    }
    let m4 = _mm_max_ps(_mm256_castps256_ps128(mv), _mm256_extractf128_ps(mv, 1));
    let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    let mut m = _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1)));
    for &v in &x[n8..] {
        if v > m {
            m = v;
        }
    }
    for j in 0..n {
        y[j] = (x[j] - m).exp();
    }
    let mut acc = _mm256_setzero_ps();
    i = 0;
    while i < n8 {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(y.as_ptr().add(i)));
        i += 8;
    }
    let mut s = reduce8(acc);
    for &e in &y[n8..] {
        s += e;
    }
    let sv = _mm256_set1_ps(s);
    i = 0;
    while i < n8 {
        _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_div_ps(_mm256_loadu_ps(y.as_ptr().add(i)), sv));
        i += 8;
    }
    for v in &mut y[n8..] {
        *v /= s;
    }
}

// ------------------------------------------------------------------------------------ Q8_K path (Phase 3)
//
// K-quant weights times a Q8_K activation row, the ggml AVX2 kernels (`ggml-cpu/arch/x86/quants.c`,
// `ggml_vec_dot_q{4,5,6}_K_q8_K`) with `mul` + `add` instead of `fmadd`. Bit-identical to `kdot.rs`, which
// reproduces the same lane structure (see `docs/kquant-dot.md`): exact i32 lanes per super-block, one f32
// scale per super-block into 8 (and 4) f32 lanes, one reduction per row.

/// Software prefetch distance in bytes: a single thread streaming a row from DRAM is latency-bound without it
/// (3.3 GB/s vs 6.6 GB/s from cache for Q4_K on the i7-9750H); ~1 KB ahead covers the DRAM latency at the
/// kernel's consumption rate. Every 64-byte line ahead is touched by issuing prefetches at most 64 bytes apart.
const PF_DIST: usize = 1024;

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn prefetch(p: *const u8) {
    _mm_prefetch(p as *const i8, _MM_HINT_T0);
}

/// ggml's `get_scale_shuffle_k4`: row `i` (32 bytes) broadcasts i16 lane `i` of a 128-bit vector duplicated
/// in both halves (`(2i, 2i+1)` repeated), so `shuffle_epi8(scales, row_i)` is scale `i` in all 16 lanes.
#[repr(align(32))]
struct Shuf([u8; 256]);
static SHUF_K4: Shuf = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = ((i / 32) * 2 + (i % 2)) as u8;
        i += 1;
    }
    Shuf(t)
};

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn scale_shuf(i: usize) -> __m256i {
    _mm256_load_si256(SHUF_K4.0.as_ptr().add(32 * i) as *const __m256i)
}

/// The 12 packed scale/min bytes of a Q4_K / Q5_K super-block as 16 i16 lanes `sc[0..8], m[0..8]`, the
/// values `get_scale_min_k4` yields, unpacked with vector ops (ggml does it with scalar `utmp` masks):
/// bytes 0..8 masked to 6 bits are `sc[0..4], m[0..4]`; `sc[4+j]` is the low nibble of byte `8+j` with bits
/// 6..8 of byte `j` above it, `m[4+j]` the high nibble of byte `8+j` with bits 6..8 of byte `4+j` above it.
/// Reads 16 bytes: the 4 after the scales are the first code bytes of the same block.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn scales_mins_k4(p: *const u8) -> __m256i {
    let v = _mm_loadu_si128(p as *const __m128i);
    let lo = _mm_and_si128(v, _mm_set1_epi8(0x3F));
    // bits 6..8 of bytes 0..8 moved to bits 4..6 (16-bit shifts; the masks keep every step inside its byte)
    let top = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(v, 6), _mm_set1_epi8(0x03)), 4);
    // bytes 8..12 twice: low nibbles at 0..4 (sc[4..8]), high nibbles at 4..8 (m[4..8])
    let c = _mm_shuffle_epi8(v, _mm_setr_epi8(8, 9, 10, 11, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1));
    let cl = _mm_and_si128(c, _mm_set1_epi8(0x0F));
    let ch = _mm_and_si128(_mm_srli_epi16(c, 4), _mm_set1_epi8(0x0F));
    let hi = _mm_or_si128(_mm_blend_epi16(cl, ch, 0b1100), top);
    // lo = [sc0..4, m0..4, ..], hi = [sc4..8, m4..8, ..] -> [sc0..4, sc4..8, m0..4, m4..8]
    _mm256_cvtepu8_epi16(_mm_unpacklo_epi32(lo, hi))
}

/// `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))` (as `reduce8`).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_ps8(v: __m256) -> f32 {
    reduce8(v)
}

/// `(m0+m2)+(m1+m3)`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_ps4(m: __m128) -> f32 {
    let s2 = _mm_add_ps(m, _mm_movehl_ps(m, m));
    _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)))
}

/// Mins folded through the activation sub-block sums: `dmin * (m[2i] * q8s[2i] + m[2i+1] * q8s[2i+1])` into
/// 4 f32 lanes, with `q8s[s] = bsums[2s] + bsums[2s+1]` precomputed by the quantiser.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn mins_term(sm: __m256i, q8s: *const i16, dmin: f32, accm: __m128) -> __m128 {
    let prod = _mm_madd_epi16(_mm256_extracti128_si256(sm, 1), _mm_loadu_si128(q8s as *const __m128i));
    _mm_add_ps(accm, _mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)))
}

/// Q4_K x Q8_K.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q4_k_q8k(w: &[u8], x: &Q8KRow) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), nb * Q4_K_BLOCK_BYTES, "dot_q4_k_q8k: weight row bytes");
    let m4 = _mm256_set1_epi8(0xF);
    let mut acc = _mm256_setzero_ps();
    let mut accm = _mm_setzero_ps();
    for sb in 0..nb {
        let wb = w.as_ptr().add(sb * Q4_K_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        prefetch(wb.add(PF_DIST + 48));
        prefetch(wb.add(PF_DIST + 96));
        let dx = *x.d.get_unchecked(sb);
        let d = dx * f16_at(wb);
        let dmin = -dx * f16_at(wb.add(2));
        let sm = scales_mins_k4(wb.add(4));
        accm = mins_term(sm, x.q8s.as_ptr().add(sb * 8), dmin, accm);
        let sc128 = _mm256_castsi256_si128(sm);
        let scales = _mm256_set_m128i(sc128, sc128);
        let q4 = wb.add(16);
        let q8 = x.qs.as_ptr().add(sb * 256);
        let mut sumi = _mm256_setzero_si256();
        for j in 0..4 {
            let q4bits = load(q4.add(32 * j));
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            let q8l = load(q8.add(64 * j) as *const u8);
            let q8h = load(q8.add(64 * j + 32) as *const u8);
            let p16l = _mm256_madd_epi16(_mm256_shuffle_epi8(scales, scale_shuf(2 * j)), _mm256_maddubs_epi16(q4l, q8l));
            let p16h = _mm256_madd_epi16(_mm256_shuffle_epi8(scales, scale_shuf(2 * j + 1)), _mm256_maddubs_epi16(q4h, q8h));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)));
    }
    hsum_ps8(acc) + hsum_ps4(accm)
}

/// Q5_K x Q8_K: as Q4_K with the fifth bit from `qh` (bit `s` of every byte for sub-block `s`).
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q5_k_q8k(w: &[u8], x: &Q8KRow) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), nb * Q5_K_BLOCK_BYTES, "dot_q5_k_q8k: weight row bytes");
    let m4 = _mm256_set1_epi8(0xF);
    let mone = _mm256_set1_epi8(1);
    let mut acc = _mm256_setzero_ps();
    let mut accm = _mm_setzero_ps();
    for sb in 0..nb {
        let wb = w.as_ptr().add(sb * Q5_K_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        prefetch(wb.add(PF_DIST + 60));
        prefetch(wb.add(PF_DIST + 120));
        let dx = *x.d.get_unchecked(sb);
        let d = dx * f16_at(wb);
        let dmin = -dx * f16_at(wb.add(2));
        let sm = scales_mins_k4(wb.add(4));
        accm = mins_term(sm, x.q8s.as_ptr().add(sb * 8), dmin, accm);
        let sc128 = _mm256_castsi256_si128(sm);
        let scales = _mm256_set_m128i(sc128, sc128);
        let mut hb = load(wb.add(16));
        let q5 = wb.add(48);
        let q8 = x.qs.as_ptr().add(sb * 256);
        let mut sumi = _mm256_setzero_si256();
        for j in 0..4 {
            let q5bits = load(q5.add(32 * j));
            // the fifth bits of sub-blocks 2j and 2j+1: bit 0 of every byte of `hb`, moved to bit 4, `hb` shifted
            // right by one each time (a 16-bit shift spills a high byte's bit into the low byte's bit 7, never read)
            let h0 = _mm256_slli_epi16(_mm256_and_si256(hb, mone), 4);
            hb = _mm256_srli_epi16(hb, 1);
            let h1 = _mm256_slli_epi16(_mm256_and_si256(hb, mone), 4);
            hb = _mm256_srli_epi16(hb, 1);
            let q5_0 = _mm256_add_epi8(_mm256_and_si256(q5bits, m4), h0);
            let q5_1 = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4), h1);
            let q8_0 = load(q8.add(64 * j) as *const u8);
            let q8_1 = load(q8.add(64 * j + 32) as *const u8);
            let p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales, scale_shuf(2 * j)), _mm256_maddubs_epi16(q5_0, q8_0));
            let p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales, scale_shuf(2 * j + 1)), _mm256_maddubs_epi16(q5_1, q8_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)));
    }
    hsum_ps8(acc) + hsum_ps4(accm)
}

/// 16 i16 lanes: `sc[2m]` in lanes 0..8, `sc[2m+1]` in lanes 8..16 (sign-extended int8 scales).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn q6_scale_pair(scales: __m128i, m: usize) -> __m256i {
    let lo = 0x0101_0101_0101_0101u64.wrapping_mul((2 * m) as u64);
    let hi = 0x0101_0101_0101_0101u64.wrapping_mul((2 * m + 1) as u64);
    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, _mm_set_epi64x(hi as i64, lo as i64)))
}

/// Q6_K x Q8_K: unsigned 6-bit codes, int8 group scales, the -32 offset folded through `bsums`.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q6_k_q8k(w: &[u8], x: &Q8KRow) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), nb * Q6_K_BLOCK_BYTES, "dot_q6_k_q8k: weight row bytes");
    let m48 = _mm256_set1_epi8(0x30);
    let m15 = _mm256_set1_epi8(15);
    let mut acc = _mm256_setzero_ps();
    for sb in 0..nb {
        let wb = w.as_ptr().add(sb * Q6_K_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        prefetch(wb.add(PF_DIST + 56));
        prefetch(wb.add(PF_DIST + 112));
        prefetch(wb.add(PF_DIST + 168));
        let d = *x.d.get_unchecked(sb) * f16_at(wb.add(208));
        let scales = _mm_loadu_si128(wb.add(192) as *const __m128i);
        let q8sums = _mm256_loadu_si256(x.bsums.as_ptr().add(sb * 16) as *const __m256i);
        let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, _mm256_cvtepi8_epi16(scales)), 5);
        let q8 = x.qs.as_ptr().add(sb * 256);
        let mut sumi = _mm256_setzero_si256();
        for j in 0..2 {
            let q6 = q6_codes(wb, j, m15, m48);
            let q8p = q8.add(128 * j) as *const u8;
            let p16_0 = _mm256_madd_epi16(q6_scale_pair(scales, 4 * j), _mm256_maddubs_epi16(q6[0], load(q8p)));
            let p16_1 = _mm256_madd_epi16(q6_scale_pair(scales, 4 * j + 1), _mm256_maddubs_epi16(q6[1], load(q8p.add(32))));
            let p16_2 = _mm256_madd_epi16(q6_scale_pair(scales, 4 * j + 2), _mm256_maddubs_epi16(q6[2], load(q8p.add(64))));
            let p16_3 = _mm256_madd_epi16(q6_scale_pair(scales, 4 * j + 3), _mm256_maddubs_epi16(q6[3], load(q8p.add(96))));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }
        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)));
    }
    hsum_ps8(acc)
}

/// Dispatch (K-quant weight types only).
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q8k(t: GgmlType, w: &[u8], x: &Q8KRow) -> Option<f32> {
    Some(match t {
        GgmlType::Q4_K => dot_q4_k_q8k(w, x),
        GgmlType::Q5_K => dot_q5_k_q8k(w, x),
        GgmlType::Q6_K => dot_q6_k_q8k(w, x),
        _ => return None,
    })
}

/// Q8_K quantisation of one row (length a multiple of 256), bit-identical to `Q8KRow::quantize_scalar` on
/// finite input: vector abs-max, the sign taken from the first element whose |x| equals it, `iscale`, then
/// `cvtps_epi32` (round to nearest even, like ggml's `nearest_int`), `min(127)`, pack; `bsums` from the
/// packed bytes; `d = 1 / iscale`.
#[target_feature(enable = "avx2")]
pub unsafe fn quantize_row_q8k(x: &[f32], d: &mut Vec<f32>, qs: &mut Vec<i8>, bsums: &mut Vec<i16>) {
    assert!(x.len().is_multiple_of(256));
    let signbit = _mm256_set1_ps(-0.0);
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let c127 = _mm256_set1_epi32(127);
    let base = qs.len();
    qs.resize(base + x.len(), 0);
    for b in 0..x.len() / 256 {
        let p = x.as_ptr().add(b * 256);
        let mut m = _mm256_setzero_ps();
        let mut i = 0;
        while i < 256 {
            m = _mm256_max_ps(m, _mm256_andnot_ps(signbit, _mm256_loadu_ps(p.add(i))));
            i += 8;
        }
        let m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let amax = _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1)));
        if amax == 0.0 {
            d.push(0.0);
            bsums.extend(std::iter::repeat_n(0i16, 16));
            continue;
        }
        let amv = _mm256_set1_ps(amax);
        let mut max = 0f32;
        i = 0;
        while i < 256 {
            let eq = _mm256_cmp_ps(_mm256_andnot_ps(signbit, _mm256_loadu_ps(p.add(i))), amv, _CMP_EQ_OQ);
            let mask = _mm256_movemask_ps(eq);
            if mask != 0 {
                max = *p.add(i + mask.trailing_zeros() as usize);
                break;
            }
            i += 8;
        }
        let iscale = -127.0f32 / max;
        let isv = _mm256_set1_ps(iscale);
        let out = qs.as_mut_ptr().add(base + b * 256);
        i = 0;
        while i < 256 {
            let q = |k: usize| _mm256_min_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p.add(i + k)), isv)), c127);
            let i0 = _mm256_packs_epi32(q(0), q(8));
            let i2 = _mm256_packs_epi32(q(16), q(24));
            let bytes = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(i0, i2), perm);
            _mm256_storeu_si256(out.add(i) as *mut __m256i, bytes);
            i += 32;
        }
        for g in 0..16 {
            let mut s = 0i32;
            for k in 0..16 {
                s += *out.add(16 * g + k) as i32;
            }
            bsums.push(s as i16);
        }
        d.push(1.0 / iscale);
    }
}

// ------------------------------------------------------------------------- blocked Q8_K kernels (Phase 5.5)
//
// One weight super-block against `t <= T_MAX` activation rows (`kdot.rs`, blocked form): the codes are
// unpacked and the eight scale broadcasts (`shuffle_epi8` rows, Q4_K / Q5_K; scale pairs, Q6_K) built once,
// then every activation row is multiplied through them. Per row the operations are those of the single-row
// kernel above, in the same order (exact i32 lanes, one `mul` + `add` per super-block into the row's own
// 8-lane accumulator, the same final reduction), so `out[i]` is `dot_q*_k_q8k(w, xs[i])` bit for bit; what is
// shared between the rows is exactly the unpack. The per-row accumulators live in stack arrays (there are up
// to 16 of them, more than the register file), the unpacked codes and scales in another; both are L1 traffic.

use super::kdot::T_MAX;

/// Per-activation base pointers of a tile, so the inner loop indexes arrays rather than `Vec`s.
struct Tile {
    qs: [*const u8; T_MAX],
    d: [*const f32; T_MAX],
    q8s: [*const i16; T_MAX],
    bsums: [*const i16; T_MAX],
}

impl Tile {
    fn new(xs: &[&Q8KRow]) -> Tile {
        let mut t = Tile { qs: [std::ptr::null(); T_MAX], d: [std::ptr::null(); T_MAX], q8s: [std::ptr::null(); T_MAX], bsums: [std::ptr::null(); T_MAX] };
        for (i, x) in xs.iter().enumerate() {
            t.qs[i] = x.qs.as_ptr() as *const u8;
            t.d[i] = x.d.as_ptr();
            t.q8s[i] = x.q8s.as_ptr();
            t.bsums[i] = x.bsums.as_ptr();
        }
        t
    }
}

/// Shape checks of the blocked kernels: `1..=T_MAX` rows of the same length, `out` one per row.
fn tile_shape(w_len: usize, block_bytes: usize, xs: &[&Q8KRow], out: &[f32]) -> usize {
    let t = xs.len();
    assert!((1..=T_MAX).contains(&t), "dot_q8k_t: {t} activation rows (1..={T_MAX})");
    assert_eq!(out.len(), t, "dot_q8k_t: output length");
    let nb = xs[0].n_blocks();
    for x in xs {
        assert_eq!(x.n_blocks(), nb, "dot_q8k_t: activation rows of different lengths");
    }
    assert_eq!(w_len, nb * block_bytes, "dot_q8k_t: weight row bytes");
    nb
}

/// Prefetch the four lines of activation `a`'s next super-block (the tile lives in L2; the L1 streamer does
/// not keep up with 8 interleaved streams, and a maddubs waiting on an L2 load stalls the scheduler).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn prefetch_next_block(q8: *const u8) {
    let p = q8.add(256);
    prefetch(p);
    prefetch(p.add(64));
    prefetch(p.add(128));
    prefetch(p.add(192));
}

/// One super-block of a Q4_K / Q5_K row against activations `i0..i0 + A` of the tile: sub-blocks outer,
/// activations inner, so every sub-block issues `A` independent `maddubs -> madd -> add` triples into `A`
/// register accumulators (the single-chain form retired one uop per cycle; this is what gives the scheduler
/// ready work). Integer sums are exact in any order, so the lanes are the single-row kernel's; the f32 tail
/// per activation (the min term into `accm`, `d * sumi` into `acc`) is that kernel's, in its order.
/// (Index loops over `a`: `q8`, `sumi` and the tile arrays are parallel, indexed by the same activation.)
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn k45_group<const A: usize>(tile: &Tile, i0: usize, sb: usize, dw: f32, dminw: f32, mins: __m128i, codes: &[__m256i; 8], scb: &[__m256i; 8], acc: &mut [__m256; T_MAX], accm: &mut [__m128; T_MAX]) {
    let mut q8 = [std::ptr::null::<u8>(); A];
    for a in 0..A {
        q8[a] = tile.qs.get_unchecked(i0 + a).add(sb * 256);
        prefetch_next_block(q8[a]);
    }
    let mut sumi = [_mm256_setzero_si256(); A];
    for s in 0..8 {
        let c = codes[s];
        let sc = scb[s];
        for a in 0..A {
            sumi[a] = _mm256_add_epi32(sumi[a], _mm256_madd_epi16(sc, _mm256_maddubs_epi16(c, load(q8[a].add(32 * s)))));
        }
    }
    for a in 0..A {
        let i = i0 + a;
        let dx = *tile.d.get_unchecked(i).add(sb);
        let d = dx * dw;
        let dmin = -dx * dminw;
        let prod = _mm_madd_epi16(mins, _mm_loadu_si128(tile.q8s.get_unchecked(i).add(sb * 8) as *const __m128i));
        *accm.get_unchecked_mut(i) = _mm_add_ps(*accm.get_unchecked(i), _mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)));
        *acc.get_unchecked_mut(i) = _mm256_add_ps(*acc.get_unchecked(i), _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[a])));
    }
}

/// The tile's activations in groups of 8, 4, 2, 1 through `k45_group`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn k45_tile(t: usize, tile: &Tile, sb: usize, dw: f32, dminw: f32, mins: __m128i, codes: &[__m256i; 8], scb: &[__m256i; 8], acc: &mut [__m256; T_MAX], accm: &mut [__m128; T_MAX]) {
    let mut i = 0;
    while i + 8 <= t {
        k45_group::<8>(tile, i, sb, dw, dminw, mins, codes, scb, acc, accm);
        i += 8;
    }
    if i + 4 <= t {
        k45_group::<4>(tile, i, sb, dw, dminw, mins, codes, scb, acc, accm);
        i += 4;
    }
    if i + 2 <= t {
        k45_group::<2>(tile, i, sb, dw, dminw, mins, codes, scb, acc, accm);
        i += 2;
    }
    if i < t {
        k45_group::<1>(tile, i, sb, dw, dminw, mins, codes, scb, acc, accm);
    }
}

/// As `k45_group` for a Q6_K super-block: the eight scale pairs in `scp`, the `-32` fold through `bsums`
/// per activation (`sumi - (bsums . scales) << 5`, exact), then `d * sumi` into `acc`.
#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn q6_group<const A: usize>(tile: &Tile, i0: usize, sb: usize, dw: f32, scales16: __m256i, codes: &[__m256i; 8], scp: &[__m256i; 8], acc: &mut [__m256; T_MAX]) {
    let mut q8 = [std::ptr::null::<u8>(); A];
    for a in 0..A {
        q8[a] = tile.qs.get_unchecked(i0 + a).add(sb * 256);
        prefetch_next_block(q8[a]);
    }
    let mut sumi = [_mm256_setzero_si256(); A];
    for s in 0..8 {
        let c = codes[s];
        let sc = scp[s];
        for a in 0..A {
            sumi[a] = _mm256_add_epi32(sumi[a], _mm256_madd_epi16(sc, _mm256_maddubs_epi16(c, load(q8[a].add(32 * s)))));
        }
    }
    for a in 0..A {
        let i = i0 + a;
        let d = *tile.d.get_unchecked(i).add(sb) * dw;
        let q8sums = _mm256_loadu_si256(tile.bsums.get_unchecked(i).add(sb * 16) as *const __m256i);
        let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales16), 5);
        let s = _mm256_sub_epi32(sumi[a], q8sclsub);
        *acc.get_unchecked_mut(i) = _mm256_add_ps(*acc.get_unchecked(i), _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(s)));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn q6_tile(t: usize, tile: &Tile, sb: usize, dw: f32, scales16: __m256i, codes: &[__m256i; 8], scp: &[__m256i; 8], acc: &mut [__m256; T_MAX]) {
    let mut i = 0;
    while i + 8 <= t {
        q6_group::<8>(tile, i, sb, dw, scales16, codes, scp, acc);
        i += 8;
    }
    if i + 4 <= t {
        q6_group::<4>(tile, i, sb, dw, scales16, codes, scp, acc);
        i += 4;
    }
    if i + 2 <= t {
        q6_group::<2>(tile, i, sb, dw, scales16, codes, scp, acc);
        i += 2;
    }
    if i < t {
        q6_group::<1>(tile, i, sb, dw, scales16, codes, scp, acc);
    }
}

/// Q4_K x `t` Q8_K rows: `out[i] == dot_q4_k_q8k(w, xs[i])` bit for bit, the super-block unpacked once.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q4_k_q8k_t(w: &[u8], xs: &[&Q8KRow], out: &mut [f32]) {
    let t = xs.len();
    let nb = tile_shape(w.len(), Q4_K_BLOCK_BYTES, xs, out);
    let tile = Tile::new(xs);
    let m4 = _mm256_set1_epi8(0xF);
    let mut acc = [_mm256_setzero_ps(); T_MAX];
    let mut accm = [_mm_setzero_ps(); T_MAX];
    let mut codes = [_mm256_setzero_si256(); 8];
    let mut scb = [_mm256_setzero_si256(); 8];
    for sb in 0..nb {
        let wb = w.as_ptr().add(sb * Q4_K_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        prefetch(wb.add(PF_DIST + 48));
        prefetch(wb.add(PF_DIST + 96));
        let dw = f16_at(wb);
        let dminw = f16_at(wb.add(2));
        let sm = scales_mins_k4(wb.add(4));
        let sc128 = _mm256_castsi256_si128(sm);
        let scales = _mm256_set_m128i(sc128, sc128);
        let mins = _mm256_extracti128_si256(sm, 1);
        let q4 = wb.add(16);
        for j in 0..4 {
            let q4bits = load(q4.add(32 * j));
            codes[2 * j] = _mm256_and_si256(q4bits, m4);
            codes[2 * j + 1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (s, v) in scb.iter_mut().enumerate() {
            *v = _mm256_shuffle_epi8(scales, scale_shuf(s));
        }
        k45_tile(t, &tile, sb, dw, dminw, mins, &codes, &scb, &mut acc, &mut accm);
    }
    for i in 0..t {
        out[i] = hsum_ps8(acc[i]) + hsum_ps4(accm[i]);
    }
}

/// Q5_K x `t` Q8_K rows: `out[i] == dot_q5_k_q8k(w, xs[i])` bit for bit.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q5_k_q8k_t(w: &[u8], xs: &[&Q8KRow], out: &mut [f32]) {
    let t = xs.len();
    let nb = tile_shape(w.len(), Q5_K_BLOCK_BYTES, xs, out);
    let tile = Tile::new(xs);
    let m4 = _mm256_set1_epi8(0xF);
    let mone = _mm256_set1_epi8(1);
    let mut acc = [_mm256_setzero_ps(); T_MAX];
    let mut accm = [_mm_setzero_ps(); T_MAX];
    let mut codes = [_mm256_setzero_si256(); 8];
    let mut scb = [_mm256_setzero_si256(); 8];
    for sb in 0..nb {
        let wb = w.as_ptr().add(sb * Q5_K_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        prefetch(wb.add(PF_DIST + 60));
        prefetch(wb.add(PF_DIST + 120));
        let dw = f16_at(wb);
        let dminw = f16_at(wb.add(2));
        let sm = scales_mins_k4(wb.add(4));
        let sc128 = _mm256_castsi256_si128(sm);
        let scales = _mm256_set_m128i(sc128, sc128);
        let mins = _mm256_extracti128_si256(sm, 1);
        let mut hb = load(wb.add(16));
        let q5 = wb.add(48);
        for j in 0..4 {
            let q5bits = load(q5.add(32 * j));
            let h0 = _mm256_slli_epi16(_mm256_and_si256(hb, mone), 4);
            hb = _mm256_srli_epi16(hb, 1);
            let h1 = _mm256_slli_epi16(_mm256_and_si256(hb, mone), 4);
            hb = _mm256_srli_epi16(hb, 1);
            codes[2 * j] = _mm256_add_epi8(_mm256_and_si256(q5bits, m4), h0);
            codes[2 * j + 1] = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4), h1);
        }
        for (s, v) in scb.iter_mut().enumerate() {
            *v = _mm256_shuffle_epi8(scales, scale_shuf(s));
        }
        k45_tile(t, &tile, sb, dw, dminw, mins, &codes, &scb, &mut acc, &mut accm);
    }
    for i in 0..t {
        out[i] = hsum_ps8(acc[i]) + hsum_ps4(accm[i]);
    }
}

/// Q6_K x `t` Q8_K rows: `out[i] == dot_q6_k_q8k(w, xs[i])` bit for bit; the codes and the eight scale pairs
/// unpacked once per super-block, the `-32` fold through `bsums` per row.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q6_k_q8k_t(w: &[u8], xs: &[&Q8KRow], out: &mut [f32]) {
    let t = xs.len();
    let nb = tile_shape(w.len(), Q6_K_BLOCK_BYTES, xs, out);
    let tile = Tile::new(xs);
    let m48 = _mm256_set1_epi8(0x30);
    let m15 = _mm256_set1_epi8(15);
    let mut acc = [_mm256_setzero_ps(); T_MAX];
    let mut codes = [_mm256_setzero_si256(); 8];
    let mut scp = [_mm256_setzero_si256(); 8];
    for sb in 0..nb {
        let wb = w.as_ptr().add(sb * Q6_K_BLOCK_BYTES);
        prefetch(wb.add(PF_DIST));
        prefetch(wb.add(PF_DIST + 56));
        prefetch(wb.add(PF_DIST + 112));
        prefetch(wb.add(PF_DIST + 168));
        let dw = f16_at(wb.add(208));
        let scales = _mm_loadu_si128(wb.add(192) as *const __m128i);
        let scales16 = _mm256_cvtepi8_epi16(scales);
        for j in 0..2 {
            let q6 = q6_codes(wb, j, m15, m48);
            codes[4 * j..4 * j + 4].copy_from_slice(&q6);
        }
        for (s, v) in scp.iter_mut().enumerate() {
            *v = q6_scale_pair(scales, s);
        }
        q6_tile(t, &tile, sb, dw, scales16, &codes, &scp, &mut acc);
    }
    for i in 0..t {
        out[i] = hsum_ps8(acc[i]);
    }
}

/// Blocked dispatch (K-quant weight types only): `true` if handled.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q8k_t(t: GgmlType, w: &[u8], xs: &[&Q8KRow], out: &mut [f32]) -> bool {
    match t {
        GgmlType::Q4_K => dot_q4_k_q8k_t(w, xs, out),
        GgmlType::Q5_K => dot_q5_k_q8k_t(w, xs, out),
        GgmlType::Q6_K => dot_q6_k_q8k_t(w, xs, out),
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::get_scale_min_k4;

    #[test]
    fn scales_mins_k4_matches_scalar_unpack() {
        if !super::super::simd::avx2_detected() {
            return;
        }
        let mut seed = 0x1234_5678u32;
        for _ in 0..2000 {
            let mut bytes = [0u8; 16];
            for b in bytes.iter_mut() {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *b = (seed >> 24) as u8;
            }
            let v = unsafe { scales_mins_k4(bytes.as_ptr()) };
            let mut lanes = [0i16; 16];
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v) };
            for s in 0..8 {
                let (sc, m) = get_scale_min_k4(s, &bytes[..12]);
                assert_eq!((lanes[s], lanes[8 + s]), (sc as i16, m as i16), "bytes {bytes:?} sub-block {s}: lanes {lanes:?}");
            }
        }
    }
}
