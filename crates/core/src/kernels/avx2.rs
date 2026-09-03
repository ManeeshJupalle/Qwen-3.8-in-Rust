//! AVX2 versions of the hot kernels (x86_64 only), selected at runtime by `super::simd::use_avx2()`.
//! The scalar kernels stay the reference; `tests/avx2.rs` compares the two on every kernel fixture and on
//! random rows.
//!
//! Contract: every function here is BIT-IDENTICAL to its scalar counterpart on finite inputs.
//! - Integer dot kernels (`dot_q8_0_q8`, `dot_q4_0_q8`, `dot_q4_k_q8`, `dot_q5_k_q8`, `dot_q6_k_q8`): the
//!   per-block integer sums are exact in both paths (i8 x i8 products widened through `maddubs`/`madd` with
//!   no saturation, see the bounds at each kernel); the f32 expression per block is then evaluated with the
//!   same scalar operations, in the same order, without FMA, and block partials are accumulated left to right
//!   exactly as `dot.rs` documents.
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
use crate::gguf::GgmlType;
use crate::quant::{get_scale_min_k4, Q4_0_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES};

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

/// Exact `sum(w[j] * x[j])` for unsigned `w` (<= 63): pairs at most 2 * 63 * 127 = 16002.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dot32_u8(w: __m256i, x: __m256i) -> i32 {
    hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1)))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sum32_i8(x: __m256i) -> i32 {
    dot32_u8(_mm256_set1_epi8(1), x)
}

/// Q8_0 weights: per block `(d_w * d_x) * sum(q_w * q_x)`.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q8_0_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), nb * Q8_0_BLOCK_BYTES, "dot_q8_0_q8: weight row bytes");
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = w.as_ptr().add(b * Q8_0_BLOCK_BYTES);
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

/// Q4_K weights: per sub-block `d_x * ((d * sc) * sum(q_w * q_x) - (dmin * m) * sum(q_x))`, nibbles unsigned.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q4_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q4_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * Q4_K_BLOCK_BYTES, "dot_q4_k_q8: weight row bytes");
    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = 0f32;
    for sb in 0..nb / 8 {
        let wb = w.as_ptr().add(sb * Q4_K_BLOCK_BYTES);
        let d = f16_at(wb);
        let dmin = f16_at(wb.add(2));
        let scales = &w[sb * Q4_K_BLOCK_BYTES + 4..sb * Q4_K_BLOCK_BYTES + 16];
        let qs = wb.add(16);
        for p in 0..4 {
            let q = load(qs.add(p * 32));
            let lo = _mm256_and_si256(q, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            for (half, wv) in [(0usize, lo), (1, hi)] {
                let s = 2 * p + half;
                let (sc, m) = get_scale_min_k4(s, scales);
                let xb = sb * 8 + s;
                let xv = load_x(x, xb);
                let isum = dot32_u8(wv, xv);
                let xsum = sum32_i8(xv);
                acc += f16_bits(x.d[xb]) * ((d * sc as f32) * isum as f32 - (dmin * m as f32) * xsum as f32);
            }
        }
    }
    acc
}

/// Q5_K weights: as Q4_K plus the fifth bit from `qh` (bit `2p + half` of `qh[j]` for sub-block `2p + half`).
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q5_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q5_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * Q5_K_BLOCK_BYTES, "dot_q5_k_q8: weight row bytes");
    let m4 = _mm256_set1_epi8(0x0F);
    let one = _mm256_set1_epi8(1);
    let mut acc = 0f32;
    for sb in 0..nb / 8 {
        let wb = w.as_ptr().add(sb * Q5_K_BLOCK_BYTES);
        let d = f16_at(wb);
        let dmin = f16_at(wb.add(2));
        let scales = &w[sb * Q5_K_BLOCK_BYTES + 4..sb * Q5_K_BLOCK_BYTES + 16];
        let qh = load(wb.add(16));
        let ql = wb.add(48);
        for p in 0..4 {
            let q = load(ql.add(p * 32));
            let lo = _mm256_and_si256(q, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            for (half, base) in [(0usize, lo), (1, hi)] {
                let s = 2 * p + half;
                // bit (2p + half) of every qh byte, moved to bit 4 (shifts stay inside the byte after the mask)
                let bit = _mm256_and_si256(shift_right_epi16(qh, (2 * p + half) as i32), one);
                let wv = _mm256_or_si256(base, _mm256_slli_epi16(bit, 4));
                let (sc, m) = get_scale_min_k4(s, scales);
                let xb = sb * 8 + s;
                let xv = load_x(x, xb);
                let isum = dot32_u8(wv, xv);
                let xsum = sum32_i8(xv);
                acc += f16_bits(x.d[xb]) * ((d * sc as f32) * isum as f32 - (dmin * m as f32) * xsum as f32);
            }
        }
    }
    acc
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

/// Q6_K weights: per activation block (two 16-groups g0, g1 with int8 scales)
/// `d_x * ((d * sc_g0) * sum16(q_w * q_x) + (d * sc_g1) * sum16(q_w * q_x))`, `q_w = 6-bit - 32`.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn dot_q6_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q6_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * Q6_K_BLOCK_BYTES, "dot_q6_k_q8: weight row bytes");
    let m4 = _mm256_set1_epi8(0x0F);
    let m2 = _mm256_set1_epi8(3);
    let off = _mm256_set1_epi8(32);
    let mut acc = 0f32;
    for sb in 0..nb / 8 {
        let wb = w.as_ptr().add(sb * Q6_K_BLOCK_BYTES);
        let d = f16_at(wb.add(208));
        for half in 0..2 {
            let ql = wb.add(half * 64);
            let qhv = load(wb.add(128 + half * 32));
            let sc = wb.add(192 + half * 8);
            let ql0 = load(ql);
            let ql1 = load(ql.add(32));
            for k in 0..4 {
                let lo = match k {
                    0 => _mm256_and_si256(ql0, m4),
                    1 => _mm256_and_si256(ql1, m4),
                    2 => _mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4),
                    _ => _mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4),
                };
                let hi = _mm256_and_si256(shift_right_epi16(qhv, (2 * k) as i32), m2);
                let qv = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hi, 4)), off);
                let xb = sb * 8 + half * 4 + k;
                let p32 = dot32_i8(qv, load_x(x, xb));
                // lanes 0..4 hold elements 0..16 (scale sc[2k]), lanes 4..8 elements 16..32 (scale sc[2k+1])
                let isum0 = hsum128_i32(_mm256_castsi256_si128(p32));
                let isum1 = hsum128_i32(_mm256_extracti128_si256(p32, 1));
                let s0 = (*sc.add(2 * k) as i8) as f32;
                let s1 = (*sc.add(2 * k + 1) as i8) as f32;
                acc += f16_bits(x.d[xb]) * ((d * s0) * isum0 as f32 + (d * s1) * isum1 as f32);
            }
        }
    }
    acc
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
