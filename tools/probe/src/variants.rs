//! The single-row Q4_K x Q8_K AVX2 kernel with its pieces switchable, timed on an L1-resident row, to find where
//! the 54 cycles per super-block go. Semantics change when a piece is removed (timing only).
use std::arch::x86_64::*;

use aqueduct_core::kernels::q8k::Q8KRow;

const PF_DIST: usize = 1024;

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

#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn f16_at(b: *const u8) -> f32 {
    let bits = u16::from_le_bytes([*b, *b.add(1)]);
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load(p: *const u8) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn scales_mins_k4(p: *const u8) -> __m256i {
    let v = _mm_loadu_si128(p as *const __m128i);
    let lo = _mm_and_si128(v, _mm_set1_epi8(0x3F));
    let top = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(v, 6), _mm_set1_epi8(0x03)), 4);
    let c = _mm_shuffle_epi8(v, _mm_setr_epi8(8, 9, 10, 11, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1));
    let cl = _mm_and_si128(c, _mm_set1_epi8(0x0F));
    let ch = _mm_and_si128(_mm_srli_epi16(c, 4), _mm_set1_epi8(0x0F));
    let hi = _mm_or_si128(_mm_blend_epi16(cl, ch, 0b1100), top);
    _mm256_cvtepu8_epi16(_mm_unpacklo_epi32(lo, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce8(acc: __m256) -> f32 {
    let s4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)))
}

/// PF: weight prefetches; HDR: real header (f16 converts + scale unpack) else constants; MINS: the min term;
/// SHUF: per-sub-block scale broadcast shuffles else one constant scale vector; UNPACK: the nibble and/shift
/// else the raw bytes twice; TAIL: the per-super-block f32 cvt/mul/add else an i32 accumulator.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn q4k<const PF: bool, const HDR: bool, const MINS: bool, const SHUF: bool, const UNPACK: bool, const TAIL: bool>(w: &[u8], x: &Q8KRow) -> f32 {
    let nb = x.n_blocks();
    let m4 = _mm256_set1_epi8(0xF);
    let mut acc = _mm256_setzero_ps();
    let mut accm = _mm_setzero_ps();
    let mut acci = _mm256_setzero_si256();
    let const_scales = _mm256_set1_epi16(3);
    for sb in 0..nb {
        let wb = w.as_ptr().add(sb * 144);
        if PF {
            _mm_prefetch(wb.add(PF_DIST) as *const i8, _MM_HINT_T0);
            _mm_prefetch(wb.add(PF_DIST + 48) as *const i8, _MM_HINT_T0);
            _mm_prefetch(wb.add(PF_DIST + 96) as *const i8, _MM_HINT_T0);
        }
        let dx = *x.d.get_unchecked(sb);
        let (d, dmin, scales, sm) = if HDR {
            let d = dx * f16_at(wb);
            let dmin = -dx * f16_at(wb.add(2));
            let sm = scales_mins_k4(wb.add(4));
            let sc128 = _mm256_castsi256_si128(sm);
            (d, dmin, _mm256_set_m128i(sc128, sc128), sm)
        } else {
            (dx * 0.37, -dx * 0.11, const_scales, const_scales)
        };
        if MINS {
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(sm, 1), _mm_loadu_si128(x.q8s.as_ptr().add(sb * 8) as *const __m128i));
            accm = _mm_add_ps(accm, _mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)));
        }
        let q4 = wb.add(16);
        let q8 = x.qs.as_ptr().add(sb * 256) as *const u8;
        let mut sumi = _mm256_setzero_si256();
        for j in 0..4 {
            let q4bits = load(q4.add(32 * j));
            let (q4l, q4h) = if UNPACK { (_mm256_and_si256(q4bits, m4), _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4)) } else { (q4bits, q4bits) };
            let q8l = load(q8.add(64 * j));
            let q8h = load(q8.add(64 * j + 32));
            let (s0, s1) = if SHUF { (_mm256_shuffle_epi8(scales, scale_shuf(2 * j)), _mm256_shuffle_epi8(scales, scale_shuf(2 * j + 1))) } else { (scales, scales) };
            let p16l = _mm256_madd_epi16(s0, _mm256_maddubs_epi16(q4l, q8l));
            let p16h = _mm256_madd_epi16(s1, _mm256_maddubs_epi16(q4h, q8h));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }
        if TAIL {
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)));
        } else {
            acci = _mm256_add_epi32(acci, sumi);
        }
    }
    if TAIL {
        reduce8(acc) + reduce8(_mm256_castps128_ps256(accm))
    } else {
        reduce8(_mm256_cvtepi32_ps(acci))
    }
}
