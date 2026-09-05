//! Restructured single-row Q4_K x Q8_K kernels (same arithmetic per super-block), timed on the L1 row:
//! B: the next super-block's header (f16 converts, scale unpack) computed one iteration ahead;
//! C: two independent integer accumulators (even / odd sub-block pairs) so the add chain is half as deep;
//! E: ggml's scalar `utmp` scale unpack (scalar ALU ports) instead of the vector shuffle unpack;
//! F: B + C + E together.
use std::arch::x86_64::*;

use aqueduct_core::kernels::q8k::Q8KRow;

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
unsafe fn scales_mins_vec(p: *const u8) -> __m256i {
    let v = _mm_loadu_si128(p as *const __m128i);
    let lo = _mm_and_si128(v, _mm_set1_epi8(0x3F));
    let top = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(v, 6), _mm_set1_epi8(0x03)), 4);
    let c = _mm_shuffle_epi8(v, _mm_setr_epi8(8, 9, 10, 11, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1));
    let cl = _mm_and_si128(c, _mm_set1_epi8(0x0F));
    let ch = _mm_and_si128(_mm_srli_epi16(c, 4), _mm_set1_epi8(0x0F));
    let hi = _mm_or_si128(_mm_blend_epi16(cl, ch, 0b1100), top);
    _mm256_cvtepu8_epi16(_mm_unpacklo_epi32(lo, hi))
}

/// ggml's scalar unpack: 12 bytes -> utmp[4] u32 -> 16 i16 lanes `sc[0..8], m[0..8]`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn scales_mins_scalar(p: *const u8) -> __m256i {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let mut utmp = [0u32; 4];
    std::ptr::copy_nonoverlapping(p, utmp.as_mut_ptr() as *mut u8, 12);
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    // utmp[0..2] = sc[0..8] as bytes, utmp[2..4] = m[0..8]
    _mm256_cvtepu8_epi16(_mm_loadu_si128(utmp.as_ptr() as *const __m128i))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn reduce8(acc: __m256) -> f32 {
    let s4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum4(m: __m128) -> f32 {
    let s2 = _mm_add_ps(m, _mm_movehl_ps(m, m));
    _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)))
}

struct Hdr {
    d: f32,
    dmin: f32,
    scales: __m256i,
    mins: __m128i,
}

#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn header<const SCALAR: bool>(wb: *const u8, dx: f32) -> Hdr {
    let d = dx * f16_at(wb);
    let dmin = -dx * f16_at(wb.add(2));
    let sm = if SCALAR { scales_mins_scalar(wb.add(4)) } else { scales_mins_vec(wb.add(4)) };
    let sc128 = _mm256_castsi256_si128(sm);
    Hdr { d, dmin, scales: _mm256_set_m128i(sc128, sc128), mins: _mm256_extracti128_si256(sm, 1) }
}

/// PIPE: header of sb+1 computed during sb; SPLIT: two integer accumulators; SCALAR: scalar scale unpack.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn q4k<const PIPE: bool, const SPLIT: bool, const SCALAR: bool>(w: &[u8], x: &Q8KRow) -> f32 {
    let nb = x.n_blocks();
    let m4 = _mm256_set1_epi8(0xF);
    let mut acc = _mm256_setzero_ps();
    let mut accm = _mm_setzero_ps();
    let mut next = header::<SCALAR>(w.as_ptr(), *x.d.get_unchecked(0));
    for sb in 0..nb {
        let wb = w.as_ptr().add(sb * 144);
        let h = if PIPE {
            let cur = Hdr { d: next.d, dmin: next.dmin, scales: next.scales, mins: next.mins };
            if sb + 1 < nb {
                next = header::<SCALAR>(wb.add(144), *x.d.get_unchecked(sb + 1));
            }
            cur
        } else {
            header::<SCALAR>(wb, *x.d.get_unchecked(sb))
        };
        let prod = _mm_madd_epi16(h.mins, _mm_loadu_si128(x.q8s.as_ptr().add(sb * 8) as *const __m128i));
        accm = _mm_add_ps(accm, _mm_mul_ps(_mm_set1_ps(h.dmin), _mm_cvtepi32_ps(prod)));
        let q4 = wb.add(16);
        let q8 = x.qs.as_ptr().add(sb * 256) as *const u8;
        let mut sumi0 = _mm256_setzero_si256();
        let mut sumi1 = _mm256_setzero_si256();
        for j in 0..4 {
            let q4bits = load(q4.add(32 * j));
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            let p16l = _mm256_madd_epi16(_mm256_shuffle_epi8(h.scales, scale_shuf(2 * j)), _mm256_maddubs_epi16(q4l, load(q8.add(64 * j))));
            let p16h = _mm256_madd_epi16(_mm256_shuffle_epi8(h.scales, scale_shuf(2 * j + 1)), _mm256_maddubs_epi16(q4h, load(q8.add(64 * j + 32))));
            if SPLIT {
                sumi0 = _mm256_add_epi32(sumi0, p16l);
                sumi1 = _mm256_add_epi32(sumi1, p16h);
            } else {
                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(p16l, p16h));
            }
        }
        let sumi = if SPLIT { _mm256_add_epi32(sumi0, sumi1) } else { sumi0 };
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(h.d), _mm256_cvtepi32_ps(sumi)));
    }
    reduce8(acc) + hsum4(accm)
}
