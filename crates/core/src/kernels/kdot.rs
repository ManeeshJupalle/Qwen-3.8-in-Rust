//! K-quant weight rows times a Q8_K activation row, the scalar reference for `avx2.rs` (Phase 3).
//! Integer math and summation order follow ggml's AVX2 `ggml_vec_dot_q{4,5,6}_K_q8_K` exactly, so the AVX2
//! kernels are bit-identical to these (it costs nothing: the lane structure is what the vector code does
//! anyway). Full derivation in `docs/kquant-dot.md`; the shape of every kernel is:
//!
//! - per super-block of 256 weights, exact i32 partial sums in 8 lanes, lane `i` holding the elements whose
//!   position inside each 32-element sub-block is `4i..4i+4` (the `maddubs` pair sums then the `madd` pair
//!   sums of the vector code), with the 6-bit (Q4_K/Q5_K) or int8 (Q6_K) sub-block scale multiplied in as
//!   an integer;
//! - the sub-block mins (Q4_K/Q5_K) are folded through the activation group sums `bsums` in 4 exact i32
//!   lanes, lane `i` holding sub-blocks `2i` and `2i+1`; the Q6_K `-32` offset is folded through `bsums`
//!   into the 8 main lanes (lane `i` holding groups `2i`, `2i+1`);
//! - once per super-block each lane is converted to f32 and multiplied by the super-block scale
//!   (`d_x * d_w`, or `-d_x * dmin_w` for the mins) and added to an 8-lane (4-lane) f32 accumulator;
//! - at the end of the row the lanes are reduced as `((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7))` and
//!   `((m0+m2)+(m1+m3))`, and the two totals added.
//!
//! No FMA: multiply then add, in both paths.

use super::q8k::Q8KRow;
use super::rmsnorm::reduce8;
use crate::gguf::GgmlType;
use crate::quant::{f16_to_f32, get_scale_min_k4, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

#[inline]
fn reduce4(m: &[f32; 4]) -> f32 {
    (m[0] + m[2]) + (m[1] + m[3])
}

/// Shared Q4_K / Q5_K body: `qw(block, s, j)` yields the unsigned code of element `j` of sub-block `s`.
#[inline(always)]
fn dot_q45_k(w: &[u8], x: &Q8KRow, block_bytes: usize, qw: impl Fn(&[u8], usize, usize) -> i32) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), nb * block_bytes, "dot_q4/5_k_q8k: weight row bytes");
    let mut acc = [0f32; 8];
    let mut accm = [0f32; 4];
    for sb in 0..nb {
        let wb = &w[sb * block_bytes..(sb + 1) * block_bytes];
        let dx = x.d[sb];
        let d = dx * f16_at(wb, 0);
        let dmin = -dx * f16_at(wb, 2);
        let scales = &wb[4..16];
        let xq = x.block(sb);
        let q8s = x.q8s(sb);
        let mut sc = [0i32; 8];
        let mut mn = [0i32; 8];
        for s in 0..8 {
            let (a, b) = get_scale_min_k4(s, scales);
            sc[s] = a as i32;
            mn[s] = b as i32;
        }
        // mins through the sub-block sums: 4 lanes, lane i = sub-blocks 2i and 2i+1
        for i in 0..4 {
            let prod = mn[2 * i] * q8s[2 * i] as i32 + mn[2 * i + 1] * q8s[2 * i + 1] as i32;
            accm[i] += dmin * prod as f32;
        }
        // main term: 8 lanes by position within the sub-block
        let mut sumi = [0i32; 8];
        for s in 0..8 {
            let xs = &xq[s * 32..(s + 1) * 32];
            for j in 0..32 {
                sumi[j / 4] += sc[s] * qw(wb, s, j) * xs[j] as i32;
            }
        }
        for i in 0..8 {
            acc[i] += d * sumi[i] as f32;
        }
    }
    reduce8(&acc) + reduce4(&accm)
}

/// Q4_K weights (`docs/quant-layouts.md`): sub-block `s` lives in the low (`s` even) or high (`s` odd) nibbles
/// of bytes `16 + 32 * (s / 2) ..`.
pub fn dot_q4_k_q8k(w: &[u8], x: &Q8KRow) -> f32 {
    dot_q45_k(w, x, Q4_K_BLOCK_BYTES, |wb, s, j| {
        let q = wb[16 + (s / 2) * 32 + j];
        (if s % 2 == 0 { q & 0xF } else { q >> 4 }) as i32
    })
}

/// Q5_K weights: as Q4_K plus bit `s` of `qh[j]` as the fifth bit.
pub fn dot_q5_k_q8k(w: &[u8], x: &Q8KRow) -> f32 {
    dot_q45_k(w, x, Q5_K_BLOCK_BYTES, |wb, s, j| {
        let q = wb[48 + (s / 2) * 32 + j];
        let lo = (if s % 2 == 0 { q & 0xF } else { q >> 4 }) as i32;
        lo + if wb[16 + j] & (1 << s) != 0 { 16 } else { 0 }
    })
}

/// Q6_K weights: unsigned 6-bit codes times the activation codes with the int8 group scale, minus
/// `32 * sc * bsums` per 16-group folded into the lanes (`sumi[i] -= 32 * (sc[2i] * bs[2i] + sc[2i+1] * bs[2i+1])`).
pub fn dot_q6_k_q8k(w: &[u8], x: &Q8KRow) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), nb * Q6_K_BLOCK_BYTES, "dot_q6_k_q8k: weight row bytes");
    let mut acc = [0f32; 8];
    for sb in 0..nb {
        let wb = &w[sb * Q6_K_BLOCK_BYTES..(sb + 1) * Q6_K_BLOCK_BYTES];
        let ql_all = &wb[0..128];
        let qh_all = &wb[128..192];
        let sc_all = &wb[192..208];
        let d = x.d[sb] * f16_at(wb, 208);
        let xq = x.block(sb);
        let bs = x.bsums(sb);
        let mut sumi = [0i32; 8];
        for half in 0..2 {
            let ql = &ql_all[half * 64..(half + 1) * 64];
            let qh = &qh_all[half * 32..(half + 1) * 32];
            let sc = &sc_all[half * 8..(half + 1) * 8];
            for k in 0..4 {
                let xs = &xq[half * 128 + k * 32..half * 128 + (k + 1) * 32];
                for l in 0..32 {
                    let lo = match k {
                        0 => ql[l] & 0xF,
                        1 => ql[l + 32] & 0xF,
                        2 => ql[l] >> 4,
                        _ => ql[l + 32] >> 4,
                    };
                    let hi = (qh[l] >> (2 * k)) & 3;
                    let qw = (lo | (hi << 4)) as i32;
                    let scale = (sc[2 * k + l / 16] as i8) as i32;
                    sumi[l / 4] += scale * qw * xs[l] as i32;
                }
            }
        }
        for i in 0..8 {
            let s0 = (sc_all[2 * i] as i8) as i32;
            let s1 = (sc_all[2 * i + 1] as i8) as i32;
            sumi[i] -= 32 * (bs[2 * i] as i32 * s0 + bs[2 * i + 1] as i32 * s1);
        }
        for i in 0..8 {
            acc[i] += d * sumi[i] as f32;
        }
    }
    reduce8(&acc)
}

/// Dispatch by weight type and instruction path (K-quant types only).
pub fn dot_q8k(t: GgmlType, w: &[u8], x: &Q8KRow) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::simd::use_avx2() {
        // SAFETY: `use_avx2` is only true when the CPU reports AVX2 and F16C.
        if let Some(v) = unsafe { super::avx2::dot_q8k(t, w, x) } {
            return v;
        }
    }
    dot_q8k_scalar(t, w, x)
}

/// The scalar reference dispatch.
pub fn dot_q8k_scalar(t: GgmlType, w: &[u8], x: &Q8KRow) -> f32 {
    match t {
        GgmlType::Q4_K => dot_q4_k_q8k(w, x),
        GgmlType::Q5_K => dot_q5_k_q8k(w, x),
        GgmlType::Q6_K => dot_q6_k_q8k(w, x),
        other => panic!("dot_q8k_scalar: {other:?} weights take a Q8_0 activation row, not Q8_K"),
    }
}
