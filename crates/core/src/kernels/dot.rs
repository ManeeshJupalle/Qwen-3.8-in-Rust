//! Row dot products: one row of weights (ggml block layout) times one Q8_0 activation row -> f32.
//!
//! Q8_0, Q4_0 and F32 weights (Phase 2 order): the row is walked in 32-element activation blocks; inside a
//! block the integer products `q_w * q_x` are accumulated exactly in i32, the block partial is converted to
//! f32 and scaled, block partials are accumulated left to right in f32 (for f32 weights the block partial is
//! a left-to-right f32 sum of 32 products). Expected error ~ (n/32) * eps * rms(block partial), inside the
//! budget k*sqrt(n)*eps*max|term| (k = 8).
//!
//! K-quant weights (Phase 3 order, `docs/kquant-dot.md`, the scalar reference of the AVX2 kernels and
//! bit-identical to them): per 32-element sub-block the exact i32 lane sums `p32[i] = sum_{j/4 == i} q_w q_x`
//! are scaled by one f32 factor `d_x * (d * sc)` into 8 f32 lanes (lane `i` = positions `4i..4i+4` of every
//! sub-block); the Q4_K/Q5_K mins go through `((m as f32) * -dmin) * (d_x * sum q_x)` into 8 more lanes (lane
//! = sub-block index within the super-block); the Q6_K `-32` offset is subtracted in integer from lanes 0 and
//! 4 (`32 * sum q_x` of each 16-group). Lanes are reduced once per row as `((l0+l4)+(l2+l6)) +
//! ((l1+l5)+(l3+l7))`. Lanes cancel against each other, so the error is bounded by
//! `k * eps * sum_j (|main_j| + |min_j|) |x_j|` (the term magnitudes), not by the result.

use super::q8::Q8Row;
use super::rmsnorm::reduce8;
use crate::gguf::GgmlType;
use crate::quant::{f16_to_f32, get_scale_min_k4, Q4_0_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES};

#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// Q8_0 weights: per block `(d_w * d_x) * sum(q_w * q_x)`.
pub fn dot_q8_0_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), nb * Q8_0_BLOCK_BYTES, "dot_q8_0_q8: weight row bytes");
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = &w[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
        let dw = f16_at(wb, 0);
        let xq = x.block(b);
        let mut isum: i32 = 0;
        for j in 0..32 {
            isum += (wb[2 + j] as i8) as i32 * xq[j] as i32;
        }
        acc += (dw * x.scale(b)) * isum as f32;
    }
    acc
}

/// Q4_0 weights: per block `(d_w * d_x) * sum((q_w - 8) * q_x)`; low nibbles are values 0..16, high 16..32.
pub fn dot_q4_0_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), nb * Q4_0_BLOCK_BYTES, "dot_q4_0_q8: weight row bytes");
    let mut acc = 0f32;
    for b in 0..nb {
        let wb = &w[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES];
        let dw = f16_at(wb, 0);
        let xq = x.block(b);
        let mut isum: i32 = 0;
        for j in 0..16 {
            let lo = (wb[2 + j] & 0x0F) as i32 - 8;
            let hi = (wb[2 + j] >> 4) as i32 - 8;
            isum += lo * xq[j] as i32 + hi * xq[j + 16] as i32;
        }
        acc += (dw * x.scale(b)) * isum as f32;
    }
    acc
}

/// Shared Q4_K / Q5_K body: `qw(block, s, j)` yields the unsigned code of element `j` of sub-block `s`.
#[inline(always)]
#[allow(clippy::needless_range_loop)] // `s` indexes the min lane, the scale pair and the activation block
fn dot_q45_k_q8(w: &[u8], x: &Q8Row, block_bytes: usize, qw: impl Fn(&[u8], usize, usize) -> i32) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q4/5_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * block_bytes, "dot_q4/5_k_q8: weight row bytes");
    let mut acc = [0f32; 8];
    let mut accm = [0f32; 8];
    for sb in 0..nb / 8 {
        let wb = &w[sb * block_bytes..(sb + 1) * block_bytes];
        let d = f16_at(wb, 0);
        let dmin = f16_at(wb, 2);
        let scales = &wb[4..16];
        for s in 0..8 {
            let (sc, m) = get_scale_min_k4(s, scales);
            let xb = sb * 8 + s;
            let xq = x.block(xb);
            let mut p32 = [0i32; 8];
            for j in 0..32 {
                p32[j / 4] += qw(wb, s, j) * xq[j] as i32;
            }
            let f = x.d32[xb] * (d * sc as f32);
            for i in 0..8 {
                acc[i] += p32[i] as f32 * f;
            }
            accm[s] += (m as f32 * -dmin) * x.dxs[xb];
        }
    }
    reduce8(&acc) + reduce8(&accm)
}

/// Q4_K weights (`docs/quant-layouts.md`): sub-block `s` lives in the low (`s` even) or high (`s` odd) nibbles
/// of bytes `16 + 32 * (s / 2) ..`.
pub fn dot_q4_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    dot_q45_k_q8(w, x, Q4_K_BLOCK_BYTES, |wb, s, j| {
        let q = wb[16 + (s / 2) * 32 + j];
        (if s % 2 == 0 { q & 0xF } else { q >> 4 }) as i32
    })
}

/// Q5_K weights: as Q4_K plus bit `s` of `qh[j]` as the fifth bit.
pub fn dot_q5_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    dot_q45_k_q8(w, x, Q5_K_BLOCK_BYTES, |wb, s, j| {
        let q = wb[48 + (s / 2) * 32 + j];
        let lo = (if s % 2 == 0 { q & 0xF } else { q >> 4 }) as i32;
        lo + if wb[16 + j] & (1 << s) != 0 { 16 } else { 0 }
    })
}

/// Q6_K weights: unsigned 6-bit codes, int8 scale per 16-group (lanes 0..4 and 4..8 of a sub-block have
/// different scales), the -32 offset subtracted in integer from lanes 0 and 4 (`off6`).
pub fn dot_q6_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q6_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * Q6_K_BLOCK_BYTES, "dot_q6_k_q8: weight row bytes");
    let mut acc = [0f32; 8];
    for sb in 0..nb / 8 {
        let wb = &w[sb * Q6_K_BLOCK_BYTES..(sb + 1) * Q6_K_BLOCK_BYTES];
        let ql_all = &wb[0..128];
        let qh_all = &wb[128..192];
        let sc_all = &wb[192..208];
        let d = f16_at(wb, 208);
        for half in 0..2 {
            let ql = &ql_all[half * 64..(half + 1) * 64];
            let qh = &qh_all[half * 32..(half + 1) * 32];
            let sc = &sc_all[half * 8..(half + 1) * 8];
            for k in 0..4 {
                let xb = sb * 8 + half * 4 + k;
                let xq = x.block(xb);
                let mut p32 = [0i32; 8];
                for l in 0..32 {
                    let lo = match k {
                        0 => ql[l] & 0xF,
                        1 => ql[l + 32] & 0xF,
                        2 => ql[l] >> 4,
                        _ => ql[l + 32] >> 4,
                    };
                    let hi = (qh[l] >> (2 * k)) & 3;
                    p32[l / 4] += (lo | (hi << 4)) as i32 * xq[l] as i32;
                }
                p32[0] -= x.off6[xb * 8];
                p32[4] -= x.off6[xb * 8 + 4];
                let f_lo = x.d32[xb] * (d * (sc[2 * k] as i8) as f32);
                let f_hi = x.d32[xb] * (d * (sc[2 * k + 1] as i8) as f32);
                for i in 0..8 {
                    acc[i] += p32[i] as f32 * if i < 4 { f_lo } else { f_hi };
                }
            }
        }
    }
    reduce8(&acc)
}

/// f32 weights times a Q8_0 activation row: the activation is dequantised on the fly (`q * d`) and each
/// 32-block partial is a left-to-right f32 sum of products.
pub fn dot_f32_q8(w: &[f32], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert_eq!(w.len(), x.n, "dot_f32_q8: weight row length");
    let mut acc = 0f32;
    for b in 0..nb {
        let d = x.scale(b);
        let xq = x.block(b);
        let wb = &w[b * 32..(b + 1) * 32];
        let mut part = 0f32;
        for j in 0..32 {
            part += wb[j] * (xq[j] as f32 * d);
        }
        acc += part;
    }
    acc
}

/// f32 weights times f32 activations (used by F32-weight layers): 32-element block partials, left to right.
pub fn dot_f32(w: &[f32], x: &[f32]) -> f32 {
    assert_eq!(w.len(), x.len(), "dot_f32: lengths");
    let mut acc = 0f32;
    let mut i = 0;
    while i + 32 <= w.len() {
        let mut part = 0f32;
        for j in 0..32 {
            part += w[i + j] * x[i + j];
        }
        acc += part;
        i += 32;
    }
    let mut part = 0f32;
    while i < w.len() {
        part += w[i] * x[i];
        i += 1;
    }
    acc + part
}

/// Dispatch by weight type and instruction path. `w` is one row in ggml layout (or f32 bytes for F32).
/// The AVX2 kernels are bit-identical to the scalar ones (`avx2.rs`), so the path never changes the result.
pub fn dot_q8(t: GgmlType, w: &[u8], x: &Q8Row) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::simd::use_avx2() {
        // SAFETY: `use_avx2` is only true when the CPU reports AVX2 and F16C.
        if let Some(v) = unsafe { super::avx2::dot_q8(t, w, x) } {
            return v;
        }
    }
    dot_q8_scalar(t, w, x)
}

/// The scalar reference dispatch (what `dot_q8` computes on any CPU).
pub fn dot_q8_scalar(t: GgmlType, w: &[u8], x: &Q8Row) -> f32 {
    match t {
        GgmlType::Q8_0 => dot_q8_0_q8(w, x),
        GgmlType::Q4_0 => dot_q4_0_q8(w, x),
        GgmlType::Q4_K => dot_q4_k_q8(w, x),
        GgmlType::Q5_K => dot_q5_k_q8(w, x),
        GgmlType::Q6_K => dot_q6_k_q8(w, x),
        GgmlType::F32 => {
            let wf: Vec<f32> = w.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            dot_f32_q8(&wf, x)
        }
        other => panic!("dot_q8_scalar: unsupported weight type {other:?}"),
    }
}
