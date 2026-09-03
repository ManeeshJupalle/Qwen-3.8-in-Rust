//! Row dot products: one row of weights (ggml block layout) times one Q8_0 activation row -> f32.
//!
//! Summation order (all kernels): the row is walked in 32-element activation blocks. Inside a block the
//! integer products `q_w * q_x` are accumulated exactly in i32; the block partial is converted to f32 and
//! scaled; block partials are accumulated left to right in f32. K-quants add the sub-block scale/min
//! algebra shown per kernel. For f32 weights the block partial is a left-to-right f32 sum of 32 products.
//! Expected error ~ (n/32) * eps * rms(block partial), well inside the budget k*sqrt(n)*eps*max|term| (k = 8).

use super::q8::Q8Row;
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

/// Q4_K weights: super-blocks of 256 = 8 sub-blocks of 32 with 6-bit `sc`, `m`. Per sub-block `s` matched with
/// activation block `b`: `d_x * ((d * sc) * sum(q_w * q_x) - (dmin * m) * sum(q_x))`.
pub fn dot_q4_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q4_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * Q4_K_BLOCK_BYTES, "dot_q4_k_q8: weight row bytes");
    let mut acc = 0f32;
    for sb in 0..nb / 8 {
        let wb = &w[sb * Q4_K_BLOCK_BYTES..(sb + 1) * Q4_K_BLOCK_BYTES];
        let d = f16_at(wb, 0);
        let dmin = f16_at(wb, 2);
        let scales = &wb[4..16];
        let qs = &wb[16..144];
        // pairs of sub-blocks share 32 bytes: low nibbles = sub-block 2p, high nibbles = 2p+1
        for p in 0..4 {
            let q = &qs[p * 32..(p + 1) * 32];
            for half in 0..2 {
                let s = 2 * p + half;
                let (sc, m) = get_scale_min_k4(s, scales);
                let xb = sb * 8 + s;
                let xq = x.block(xb);
                let mut isum: i32 = 0;
                let mut xsum: i32 = 0;
                for j in 0..32 {
                    let qw = if half == 0 { q[j] & 0xF } else { q[j] >> 4 } as i32;
                    isum += qw * xq[j] as i32;
                    xsum += xq[j] as i32;
                }
                acc += x.scale(xb) * ((d * sc as f32) * isum as f32 - (dmin * m as f32) * xsum as f32);
            }
        }
    }
    acc
}

/// Q5_K weights: as Q4_K with the fifth bit from the `qh` plane (bit 2p for sub-block 2p, bit 2p+1 for 2p+1).
pub fn dot_q5_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q5_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * Q5_K_BLOCK_BYTES, "dot_q5_k_q8: weight row bytes");
    let mut acc = 0f32;
    for sb in 0..nb / 8 {
        let wb = &w[sb * Q5_K_BLOCK_BYTES..(sb + 1) * Q5_K_BLOCK_BYTES];
        let d = f16_at(wb, 0);
        let dmin = f16_at(wb, 2);
        let scales = &wb[4..16];
        let qh = &wb[16..48];
        let ql = &wb[48..176];
        for p in 0..4 {
            let q = &ql[p * 32..(p + 1) * 32];
            for half in 0..2 {
                let s = 2 * p + half;
                let bit: u8 = 1 << (2 * p + half);
                let (sc, m) = get_scale_min_k4(s, scales);
                let xb = sb * 8 + s;
                let xq = x.block(xb);
                let mut isum: i32 = 0;
                let mut xsum: i32 = 0;
                for j in 0..32 {
                    let lo = if half == 0 { q[j] & 0xF } else { q[j] >> 4 } as i32;
                    let qw = lo + if qh[j] & bit != 0 { 16 } else { 0 };
                    isum += qw * xq[j] as i32;
                    xsum += xq[j] as i32;
                }
                acc += x.scale(xb) * ((d * sc as f32) * isum as f32 - (dmin * m as f32) * xsum as f32);
            }
        }
    }
    acc
}

/// Q6_K weights: 16 int8 scales per 256 (one per 16 values). Per activation block (two 16-groups g0, g1):
/// `d_x * ((d * sc_g0) * sum16(q_w * q_x) + (d * sc_g1) * sum16(q_w * q_x))` with `q_w = 6-bit - 32`.
pub fn dot_q6_k_q8(w: &[u8], x: &Q8Row) -> f32 {
    let nb = x.n_blocks();
    assert!(nb.is_multiple_of(8), "dot_q6_k_q8: activation length must be a multiple of 256");
    assert_eq!(w.len(), nb / 8 * Q6_K_BLOCK_BYTES, "dot_q6_k_q8: weight row bytes");
    let mut acc = 0f32;
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
            // quarter k (0..4) covers values 128*half + 32*k + l, l in 0..32
            for k in 0..4 {
                let xb = sb * 8 + half * 4 + k;
                let xq = x.block(xb);
                let mut isum = [0i32; 2];
                for l in 0..32 {
                    let lo = match k {
                        0 => ql[l] & 0xF,
                        1 => ql[l + 32] & 0xF,
                        2 => ql[l] >> 4,
                        _ => ql[l + 32] >> 4,
                    };
                    let hi = (qh[l] >> (2 * k)) & 3;
                    let qw = (lo | (hi << 4)) as i32 - 32;
                    isum[l / 16] += qw * xq[l] as i32;
                }
                let s0 = (sc[2 * k] as i8) as f32;
                let s1 = (sc[2 * k + 1] as i8) as f32;
                acc += x.scale(xb) * ((d * s0) * isum[0] as f32 + (d * s1) * isum[1] as f32);
            }
        }
    }
    acc
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

/// Dispatch by weight type. `w` is one row in ggml layout (or f32 bytes for F32).
pub fn dot_q8(t: GgmlType, w: &[u8], x: &Q8Row) -> f32 {
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
        other => panic!("dot_q8: unsupported weight type {other:?}"),
    }
}
