//! Block layouts and scalar dequantisation for the ggml types present in the Q4_K_M file:
//! Q4_0, Q8_0, Q4_K, Q5_K, Q6_K and F32. Layouts follow ggml's `ggml-common.h`; the arithmetic order
//! follows `dequantize_row_*` in `ggml-quants.c` (which gguf-py's numpy implementation also follows),
//! so results are expected to be bit-identical to `tools/emit_dequant_fixtures.py`.
//!
//! No SIMD, no matmul: this module only turns bytes into f32.
//! Byte-offset tables for every layout are in `docs/quant-layouts.md`.

use crate::gguf::GgmlType;

/// Elements per Q4_0 / Q8_0 block.
pub const QK_32: usize = 32;
/// Elements per K-quant super-block.
pub const QK_K: usize = 256;

pub const Q4_0_BLOCK_BYTES: usize = 18; // f16 d + 16 x u8 nibbles
pub const Q8_0_BLOCK_BYTES: usize = 34; // f16 d + 32 x i8
pub const Q4_K_BLOCK_BYTES: usize = 144; // f16 d + f16 dmin + 12 scales + 128 nibbles
pub const Q5_K_BLOCK_BYTES: usize = 176; // f16 d + f16 dmin + 12 scales + 32 qh + 128 ql
pub const Q6_K_BLOCK_BYTES: usize = 210; // 128 ql + 64 qh + 16 i8 scales + f16 d

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum QuantError {
    #[error("{ggml_type:?}: source has {src_len} bytes but {n} elements need {want} bytes")]
    SrcLen { ggml_type: GgmlType, src_len: usize, n: usize, want: usize },
    #[error("{ggml_type:?}: element count {n} is not a multiple of block size {block}")]
    ElementCount { ggml_type: GgmlType, n: usize, block: usize },
    #[error("dequantisation of {0:?} is not implemented in this phase")]
    Unsupported(GgmlType),
}

/// IEEE binary16 -> f32 (exact).
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
fn f16_at(src: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]))
}

/// Unpack the j-th 6-bit (scale, min) pair of a Q4_K / Q5_K super-block (`get_scale_min_k4` in ggml).
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

fn check(t: GgmlType, src: &[u8], n: usize, block: usize, block_bytes: usize) -> Result<usize, QuantError> {
    if !n.is_multiple_of(block) {
        return Err(QuantError::ElementCount { ggml_type: t, n, block });
    }
    let nb = n / block;
    let want = nb * block_bytes;
    if src.len() != want {
        return Err(QuantError::SrcLen { ggml_type: t, src_len: src.len(), n, want });
    }
    Ok(nb)
}

/// Q4_0: per 32 values, f16 scale `d` then 16 bytes; low nibbles are values 0..16, high nibbles 16..32,
/// each `d * (q - 8)`.
pub fn dequant_q4_0(src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    let nb = check(GgmlType::Q4_0, src, dst.len(), QK_32, Q4_0_BLOCK_BYTES)?;
    for i in 0..nb {
        let b = &src[i * Q4_0_BLOCK_BYTES..(i + 1) * Q4_0_BLOCK_BYTES];
        let d = f16_at(b, 0);
        let qs = &b[2..18];
        let y = &mut dst[i * QK_32..(i + 1) * QK_32];
        for j in 0..16 {
            let x0 = (qs[j] & 0x0F) as i32 - 8;
            let x1 = (qs[j] >> 4) as i32 - 8;
            y[j] = x0 as f32 * d;
            y[j + 16] = x1 as f32 * d;
        }
    }
    Ok(())
}

/// Q8_0: per 32 values, f16 scale `d` then 32 signed bytes, each `q * d`.
pub fn dequant_q8_0(src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    let nb = check(GgmlType::Q8_0, src, dst.len(), QK_32, Q8_0_BLOCK_BYTES)?;
    for i in 0..nb {
        let b = &src[i * Q8_0_BLOCK_BYTES..(i + 1) * Q8_0_BLOCK_BYTES];
        let d = f16_at(b, 0);
        let y = &mut dst[i * QK_32..(i + 1) * QK_32];
        for j in 0..32 {
            y[j] = (b[2 + j] as i8) as f32 * d;
        }
    }
    Ok(())
}

/// Q4_K: per 256 values, f16 `d`, f16 `dmin`, 12 bytes of packed 6-bit scales/mins (8 each),
/// 128 bytes of nibbles. Sub-blocks of 32; for each pair of sub-blocks the 32 bytes hold the
/// first sub-block in the low nibbles and the second in the high nibbles.
/// `y = (d*sc) * q - (dmin*m)`.
pub fn dequant_q4_k(src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    let nb = check(GgmlType::Q4_K, src, dst.len(), QK_K, Q4_K_BLOCK_BYTES)?;
    for i in 0..nb {
        let b = &src[i * Q4_K_BLOCK_BYTES..(i + 1) * Q4_K_BLOCK_BYTES];
        let d = f16_at(b, 0);
        let min = f16_at(b, 2);
        let scales = &b[4..16];
        let qs = &b[16..144];
        let y = &mut dst[i * QK_K..(i + 1) * QK_K];
        let mut is = 0usize;
        let mut q = 0usize;
        let mut yo = 0usize;
        for _ in (0..QK_K).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for l in 0..32 {
                y[yo + l] = d1 * (qs[q + l] & 0xF) as f32 - m1;
            }
            for l in 0..32 {
                y[yo + 32 + l] = d2 * (qs[q + l] >> 4) as f32 - m2;
            }
            q += 32;
            is += 2;
            yo += 64;
        }
    }
    Ok(())
}

/// Q5_K: like Q4_K plus a 32-byte high-bit plane `qh` between the scales and the nibbles.
/// Bit `2k` of `qh[l]` is the 5th bit of value `64k + l`, bit `2k+1` that of value `64k + 32 + l`.
pub fn dequant_q5_k(src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    let nb = check(GgmlType::Q5_K, src, dst.len(), QK_K, Q5_K_BLOCK_BYTES)?;
    for i in 0..nb {
        let b = &src[i * Q5_K_BLOCK_BYTES..(i + 1) * Q5_K_BLOCK_BYTES];
        let d = f16_at(b, 0);
        let min = f16_at(b, 2);
        let scales = &b[4..16];
        let qh = &b[16..48];
        let ql = &b[48..176];
        let y = &mut dst[i * QK_K..(i + 1) * QK_K];
        let mut is = 0usize;
        let mut q = 0usize;
        let mut yo = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in (0..QK_K).step_by(64) {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * sc as f32;
            let m1 = min * m as f32;
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc as f32;
            let m2 = min * m as f32;
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                y[yo + l] = d1 * ((ql[q + l] & 0xF) + hi) as f32 - m1;
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                y[yo + 32 + l] = d2 * ((ql[q + l] >> 4) + hi) as f32 - m2;
            }
            q += 32;
            is += 2;
            yo += 64;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(())
}

/// Q6_K: per 256 values, 128 bytes of low nibbles `ql`, 64 bytes of 2-bit high planes `qh`,
/// 16 signed 8-bit sub-block scales, then f16 `d` at the end. Values are 6-bit minus 32;
/// `y = (d*sc) * q`. Two halves of 128 values; in each half value `l` (0..32) of quarter `k` (0..4)
/// takes its low nibble from `ql[l + 32*(k&1)]` (low nibble for k<2, high nibble for k>=2)
/// and its high bits from `(qh[l] >> 2k) & 3`.
pub fn dequant_q6_k(src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    let nb = check(GgmlType::Q6_K, src, dst.len(), QK_K, Q6_K_BLOCK_BYTES)?;
    for i in 0..nb {
        let b = &src[i * Q6_K_BLOCK_BYTES..(i + 1) * Q6_K_BLOCK_BYTES];
        let ql_all = &b[0..128];
        let qh_all = &b[128..192];
        let sc_all = &b[192..208];
        let d = f16_at(b, 208);
        let y = &mut dst[i * QK_K..(i + 1) * QK_K];
        for half in 0..2 {
            let ql = &ql_all[half * 64..(half + 1) * 64];
            let qh = &qh_all[half * 32..(half + 1) * 32];
            let sc = &sc_all[half * 8..(half + 1) * 8];
            let yo = half * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | (((qh[l]) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                y[yo + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                y[yo + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                y[yo + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                y[yo + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
    Ok(())
}

/// F32: little-endian bytes to f32.
pub fn dequant_f32(src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    check(GgmlType::F32, src, dst.len(), 1, 4)?;
    for (i, y) in dst.iter_mut().enumerate() {
        *y = f32::from_le_bytes([src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]]);
    }
    Ok(())
}

/// Dispatch on type. `n` is the element count.
pub fn dequantize(t: GgmlType, src: &[u8], n: usize) -> Result<Vec<f32>, QuantError> {
    let mut dst = vec![0f32; n];
    match t {
        GgmlType::Q4_0 => dequant_q4_0(src, &mut dst)?,
        GgmlType::Q8_0 => dequant_q8_0(src, &mut dst)?,
        GgmlType::Q4_K => dequant_q4_k(src, &mut dst)?,
        GgmlType::Q5_K => dequant_q5_k(src, &mut dst)?,
        GgmlType::Q6_K => dequant_q6_k(src, &mut dst)?,
        GgmlType::F32 => dequant_f32(src, &mut dst)?,
        other => return Err(QuantError::Unsupported(other)),
    }
    Ok(dst)
}

/// Standard CRC-32 (IEEE 802.3, reflected, init/final 0xFFFFFFFF), same as Python's `zlib.crc32`.
/// Used by the parity tests to compare full dequantised tensors against a digest stored in the fixture.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, e) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *e = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_exact() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8); // smallest subnormal
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
    }

    #[test]
    fn scale_min_unpack_matches_ggml_packing() {
        // pack as ggml's quantize_row_q4_K_ref does, then unpack
        let sc = [63u8, 1, 2, 4, 8, 16, 32, 33];
        let mn = [62u8, 3, 5, 9, 17, 31, 47, 60];
        let mut q = [0u8; 12];
        for j in 0..8 {
            if j < 4 {
                q[j] = sc[j];
                q[j + 4] = mn[j];
            } else {
                q[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4);
                q[j - 4] |= (sc[j] >> 4) << 6;
                q[j] |= (mn[j] >> 4) << 6;
            }
        }
        for j in 0..8 {
            assert_eq!(get_scale_min_k4(j, &q), (sc[j], mn[j]), "j={j}");
        }
    }

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn length_errors_are_reported() {
        let mut dst = vec![0f32; 32];
        assert!(matches!(dequant_q4_0(&[0u8; 17], &mut dst), Err(QuantError::SrcLen { .. })));
        let mut dst = vec![0f32; 33];
        assert!(matches!(dequant_q8_0(&[0u8; 34], &mut dst), Err(QuantError::ElementCount { .. })));
    }
}
