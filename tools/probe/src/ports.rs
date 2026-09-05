//! Throughput of the instruction classes the K-quant kernels lean on: 8 independent chains, register operands,
//! cycles per instruction (the reciprocal throughput). Skylake tables say maddubs/madd 0.5 (p01), paddd 0.33
//! (p015), pshufb 1.0 (p5), cvtdq2ps 0.5 (p01), mulps 0.5 (p01), addps 0.5 (p01).
use std::arch::x86_64::*;
use std::time::Instant;

macro_rules! probe {
    ($name:ident, $init:expr, $step:expr) => {
        #[target_feature(enable = "avx2")]
        pub unsafe fn $name(ghz: f64) -> f64 {
            let iters = 40_000_000u64;
            let k = $init;
            let mut a = [k; 8];
            for (i, v) in a.iter_mut().enumerate() {
                *v = _mm256_add_epi32(*v, _mm256_set1_epi32(i as i32));
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                for j in 0..8 {
                    a[j] = $step(a[j], k);
                }
            }
            let mut s = _mm256_setzero_si256();
            for v in a {
                s = _mm256_add_epi32(s, v);
            }
            let mut out = [0i32; 8];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, s);
            std::hint::black_box(out);
            t0.elapsed().as_secs_f64() * ghz * 1e9 / (iters * 8) as f64
        }
    };
}

// each step is one instruction whose result feeds the same chain (8 independent chains of latency L:
// throughput-bound when 8 * TP >= L, which holds for every class below)
probe!(maddubs, _mm256_set1_epi8(std::hint::black_box(3)), |a, k| _mm256_maddubs_epi16(a, k));
probe!(maddwd, _mm256_set1_epi16(std::hint::black_box(3)), |a, k| _mm256_madd_epi16(a, k));
probe!(paddd, _mm256_set1_epi32(std::hint::black_box(3)), |a, k| _mm256_add_epi32(a, k));
probe!(pshufb, _mm256_set1_epi8(std::hint::black_box(3)), |a, k| _mm256_shuffle_epi8(a, k));
probe!(pand, _mm256_set1_epi32(std::hint::black_box(3)), |a, k| _mm256_and_si256(a, k));
probe!(psrlw, _mm256_set1_epi16(std::hint::black_box(3)), |a, _k| _mm256_srli_epi16(a, 1));
probe!(cvt_mul_add, _mm256_set1_epi32(std::hint::black_box(3)), |a, k| _mm256_castps_si256(_mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a), _mm256_castsi256_ps(k)), _mm256_castsi256_ps(k))));
probe!(mulps, _mm256_set1_epi32(std::hint::black_box(0x3f80_0001)), |a, k| _mm256_castps_si256(_mm256_mul_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(k))));
probe!(addps, _mm256_set1_epi32(std::hint::black_box(0x3f80_0001)), |a, k| _mm256_castps_si256(_mm256_add_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(k))));
probe!(maddubs_madd, _mm256_set1_epi8(std::hint::black_box(3)), |a, k| _mm256_madd_epi16(_mm256_maddubs_epi16(a, k), k));

/// maddubs -> madd -> add with the activation operand loaded from an L1 buffer at the given offset from a
/// 64-byte line (16: what `Vec<i8>` gets from the Windows heap, half the 32-byte loads split a line; 0: none do).
#[target_feature(enable = "avx2")]
pub unsafe fn maddubs_l1_align(ghz: f64, offset: usize) -> f64 {
    let iters = 20_000_000u64;
    let raw: Vec<u8> = (0..8 * 256 + 128).map(|i| (i * 7 % 251) as u8).collect();
    let mut p = raw.as_ptr();
    while (p as usize) % 64 != 0 {
        p = p.add(1);
    }
    let p = p.add(offset);
    let w = _mm256_set1_epi8(std::hint::black_box(7));
    let sc = _mm256_set1_epi16(std::hint::black_box(5));
    let mut a = [_mm256_setzero_si256(); 8];
    let t0 = Instant::now();
    for i in 0..iters {
        let off = ((i & 7) * 32) as usize;
        for k in 0..8 {
            let xv = _mm256_loadu_si256(p.add(k * 256 + off) as *const __m256i);
            a[k] = _mm256_add_epi32(a[k], _mm256_madd_epi16(sc, _mm256_maddubs_epi16(w, xv)));
        }
    }
    let mut s = _mm256_setzero_si256();
    for v in a {
        s = _mm256_add_epi32(s, v);
    }
    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, s);
    std::hint::black_box(out);
    t0.elapsed().as_secs_f64() * ghz * 1e9 / (iters * 8) as f64
}
