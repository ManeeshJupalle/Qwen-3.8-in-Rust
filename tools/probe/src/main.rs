//! Micro-probes of the i7-9750H for the blocked-kernel investigation: the AVX2 integer clock, the throughput
//! of maddubs -> madd -> add triples from registers and from L1 (indexed addressing), and the single-row Q4_K
//! kernel on an L1-resident row with its pieces switched off one at a time.
mod ports;
mod variants;
mod variants2;

use std::arch::x86_64::*;
use std::time::Instant;

use aqueduct_core::kernels::kdot::dot_q8k;
use aqueduct_core::kernels::q8k::Q8KRow;
use aqueduct_core::GgmlType;

/// AVX2 integer clock: `x = x + (x >> 7)` on ymm (shift 1 cycle, add 1 cycle).
#[target_feature(enable = "avx2")]
unsafe fn clock_avx2() -> f64 {
    let iters = 300_000_000u64;
    let mut x = _mm256_set1_epi32(std::hint::black_box(12345));
    let t0 = Instant::now();
    for _ in 0..iters {
        x = _mm256_add_epi32(x, _mm256_srli_epi32(x, 7));
    }
    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, x);
    std::hint::black_box(out);
    2.0 * iters as f64 / t0.elapsed().as_secs_f64() / 1e9
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn row_and_act(cols: usize) -> (Vec<u8>, Q8KRow) {
    let nb = cols / 256;
    let mut rng = Rng(0x1234_5678);
    let mut row: Vec<u8> = (0..nb * 144).map(|_| (rng.next() >> 24) as u8).collect();
    for blk in row.chunks_mut(144) {
        for off in [0usize, 2] {
            let bits = u16::from_le_bytes([blk[off], blk[off + 1]]);
            let fixed = (bits & 0x83FF) | ((12 + (bits >> 10) % 7) << 10);
            blk[off..off + 2].copy_from_slice(&fixed.to_le_bytes());
        }
    }
    let x: Vec<f32> = (0..cols).map(|_| (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect();
    (row, Q8KRow::quantize(&x))
}

fn time_cycles(ghz: f64, nb: usize, iters: usize, mut f: impl FnMut() -> f32) -> f64 {
    let mut acc = 0f32;
    f();
    let t0 = Instant::now();
    for _ in 0..iters {
        acc += f();
    }
    std::hint::black_box(acc);
    t0.elapsed().as_secs_f64() * ghz * 1e9 / (iters * nb) as f64
}

fn main() {
    let ghz = unsafe { clock_avx2() };
    println!("clock: avx2 integer chain {ghz:.2} GHz");
    if std::env::args().any(|a| a == "--ports") {
        unsafe {
            println!("reciprocal throughput, cycles per instruction (8 independent chains, register operands):");
            println!("  vpmaddubsw ymm : {:.2}  (table 0.50)", ports::maddubs(ghz));
            println!("  vpmaddwd ymm   : {:.2}  (table 0.50)", ports::maddwd(ghz));
            println!("  maddubs+madd   : {:.2}  per pair (table 1.00)", ports::maddubs_madd(ghz));
            println!("  vpaddd ymm     : {:.2}  (table 0.33)", ports::paddd(ghz));
            println!("  vpand ymm      : {:.2}  (table 0.33)", ports::pand(ghz));
            println!("  vpsrlw ymm     : {:.2}  (table 0.50)", ports::psrlw(ghz));
            println!("  vpshufb ymm    : {:.2}  (table 1.00)", ports::pshufb(ghz));
            println!("  vmulps ymm     : {:.2}  (table 0.50)", ports::mulps(ghz));
            println!("  vaddps ymm     : {:.2}  (table 0.50)", ports::addps(ghz));
            println!("  triple, L1 loads at line offset  0 : {:.2} cycles", ports::maddubs_l1_align(ghz, 0));
            println!("  triple, L1 loads at line offset 16 : {:.2} cycles", ports::maddubs_l1_align(ghz, 16));
            println!("  triple, L1 loads at line offset 32 : {:.2} cycles", ports::maddubs_l1_align(ghz, 32));
            println!("  triple, L1 loads at line offset 48 : {:.2} cycles", ports::maddubs_l1_align(ghz, 48));
        }
        return;
    }
    let cols = 5120usize;
    let nb = cols / 256;
    let (row, q) = row_and_act(cols);
    let iters = 100_000;
    println!("cycles per super-block, one L1-resident Q4_K row (20 super-blocks) x one Q8_K row:");
    println!("  production dot_q4_k_q8k                : {:.1}", time_cycles(ghz, nb, iters, || dot_q8k(GgmlType::Q4_K, std::hint::black_box(&row), &q)));
    macro_rules! v {
        ($name:expr, $pf:expr, $hdr:expr, $mins:expr, $shuf:expr, $unpack:expr, $tail:expr) => {
            println!("  {:<40}: {:.1}", $name, time_cycles(ghz, nb, iters, || unsafe { variants::q4k::<$pf, $hdr, $mins, $shuf, $unpack, $tail>(std::hint::black_box(&row), &q) }));
        };
    }
    v!("copy, everything on", true, true, true, true, true, true);
    v!("no prefetch", false, true, true, true, true, true);
    v!("no header (constants)", true, false, true, true, true, true);
    v!("no mins term", true, true, false, true, true, true);
    v!("no scale shuffles", true, true, true, false, true, true);
    v!("no nibble unpack", true, true, true, true, false, true);
    v!("no f32 tail (i32 acc)", true, true, true, true, true, false);
    v!("no header, no mins, no shuffles", true, false, false, false, true, true);
    v!("core only: unpack+madd, i32 acc", false, false, false, false, true, false);
    v!("bare madd, i32 acc", false, false, false, false, false, false);
    macro_rules! v2 {
        ($name:expr, $pipe:expr, $split:expr, $scalar:expr) => {
            let r = unsafe { variants2::q4k::<$pipe, $split, $scalar>(&row, &q) };
            let want = dot_q8k(GgmlType::Q4_K, &row, &q);
            println!("  {:<40}: {:.1}   (result {} vs production {})", $name, time_cycles(ghz, nb, iters, || unsafe { variants2::q4k::<$pipe, $split, $scalar>(std::hint::black_box(&row), &q) }), r, want);
        };
    }
    v2!("restructured, plain", false, false, false);
    v2!("B: pipelined header", true, false, false);
    v2!("C: split accumulators", false, true, false);
    v2!("E: scalar scale unpack", false, false, true);
    v2!("B+C", true, true, false);
    v2!("B+E", true, false, true);
    v2!("F: B+C+E", true, true, true);
    println!("clock again: {:.2} GHz", unsafe { clock_avx2() });
}
