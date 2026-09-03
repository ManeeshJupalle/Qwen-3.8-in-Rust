//! Runtime selection of the AVX2 kernels. `use_avx2()` is true when the CPU has AVX2 and F16C, the build is
//! x86_64, and nothing forced the scalar path (`AQUEDUCT_SCALAR=1` in the environment, or `force_scalar(true)`
//! from a test or the bench). The choice is cached in an atomic; the scalar kernels remain the reference and
//! are bit-identical to the AVX2 ones (see `avx2.rs`), so the selection never changes results.

use std::sync::atomic::{AtomicU8, Ordering};

const UNKNOWN: u8 = 0;
const SCALAR: u8 = 1;
const AVX2: u8 = 2;

static MODE: AtomicU8 = AtomicU8::new(UNKNOWN);

/// Does this CPU (and build) have the AVX2 + F16C path at all?
pub fn avx2_detected() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn detect() -> u8 {
    if std::env::var_os("AQUEDUCT_SCALAR").is_some() || !avx2_detected() {
        SCALAR
    } else {
        AVX2
    }
}

/// Should the dispatching kernels take the AVX2 path right now?
#[inline]
pub fn use_avx2() -> bool {
    match MODE.load(Ordering::Relaxed) {
        AVX2 => true,
        SCALAR => false,
        _ => {
            let m = detect();
            MODE.store(m, Ordering::Relaxed);
            m == AVX2
        }
    }
}

/// Force the scalar path (`true`) or go back to detection (`false`). For tests and `aqueduct bench kernels`.
pub fn force_scalar(on: bool) {
    MODE.store(if on { SCALAR } else { detect() }, Ordering::Relaxed);
}

/// Name of the path currently selected, for reports.
pub fn path_name() -> &'static str {
    if use_avx2() {
        "avx2"
    } else {
        "scalar"
    }
}
