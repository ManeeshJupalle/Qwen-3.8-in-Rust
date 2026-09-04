//! Feature-gated stage profiler for the decode path (Phase 3.6, item 2).
//!
//! Phase 3.5 measured a decode token at 1.15 s of which about 0.46 s is not matvec (finding 50), from the
//! file's type mix and the measured per-type kernel rates. That is an inference, not a measurement. This
//! module measures it directly: build with `--features profile` and every stage of `Model::forward_token`
//! reports its own wall time.
//!
//! **Exclusive time.** Scopes nest (a projection is `Quantise` inside `Matvec`), so entering a scope charges
//! the elapsed time to the *enclosing* stage first and exiting charges it to the stage being left. Each
//! stage therefore accumulates only the time spent in it and not in a child, and the totals sum to the wall
//! time of the outermost scope. That is what makes the breakdown add up to the token.
//!
//! **One thread.** Only the calling thread records. The row-parallel and head-parallel regions run on the
//! pool, and `pool::run` returns only when every participant has finished, so the caller's wall time across
//! such a region is the region's real duration including the workers. Worker threads never enter a scope.
//!
//! With the feature off, `scope!` expands to nothing and this module compiles to a few empty functions.

/// The stages a decode token is divided into. `Matvec` is the dot-product kernels (what the Phase 3.5 bench
/// measures); everything else is the "non-matvec" work finding 50 left unaccounted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum Stage {
    /// Weight-row dot products: `matvec`, `matvec2`, and the lm_head.
    Matvec,
    /// Quantising an activation vector to Q8_K / Q8_0 before a projection.
    Quantise,
    /// The DeltaNet per-head recurrence: l2norm of q/k, the delta rule step, the gated RMSNorm.
    DeltaNetRec,
    /// The causal depthwise conv before the DeltaNet projections.
    Conv,
    /// The attention per-head scores, weighted V sum and output gating (softmax counted separately).
    Attn,
    /// Softmax over the attention scores.
    Softmax,
    /// RoPE tables and rotation, plus the per-head q/k RMSNorms that feed it.
    Rope,
    /// The decoder layer's two RMSNorms and the final norm before the head.
    Norm,
    /// SwiGLU in the MLP.
    Swiglu,
    /// Residual adds, gate arithmetic, buffer copies.
    Residual,
    /// Dequantising the embedding row of the input token.
    Embed,
}

impl Stage {
    pub const ALL: [Stage; 11] = [
        Stage::Matvec,
        Stage::Quantise,
        Stage::DeltaNetRec,
        Stage::Conv,
        Stage::Attn,
        Stage::Softmax,
        Stage::Rope,
        Stage::Norm,
        Stage::Swiglu,
        Stage::Residual,
        Stage::Embed,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Matvec => "matvec",
            Stage::Quantise => "quantise",
            Stage::DeltaNetRec => "deltanet_rec",
            Stage::Conv => "conv",
            Stage::Attn => "attn",
            Stage::Softmax => "softmax",
            Stage::Rope => "rope",
            Stage::Norm => "norm",
            Stage::Swiglu => "swiglu",
            Stage::Residual => "residual",
            Stage::Embed => "embed",
        }
    }

    /// Is this stage part of the dot-product kernels the bench measures?
    pub fn is_matvec(self) -> bool {
        self == Stage::Matvec
    }
}

pub const N_STAGES: usize = Stage::ALL.len();

/// Open a profiling scope for the rest of the enclosing block. No-op without the `profile` feature.
#[macro_export]
macro_rules! prof_scope {
    ($stage:expr) => {
        #[cfg(feature = "profile")]
        let _prof_guard = $crate::prof::Guard::enter($stage);
    };
}

#[cfg(feature = "profile")]
mod imp {
    use super::{Stage, N_STAGES};
    use std::cell::RefCell;
    use std::time::Instant;

    struct Prof {
        /// (stage, when the current stretch of that stage started)
        stack: Vec<(Stage, Instant)>,
        totals: [u64; N_STAGES],
        /// Wall time of the outermost scope, to check the stages add up.
        outer_ns: u64,
    }

    thread_local! {
        static P: RefCell<Prof> = const { RefCell::new(Prof { stack: Vec::new(), totals: [0; N_STAGES], outer_ns: 0 }) };
    }

    /// Charges elapsed time to the stage on top of the stack and restarts its clock.
    fn charge_top(p: &mut Prof, now: Instant) {
        if let Some((s, t)) = p.stack.last_mut() {
            p.totals[*s as usize] += now.duration_since(*t).as_nanos() as u64;
            *t = now;
        }
    }

    pub struct Guard;

    impl Guard {
        pub fn enter(stage: Stage) -> Guard {
            P.with(|p| {
                let mut p = p.borrow_mut();
                let now = Instant::now();
                charge_top(&mut p, now);
                p.stack.push((stage, now));
            });
            Guard
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            P.with(|p| {
                let mut p = p.borrow_mut();
                let now = Instant::now();
                if let Some((s, t)) = p.stack.pop() {
                    let d = now.duration_since(t).as_nanos() as u64;
                    p.totals[s as usize] += d;
                    if p.stack.is_empty() {
                        // the outermost scope just closed; its span is the sum of everything inside it
                        p.outer_ns = p.totals.iter().sum();
                    } else if let Some((_, pt)) = p.stack.last_mut() {
                        *pt = now;
                    }
                }
            });
        }
    }

    pub fn reset() {
        P.with(|p| {
            let mut p = p.borrow_mut();
            p.stack.clear();
            p.totals = [0; N_STAGES];
            p.outer_ns = 0;
        });
    }

    pub fn totals_ns() -> [u64; N_STAGES] {
        P.with(|p| p.borrow().totals)
    }

    pub fn enabled() -> bool {
        true
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    use super::N_STAGES;
    pub fn reset() {}
    pub fn totals_ns() -> [u64; N_STAGES] {
        [0; N_STAGES]
    }
    pub fn enabled() -> bool {
        false
    }
}

pub use imp::{enabled, reset, totals_ns};
#[cfg(feature = "profile")]
pub use imp::Guard;

/// Nanoseconds per stage since the last `reset`, in `Stage::ALL` order. All zero without the feature.
pub fn totals() -> Vec<(Stage, u64)> {
    let t = totals_ns();
    Stage::ALL.iter().map(|&s| (s, t[s as usize])).collect()
}

/// A one-line-per-stage report: nanoseconds, share of the total, and the matvec / non-matvec split.
pub fn report(steps: usize) -> String {
    let t = totals_ns();
    let total: u64 = t.iter().sum();
    let matvec: u64 = Stage::ALL.iter().filter(|s| s.is_matvec()).map(|&s| t[s as usize]).sum();
    let mut out = String::new();
    if !enabled() {
        return "profiling not compiled in (build with --features profile)\n".to_string();
    }
    let steps = steps.max(1) as f64;
    out.push_str(&format!("{:<14} {:>12} {:>9} {:>9}\n", "stage", "ms/token", "% token", "total ms"));
    for &s in &Stage::ALL {
        let ns = t[s as usize];
        out.push_str(&format!(
            "{:<14} {:>12.3} {:>8.1}% {:>9.1}\n",
            s.name(),
            ns as f64 / steps / 1e6,
            if total > 0 { 100.0 * ns as f64 / total as f64 } else { 0.0 },
            ns as f64 / 1e6
        ));
    }
    out.push_str(&format!(
        "{:<14} {:>12.3} {:>8.1}% {:>9.1}\n",
        "TOTAL",
        total as f64 / steps / 1e6,
        100.0,
        total as f64 / 1e6
    ));
    out.push_str(&format!(
        "{:<14} {:>12.3} {:>8.1}%\n",
        "  matvec",
        matvec as f64 / steps / 1e6,
        if total > 0 { 100.0 * matvec as f64 / total as f64 } else { 0.0 }
    ));
    out.push_str(&format!(
        "{:<14} {:>12.3} {:>8.1}%\n",
        "  non-matvec",
        (total - matvec) as f64 / steps / 1e6,
        if total > 0 { 100.0 * (total - matvec) as f64 / total as f64 } else { 0.0 }
    ));
    out
}
