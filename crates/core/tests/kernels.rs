//! 2a.1: every kernel against its Python-generated fixture (tests/fixtures/kernels/<name>/manifest.json).
//! Budgets are derived in the test from the fixture's `budget_k`, width and `max_abs` (rule 5), never typed.

mod common;

use std::path::PathBuf;

use aqueduct_core::kernels::act::{silu_slice, swiglu};
use aqueduct_core::kernels::conv::{causal_conv1d_prefill, causal_conv1d_step};
use aqueduct_core::kernels::deltanet::{deltanet_step, gate_beta, gate_g, l2norm};
use aqueduct_core::kernels::dot::dot_q8;
use aqueduct_core::kernels::matvec::{matvec, Act, WeightMat};
use aqueduct_core::kernels::q8::Q8Row;
use aqueduct_core::kernels::rmsnorm::{rmsnorm, rmsnorm_gated};
use aqueduct_core::kernels::rope::{apply_rope, cos_sin, inv_freq};
use aqueduct_core::kernels::softmax::{softmax, softmax_causal_row};
use aqueduct_core::{dequantize, GgmlType};

const EPS: f64 = f32::EPSILON as f64;

struct Fx {
    dir: PathBuf,
    manifest: serde_json::Value,
}

impl Fx {
    fn load(name: &str) -> Fx {
        let dir = common::fixture("kernels").join(name);
        let manifest = common::json(&dir.join("manifest.json"));
        Fx { dir, manifest }
    }
    fn cases(&self) -> &Vec<serde_json::Value> {
        self.manifest["cases"].as_array().unwrap()
    }
    fn k(&self) -> f64 {
        self.manifest["budget_k"].as_f64().unwrap()
    }
    fn arr(&self, field: &serde_json::Value) -> Vec<f32> {
        common::read_npy_f32(&self.dir.join(field["file"].as_str().unwrap()))
    }
    fn shape(&self, field: &serde_json::Value) -> Vec<usize> {
        field["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
    }
    fn raw(&self, field: &serde_json::Value) -> Vec<u8> {
        std::fs::read(self.dir.join(field["file"].as_str().unwrap())).unwrap()
    }
}

fn budget(k: f64, width: usize, max_abs: f64) -> f64 {
    k * (width as f64).sqrt() * EPS * max_abs
}

/// Max abs difference; non-finite values must match exactly (same sign of infinity, NaN vs NaN).
fn max_diff(got: &[f32], want: &[f32]) -> (f64, usize) {
    assert_eq!(got.len(), want.len(), "length");
    let mut m = 0f64;
    let mut at = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = if g.is_finite() && w.is_finite() {
            (g as f64 - w as f64).abs()
        } else if g.is_nan() && w.is_nan() || g == w {
            0.0
        } else {
            f64::INFINITY
        };
        if d > m {
            m = d;
            at = i;
        }
    }
    (m, at)
}

fn check(label: &str, got: &[f32], want: &[f32], bud: f64) {
    let (d, at) = max_diff(got, want);
    assert!(d <= bud, "{label}: max abs diff {d:e} at {at} (got {} want {}) exceeds budget {bud:e}", got[at], want[at]);
    println!("{label}: max diff {d:.3e} <= budget {bud:.3e}");
}

/// Row-wise check: each row's budget is k * sqrt(width) * eps * max|want_row| (rows of different
/// magnitude in one fixture do not lend each other slack). A zero row must match exactly.
fn check_rows(label: &str, got: &[f32], want: &[f32], width: usize, k: f64) {
    assert_eq!(got.len(), want.len());
    let mut worst = (0f64, 0f64, 0usize);
    for r in 0..got.len() / width {
        let (g, w) = (&got[r * width..(r + 1) * width], &want[r * width..(r + 1) * width]);
        let max_abs = w.iter().filter(|v| v.is_finite()).fold(0f64, |m, v| m.max(v.abs() as f64));
        let bud = budget(k, width, max_abs);
        let (d, at) = max_diff(g, w);
        assert!(d <= bud, "{label} row {r}: max abs diff {d:e} at {at} (got {} want {}) exceeds budget {bud:e}", g[at], w[at]);
        if bud > 0.0 && d / bud > worst.0 / worst.1.max(1e-300) {
            worst = (d, bud, r);
        }
    }
    println!("{label}: worst row {} diff {:.3e} vs budget {:.3e}", worst.2, worst.0, worst.1);
}

/// Elementwise check with a per-element budget k * eps * |want| (floored at k * eps * MIN_POSITIVE so a
/// 1-ulp difference on a subnormal result passes); zero results must be exact.
fn check_elementwise(label: &str, got: &[f32], want: &[f32], k: f64) {
    assert_eq!(got.len(), want.len());
    let floor = k * EPS * f32::MIN_POSITIVE as f64;
    let mut worst = 0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let (d, _) = max_diff(&[g], &[w]);
        let bud = if w.is_finite() { (k * EPS * w.abs() as f64).max(if w == 0.0 { 0.0 } else { floor }) } else { 0.0 };
        assert!(d <= bud, "{label}[{i}]: got {g} want {w} diff {d:e} > budget {bud:e}");
        if bud > 0.0 {
            worst = worst.max(d / bud);
        }
    }
    println!("{label}: worst element at {:.2} of its budget", worst);
}

#[test]
fn rmsnorm_matches_hf() {
    let fx = Fx::load("rmsnorm");
    for c in fx.cases() {
        let width = c["width"].as_u64().unwrap() as usize;
        let eps = c["eps"].as_f64().unwrap() as f32;
        let x = fx.arr(&c["x"]);
        let w = fx.arr(&c["w_gguf"]);
        let want = fx.arr(&c["y"]);
        let rows = x.len() / width;
        let mut got = vec![0f32; x.len()];
        for r in 0..rows {
            let xr = &x[r * width..(r + 1) * width];
            let yr = &mut got[r * width..(r + 1) * width];
            match c["kind"].as_str().unwrap() {
                "rmsnorm" => rmsnorm(xr, &w, eps, yr),
                "rmsnorm_gated" => {
                    let gate = fx.arr(&c["gate"]);
                    rmsnorm_gated(xr, &gate[r * width..(r + 1) * width], &w, eps, yr)
                }
                k => panic!("{k}"),
            }
        }
        check_rows(c["case"].as_str().unwrap(), &got, &want, width, fx.k());
    }
}

#[test]
fn act_matches_torch() {
    let fx = Fx::load("act");
    for c in fx.cases() {
        let want = fx.arr(&c["y"]);
        let mut got = vec![0f32; want.len()];
        match c["kind"].as_str().unwrap() {
            "silu" => silu_slice(&fx.arr(&c["x"]), &mut got),
            "swiglu" => swiglu(&fx.arr(&c["gate"]), &fx.arr(&c["up"]), &mut got),
            k => panic!("{k}"),
        }
        check_elementwise(c["case"].as_str().unwrap(), &got, &want, fx.k());
    }
}

#[test]
fn softmax_matches_torch() {
    let fx = Fx::load("softmax");
    for c in fx.cases() {
        let want = fx.arr(&c["y"]);
        let mut got = vec![0f32; want.len()];
        let width;
        match c["kind"].as_str().unwrap() {
            "softmax" => {
                width = c["width"].as_u64().unwrap() as usize;
                let x = fx.arr(&c["x"]);
                for r in 0..x.len() / width {
                    softmax(&x[r * width..(r + 1) * width], &mut got[r * width..(r + 1) * width]);
                }
            }
            "softmax_causal" => {
                width = c["t"].as_u64().unwrap() as usize;
                let s = fx.arr(&c["scores"]);
                for r in 0..width {
                    softmax_causal_row(&s[r * width..(r + 1) * width], r, &mut got[r * width..(r + 1) * width]);
                }
                for r in 0..width {
                    for j in r + 1..width {
                        assert_eq!(got[r * width + j], 0.0, "masked entry must be exactly 0");
                    }
                }
            }
            k => panic!("{k}"),
        }
        check_rows(c["case"].as_str().unwrap(), &got, &want, width, fx.k());
    }
}

#[test]
fn q8_quantize_is_bit_exact_with_ggml() {
    let fx = Fx::load("q8_quantize");
    for c in fx.cases() {
        let width = c["width"].as_u64().unwrap() as usize;
        let x = fx.arr(&c["x"]);
        let raw = fx.raw(&c["raw"]);
        let deq = fx.arr(&c["deq"]);
        let rows = x.len() / width;
        let row_bytes = width / 32 * 34;
        for r in 0..rows {
            let q = Q8Row::quantize(&x[r * width..(r + 1) * width]);
            assert_eq!(q.to_ggml_bytes(), &raw[r * row_bytes..(r + 1) * row_bytes], "{} row {} ({}): bytes differ", c["case"], r, c["rows"][r]);
            let d = q.dequantize();
            let (diff, _) = max_diff(&d, &deq[r * width..(r + 1) * width]);
            assert_eq!(diff, 0.0, "{} row {}: dequantised values differ", c["case"], r);
            let back = Q8Row::from_ggml_bytes(&q.to_ggml_bytes(), width);
            assert_eq!(back, q);
        }
        println!("{}: {} rows bit-exact", c["case"], rows);
    }
}

#[test]
fn dot_kernels_within_budget() {
    let fx = Fx::load("dot");
    let mut worst: Vec<(String, f64, f64)> = Vec::new();
    for c in fx.cases() {
        let t = GgmlType::from_id(c["ggml_type_id"].as_u64().unwrap() as u32, "fixture").unwrap();
        let width = c["width"].as_u64().unwrap() as usize;
        let n_rows = c["n_rows"].as_u64().unwrap() as usize;
        let w_raw = fx.raw(&c["w_raw"]);
        let x_raw = fx.raw(&c["x_raw"]);
        let (bs, ts) = t.block_layout();
        let wrb = (width as u64 / bs * ts) as usize;
        let xrb = width / 32 * 34;
        let refs = c["ref_f64"].as_array().unwrap();
        let terms = c["max_abs_term"].as_array().unwrap();
        for r in 0..n_rows {
            let x = Q8Row::from_ggml_bytes(&x_raw[r * xrb..(r + 1) * xrb], width);
            let wrow = &w_raw[r * wrb..(r + 1) * wrb];
            let got = dot_q8(t, wrow, &x) as f64;
            let want = refs[r].as_f64().unwrap();
            // K-quants: Phase 3 kernels, budget from the term magnitudes (common::kquant_terms_budget)
            let bud = if common::is_kquant(t) {
                common::kquant_terms_budget(t, wrow, &dequantize(t, wrow, width).unwrap(), &x.dequantize(), fx.k())
            } else {
                budget(fx.k(), width, terms[r].as_f64().unwrap())
            };
            let d = (got - want).abs();
            let label = format!("{} row {} ({})", c["case"].as_str().unwrap(), r, c["rows"][r].as_str().unwrap());
            assert!(d <= bud, "{label}: got {got} want {want} diff {d:e} > budget {bud:e}");
            worst.push((label, d, bud));
        }
    }
    worst.sort_by(|a, b| (b.1 / b.2.max(1e-300)).partial_cmp(&(a.1 / a.2.max(1e-300))).unwrap());
    for (l, d, b) in worst.iter().take(6) {
        println!("closest to budget: {l}: diff {d:.3e} budget {b:.3e}");
    }
}

#[test]
fn matvec_within_budget_and_thread_invariant() {
    let fx = Fx::load("matvec");
    for c in fx.cases() {
        let t = GgmlType::from_id(c["ggml_type_id"].as_u64().unwrap() as u32, "fixture").unwrap();
        let rows = c["n_rows"].as_u64().unwrap() as usize;
        let width = c["width"].as_u64().unwrap() as usize;
        let w = WeightMat::new(t, rows, width, fx.raw(&c["w_raw"]));
        let x = Q8Row::from_ggml_bytes(&fx.raw(&c["x_raw"]), width);
        let refs: Vec<f64> = c["ref_f64"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let terms: Vec<f64> = c["max_abs_term"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let mut y1 = vec![0f32; rows];
        matvec(&w, Act::Q8(&x), &mut y1, 1);
        let xd = x.dequantize();
        for r in 0..rows {
            let bud = if common::is_kquant(t) {
                common::kquant_terms_budget(t, w.row(r), &dequantize(t, w.row(r), width).unwrap(), &xd, fx.k())
            } else {
                budget(fx.k(), width, terms[r])
            };
            let d = (y1[r] as f64 - refs[r]).abs();
            assert!(d <= bud, "{} row {r}: diff {d:e} > budget {bud:e}", c["case"]);
        }
        for threads in [2, 3, 5, 8, 64] {
            let mut yn = vec![0f32; rows];
            matvec(&w, Act::Q8(&x), &mut yn, threads);
            assert_eq!(y1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), yn.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{}: {threads} threads differ from 1 thread", c["case"]);
        }
        if t == GgmlType::F32 {
            // F32 weights also accept f32 activations (the dequantised Q8 row here, so the reference holds)
            let xd = fx.arr(&c["x_deq"]);
            let mut yf = vec![0f32; rows];
            matvec(&w, Act::F32(&xd), &mut yf, 3);
            for r in 0..rows {
                let d = (yf[r] as f64 - refs[r]).abs();
                assert!(d <= budget(fx.k(), width, terms[r]), "{} f32 path row {r}: diff {d:e}", c["case"]);
            }
        }
        println!("{}: {} rows ok, thread-invariant", c["case"], rows);
    }
}

#[test]
fn rope_tables_and_rotation_match_hf() {
    let fx = Fx::load("rope");
    for c in fx.cases() {
        let head_dim = c["head_dim"].as_u64().unwrap() as usize;
        let rope_dim = c["rope_dim"].as_u64().unwrap() as usize;
        let theta = c["theta"].as_f64().unwrap() as f32;
        let positions: Vec<u32> = c["positions"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let tlen = positions.len();
        // tables
        let inv = inv_freq(rope_dim, theta);
        let inv_hf = fx.arr(&c["inv_freq"]);
        let (dinv, _) = max_diff(&inv, &inv_hf);
        let ulps: Vec<i64> = inv.iter().zip(&inv_hf).map(|(a, b)| (a.to_bits() as i64 - b.to_bits() as i64).abs()).collect();
        println!("{}: inv_freq max diff {dinv:.3e}, max ulp distance {}", c["case"], ulps.iter().max().unwrap());
        assert!(*ulps.iter().max().unwrap() <= 1, "inv_freq differs from torch by more than 1 ulp");
        let cos_hf = fx.arr(&c["cos"]);
        let sin_hf = fx.arr(&c["sin"]);
        let mut cos = vec![0f32; rope_dim];
        let mut sin = vec![0f32; rope_dim];
        let mut worst_tab = 0f64;
        for (ti, &p) in positions.iter().enumerate() {
            cos_sin(p, &inv_hf, &mut cos, &mut sin);
            let (dc, _) = max_diff(&cos, &cos_hf[ti * rope_dim..(ti + 1) * rope_dim]);
            let (ds, _) = max_diff(&sin, &sin_hf[ti * rope_dim..(ti + 1) * rope_dim]);
            worst_tab = worst_tab.max(dc).max(ds);
        }
        let tab_budget = budget(fx.k(), 1, 1.0);
        println!("{}: cos/sin tables (torch inv_freq) max diff {worst_tab:.3e} vs budget {tab_budget:.3e}", c["case"]);
        assert!(worst_tab <= tab_budget, "rope tables differ from torch beyond {tab_budget:e}");
        // rotation with the HF tables
        for (name, out_name, heads) in [("q", "q_out", 4usize), ("k", "k_out", 2usize)] {
            let x = fx.arr(&c[name]);
            let want = fx.arr(&c[out_name]);
            assert_eq!(fx.shape(&c[name]), vec![heads, tlen, head_dim]);
            let mut got = x.clone();
            for h in 0..heads {
                for ti in 0..tlen {
                    let off = (h * tlen + ti) * head_dim;
                    apply_rope(&mut got[off..off + head_dim], head_dim, rope_dim, &cos_hf[ti * rope_dim..(ti + 1) * rope_dim], &sin_hf[ti * rope_dim..(ti + 1) * rope_dim]);
                }
            }
            check(&format!("{} {name}", c["case"].as_str().unwrap()), &got, &want, budget(fx.k(), 2, c["max_abs"].as_f64().unwrap()));
            // pass-through dims must be untouched bitwise
            for h in 0..heads {
                for ti in 0..tlen {
                    let off = (h * tlen + ti) * head_dim;
                    assert_eq!(&got[off + rope_dim..off + head_dim].iter().map(|v| v.to_bits()).collect::<Vec<_>>(), &x[off + rope_dim..off + head_dim].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                }
            }
        }
    }
}

#[test]
fn conv1d_step_and_prefill_match_hf() {
    let fx = Fx::load("conv1d");
    for c in fx.cases() {
        let channels = c["channels"].as_u64().unwrap() as usize;
        let kernel = c["kernel"].as_u64().unwrap() as usize;
        let w = fx.arr(&c["w"]);
        let want = fx.arr(&c["out"]);
        let bud = budget(fx.k(), kernel, c["max_abs"].as_f64().unwrap());
        match c["kind"].as_str().unwrap() {
            "conv_step" => {
                let mut state = fx.arr(&c["state"]);
                let x_new = fx.arr(&c["x_new"]);
                let mut out = vec![0f32; channels];
                causal_conv1d_step(&mut state, &w, kernel, &x_new, &mut out, true);
                check(c["case"].as_str().unwrap(), &out, &want, bud);
                let (ds, _) = max_diff(&state, &fx.arr(&c["new_state"]));
                assert_eq!(ds, 0.0, "{}: carried window must be a bit-exact shift", c["case"]);
            }
            "conv_prefill" => {
                let t = c["t"].as_u64().unwrap() as usize;
                let x = fx.arr(&c["x"]);
                let mut state = vec![0f32; channels * (kernel - 1)];
                let mut out = vec![0f32; channels * t];
                causal_conv1d_prefill(&mut state, &w, kernel, &x, channels, t, &mut out, true);
                check(c["case"].as_str().unwrap(), &out, &want, bud);
            }
            k => panic!("{k}"),
        }
    }
}

#[test]
fn deltanet_step_matches_hf() {
    let fx = Fx::load("deltanet");
    for c in fx.cases() {
        let heads = c["heads"].as_u64().unwrap() as usize;
        let dk = c["dk"].as_u64().unwrap() as usize;
        let dv = c["dv"].as_u64().unwrap() as usize;
        let (q, k, v) = (fx.arr(&c["q"]), fx.arr(&c["k"]), fx.arr(&c["v"]));
        let (a, b, big_a, dt) = (fx.arr(&c["a"]), fx.arr(&c["b"]), fx.arr(&c["ssm_a"]), fx.arr(&c["dt_bias"]));
        let (g_hf, beta_hf) = (fx.arr(&c["g"]), fx.arr(&c["beta"]));
        let mut state = fx.arr(&c["state"]);
        let want_out = fx.arr(&c["out"]);
        let want_state = fx.arr(&c["new_state"]);
        let max_abs = c["max_abs"].as_f64().unwrap();
        // gates
        let g: Vec<f32> = (0..heads).map(|h| gate_g(a[h], dt[h], big_a[h])).collect();
        let beta: Vec<f32> = (0..heads).map(|h| gate_beta(b[h])).collect();
        check(&format!("{} g", c["case"].as_str().unwrap()), &g, &g_hf, budget(fx.k(), 1, g_hf.iter().fold(0f32, |m, v| m.max(v.abs())) as f64));
        check(&format!("{} beta", c["case"].as_str().unwrap()), &beta, &beta_hf, budget(fx.k(), 1, 1.0));
        // step per head with the HF gates (so the step is isolated from the gate rounding)
        let mut out = vec![0f32; heads * dv];
        let scale = 1.0 / (dk as f32).sqrt();
        for h in 0..heads {
            let mut qn = q[h * dk..(h + 1) * dk].to_vec();
            let mut kn = k[h * dk..(h + 1) * dk].to_vec();
            l2norm(&mut qn, 1e-6);
            l2norm(&mut kn, 1e-6);
            for x in qn.iter_mut() {
                *x *= scale;
            }
            deltanet_step(&qn, &kn, &v[h * dv..(h + 1) * dv], g_hf[h], beta_hf[h], &mut state[h * dk * dv..(h + 1) * dk * dv], &mut out[h * dv..(h + 1) * dv]);
        }
        check(&format!("{} out", c["case"].as_str().unwrap()), &out, &want_out, budget(fx.k(), dk, max_abs));
        check(&format!("{} state", c["case"].as_str().unwrap()), &state, &want_state, budget(fx.k(), dk, max_abs));
    }
}
