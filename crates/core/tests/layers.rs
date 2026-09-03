//! 2a.2: layers against the HF submodule fixtures (tests/fixtures/layers/layers/manifest.json), prefill and
//! single-token step with carried state / KV cache, plus thread invariance. Budgets: k * sqrt(width) * eps *
//! max|reference| with k from the manifest and width = the layer's largest reduction length.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use aqueduct_core::kernels::matvec::WeightMat;
use aqueduct_core::layers::gated_deltanet::GatedDeltaNet;
use aqueduct_core::layers::gqa_attention::GqaAttention;
use aqueduct_core::layers::mlp::Mlp;
use aqueduct_core::layers::{DecoderLayer, LayerError, Mixer, MixerState, TensorSource};

const EPS: f64 = f32::EPSILON as f64;

struct FixtureSource {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl TensorSource for FixtureSource {
    fn mat(&self, name: &str, rows: usize, cols: usize) -> Result<WeightMat, LayerError> {
        let (shape, data) = self.tensors.get(name).ok_or_else(|| LayerError::Missing(name.into()))?;
        if shape != &vec![rows, cols] {
            return Err(LayerError::Shape { name: name.into(), expected: vec![rows, cols], found: shape.clone() });
        }
        Ok(WeightMat::from_f32(rows, cols, data))
    }
    fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, LayerError> {
        let (shape, data) = self.tensors.get(name).ok_or_else(|| LayerError::Missing(name.into()))?;
        if data.len() != len {
            return Err(LayerError::Shape { name: name.into(), expected: vec![len], found: shape.clone() });
        }
        Ok(data.clone())
    }
}

struct Fx {
    dir: PathBuf,
    manifest: serde_json::Value,
}

impl Fx {
    fn load() -> Fx {
        let dir = common::fixture("layers").join("layers");
        let manifest = common::json(&dir.join("manifest.json"));
        Fx { dir, manifest }
    }
    fn case(&self, name: &str) -> &serde_json::Value {
        self.manifest["cases"].as_array().unwrap().iter().find(|c| c["case"] == name).unwrap_or_else(|| panic!("case {name}"))
    }
    fn k(&self) -> f64 {
        self.manifest["budget_k"].as_f64().unwrap()
    }
    fn arr(&self, field: &serde_json::Value) -> Vec<f32> {
        common::read_npy_f32(&self.dir.join(field["file"].as_str().unwrap()))
    }
    fn source(&self, c: &serde_json::Value, prefix: &str) -> FixtureSource {
        let mut tensors = HashMap::new();
        for (name, f) in c["weights"].as_object().unwrap() {
            let shape: Vec<usize> = f["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            tensors.insert(format!("{prefix}{name}"), (shape, self.arr(f)));
        }
        FixtureSource { tensors }
    }
    fn cfg(&self, key: &str) -> usize {
        self.manifest["config"][key].as_u64().unwrap_or_else(|| panic!("config {key}")) as usize
    }
}

fn max_abs(v: &[f32]) -> f64 {
    v.iter().fold(0f64, |m, x| m.max(x.abs() as f64))
}

fn max_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}

fn check(label: &str, got: &[f32], want: &[f32], width: usize, k: f64) -> f64 {
    let bud = k * (width as f64).sqrt() * EPS * max_abs(want);
    let d = max_diff(got, want);
    assert!(d <= bud, "{label}: max abs diff {d:e} exceeds budget {bud:e} (k={k}, width={width}, max|ref|={:e})", max_abs(want));
    println!("{label}: max diff {d:.3e} vs budget {bud:.3e}");
    d
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn deltanet(fx: &Fx, c: &serde_json::Value, prefix: &str) -> GatedDeltaNet {
    let src = fx.source(c, prefix);
    GatedDeltaNet::load(&src, prefix, fx.cfg("hidden_size"), fx.cfg("linear_num_key_heads"), fx.cfg("linear_num_value_heads"),
        fx.cfg("linear_key_head_dim"), fx.cfg("linear_value_head_dim"), fx.cfg("linear_conv_kernel_dim"), fx.manifest["config"]["rms_norm_eps"].as_f64().unwrap() as f32)
        .expect("load deltanet")
}

fn attention(fx: &Fx, c: &serde_json::Value, prefix: &str) -> GqaAttention {
    let src = fx.source(c, prefix);
    let hd = fx.cfg("head_dim");
    let rope_dim = (hd as f64 * fx.manifest["config"]["rope_parameters"]["partial_rotary_factor"].as_f64().unwrap()) as usize;
    GqaAttention::load(&src, prefix, fx.cfg("hidden_size"), fx.cfg("num_attention_heads"), fx.cfg("num_key_value_heads"), hd, rope_dim,
        fx.manifest["config"]["rope_parameters"]["rope_theta"].as_f64().unwrap() as f32, fx.manifest["config"]["rms_norm_eps"].as_f64().unwrap() as f32)
        .expect("load attention")
}

#[test]
fn gated_deltanet_prefill_and_step() {
    let fx = Fx::load();
    let c = fx.case("deltanet_layer");
    let layer = deltanet(&fx, c, "");
    let hidden = fx.cfg("hidden_size");
    let t = c["t"].as_u64().unwrap() as usize;
    let x = fx.arr(&c["prefill"]["x"]);
    let width = hidden.max(layer.dk * layer.dv);
    for threads in [1usize, 3] {
        let mut state = layer.new_state();
        let mut y = vec![0f32; t * hidden];
        layer.forward_prefill(&x, t, &mut state, &mut y, threads);
        check(&format!("deltanet prefill vs HF chunk kernel (threads {threads})"), &y, &fx.arr(&c["prefill"]["y"]), width, fx.k());
        check("deltanet prefill vs HF sequential recurrence", &y, &fx.arr(&c["prefill"]["y_seq"]), width, fx.k());
        check("deltanet prefill recurrent state", &state.rec, &fx.arr(&c["prefill"]["rec_state"]), width, fx.k());
        assert_eq!(max_diff(&state.conv, &fx.arr(&c["prefill"]["conv_state"])), 0.0, "conv window must be bit-exact (a shift of the inputs)");
        let mut y1 = vec![0f32; hidden];
        layer.forward_token(&fx.arr(&c["step"]["x"]), &mut state, &mut y1, threads);
        check(&format!("deltanet step with carried state (threads {threads})"), &y1, &fx.arr(&c["step"]["y"]), width, fx.k());
        check("deltanet step recurrent state", &state.rec, &fx.arr(&c["step"]["rec_state"]), width, fx.k());
        assert_eq!(max_diff(&state.conv, &fx.arr(&c["step"]["conv_state"])), 0.0);
        if threads == 1 {
            // thread invariance: repeat with 3 threads below and compare bits
            let mut s3 = layer.new_state();
            let mut y3 = vec![0f32; t * hidden];
            layer.forward_prefill(&x, t, &mut s3, &mut y3, 3);
            assert_eq!(bits(&y), bits(&y3), "deltanet: 3 threads differ from 1");
            assert_eq!(bits(&s3.rec), bits(&fx_state_after_prefill(&layer, &x, t)), "deltanet state: thread invariance");
        }
    }
}

fn fx_state_after_prefill(layer: &GatedDeltaNet, x: &[f32], t: usize) -> Vec<f32> {
    let mut s = layer.new_state();
    let mut y = vec![0f32; t * layer.hidden];
    layer.forward_prefill(x, t, &mut s, &mut y, 1);
    s.rec
}

#[test]
fn gqa_attention_prefill_and_step() {
    let fx = Fx::load();
    let c = fx.case("attention_layer");
    let layer = attention(&fx, c, "");
    let hidden = fx.cfg("hidden_size");
    let t = c["t"].as_u64().unwrap() as usize;
    let x = fx.arr(&c["prefill"]["x"]);
    let width = hidden.max(layer.head_dim).max(t);
    let mut out1 = None;
    for threads in [1usize, 4] {
        let mut cache = layer.new_cache();
        let mut y = vec![0f32; t * hidden];
        layer.forward_prefill(&x, t, 0, &mut cache, &mut y, threads);
        check(&format!("attention prefill (threads {threads})"), &y, &fx.arr(&c["prefill"]["y"]), width, fx.k());
        // KV cache: HF stores (n_kv, T, head_dim); ours is [t][kvh][d]
        let (nkv, hd) = (layer.n_head_kv, layer.head_dim);
        let k_hf = fx.arr(&c["prefill"]["k_cache"]);
        let v_hf = fx.arr(&c["prefill"]["v_cache"]);
        let mut k_ours = vec![0f32; nkv * t * hd];
        let mut v_ours = vec![0f32; nkv * t * hd];
        for h in 0..nkv {
            for ti in 0..t {
                for d in 0..hd {
                    k_ours[(h * t + ti) * hd + d] = cache.k[ti * cache.stride + h * hd + d];
                    v_ours[(h * t + ti) * hd + d] = cache.v[ti * cache.stride + h * hd + d];
                }
            }
        }
        check("attention K cache (post-RoPE)", &k_ours, &k_hf, hidden, fx.k());
        check("attention V cache", &v_ours, &v_hf, hidden, fx.k());
        let mut y1 = vec![0f32; hidden];
        layer.forward_token(&fx.arr(&c["step"]["x"]), c["step"]["position"].as_u64().unwrap() as u32, &mut cache, &mut y1, threads);
        check(&format!("attention step with KV cache (threads {threads})"), &y1, &fx.arr(&c["step"]["y"]), width, fx.k());
        match &out1 {
            None => out1 = Some((bits(&y), bits(&y1))),
            Some((a, b)) => {
                assert_eq!(a, &bits(&y), "attention prefill: thread invariance");
                assert_eq!(b, &bits(&y1), "attention step: thread invariance");
            }
        }
    }
}

#[test]
fn mlp_matches_hf() {
    let fx = Fx::load();
    let c = fx.case("mlp_layer");
    let src = fx.source(c, "");
    let hidden = fx.cfg("hidden_size");
    let inter = fx.cfg("intermediate_size");
    let mlp = Mlp::load(&src, "", hidden, inter).expect("load mlp");
    let t = c["t"].as_u64().unwrap() as usize;
    let x = fx.arr(&c["prefill"]["x"]);
    let mut y = vec![0f32; t * hidden];
    for i in 0..t {
        mlp.forward(&x[i * hidden..(i + 1) * hidden], &mut y[i * hidden..(i + 1) * hidden], 2);
    }
    check("mlp", &y, &fx.arr(&c["prefill"]["y"]), inter, fx.k());
    let mut y1 = vec![0f32; t * hidden];
    for i in 0..t {
        mlp.forward(&x[i * hidden..(i + 1) * hidden], &mut y1[i * hidden..(i + 1) * hidden], 1);
    }
    assert_eq!(bits(&y), bits(&y1), "mlp: thread invariance");
}

#[test]
fn decoder_layers_prefill_and_step() {
    let fx = Fx::load();
    let hidden = fx.cfg("hidden_size");
    let inter = fx.cfg("intermediate_size");
    let eps = fx.manifest["config"]["rms_norm_eps"].as_f64().unwrap() as f32;
    for kind in ["linear_attention", "full_attention"] {
        let c = fx.case(&format!("decoder_{kind}"));
        let src = fx.source(c, "");
        let mixer = if kind == "linear_attention" { Mixer::DeltaNet(deltanet(&fx, c, "")) } else { Mixer::Attention(attention(&fx, c, "")) };
        let layer = DecoderLayer {
            index: c["layer_idx"].as_u64().unwrap() as u32,
            attn_norm: src.vec("attn_norm.weight", hidden).unwrap(),
            post_attention_norm: src.vec("post_attention_norm.weight", hidden).unwrap(),
            mixer,
            mlp: Mlp::load(&src, "", hidden, inter).unwrap(),
            eps,
        };
        let t = c["t"].as_u64().unwrap() as usize;
        let x = fx.arr(&c["prefill"]["x"]);
        let width = hidden.max(inter);
        let mut prev: Option<Vec<u32>> = None;
        for threads in [1usize, 5] {
            let mut state = layer.new_state();
            let mut y = vec![0f32; t * hidden];
            for i in 0..t {
                let (xi, yi) = (&x[i * hidden..(i + 1) * hidden], &mut y[i * hidden..(i + 1) * hidden]);
                layer.forward_token(xi, i as u32, &mut state, yi, threads);
            }
            check(&format!("decoder {kind} prefill (threads {threads})"), &y, &fx.arr(&c["prefill"]["y"]), width, fx.k());
            let mut y1 = vec![0f32; hidden];
            layer.forward_token(&fx.arr(&c["step"]["x"]), c["step"]["position"].as_u64().unwrap() as u32, &mut state, &mut y1, threads);
            check(&format!("decoder {kind} step (threads {threads})"), &y1, &fx.arr(&c["step"]["y"]), width, fx.k());
            let b = bits(&y);
            if let Some(p) = &prev {
                assert_eq!(p, &b, "decoder {kind}: thread invariance");
            }
            prev = Some(b);
            let _ = MixerState::DeltaNet;
        }
    }
}
