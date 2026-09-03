//! 2a.3: the tiny oracle. A test-only, naive, sequential forward over models/tiny/tiny-f32.gguf (built by
//! tools/make_tiny_checkpoint.py through llama.cpp's converter) using only the Phase 2a layers and kernels,
//! against tests/fixtures/tiny/tiny/manifest.json (tools/tiny_reference.py): per-layer hidden states at every
//! prompt position, last-position logits, and 20 greedy tokens on 3 prompts. Budgets are derived from the
//! fixture's max|reference| with k = budget_k_per_layer * (L + 1). This is not the engine.

mod common;

use aqueduct_core::kernels::matvec::{matvec, Act, WeightMat};
use aqueduct_core::kernels::rmsnorm::rmsnorm;
use aqueduct_core::layers::gated_deltanet::GatedDeltaNet;
use aqueduct_core::layers::gqa_attention::GqaAttention;
use aqueduct_core::layers::mlp::Mlp;
use aqueduct_core::layers::{DecoderLayer, Mixer, MixerState, TensorSource};
use aqueduct_core::{arch_consts, Gguf, ModelConfig};

const EPS: f64 = f32::EPSILON as f64;

struct Oracle {
    cfg: ModelConfig,
    embed: Vec<f32>,
    layers: Vec<DecoderLayer>,
    output_norm: Vec<f32>,
    lm_head: WeightMat,
}

impl Oracle {
    fn load(g: &Gguf) -> Oracle {
        let cfg = ModelConfig::from_gguf(g).expect("config from the tiny GGUF");
        arch_consts(&cfg.architecture).expect("known architecture");
        let hidden = cfg.hidden_size as usize;
        let vocab = cfg.vocab_size as usize;
        let src: &dyn TensorSource = g;
        let embed = src.vec("token_embd.weight", vocab * hidden).unwrap();
        let mut layers = Vec::new();
        for i in 0..cfg.n_layer {
            let p = format!("blk.{i}.");
            let mixer = if cfg.full_attention_layers.contains(&i) {
                Mixer::Attention(
                    GqaAttention::load(src, &p, hidden, cfg.n_head as usize, cfg.n_head_kv as usize, cfg.head_dim_k as usize, cfg.rope_dim as usize, cfg.rope_freq_base, cfg.rms_norm_eps).unwrap(),
                )
            } else {
                Mixer::DeltaNet(
                    GatedDeltaNet::load(src, &p, hidden, cfg.dn_n_k_heads as usize, cfg.dn_n_v_heads as usize, cfg.dn_head_dim_k as usize, cfg.dn_head_dim_v as usize, cfg.dn_conv_kernel as usize, cfg.rms_norm_eps).unwrap(),
                )
            };
            layers.push(DecoderLayer {
                index: i,
                attn_norm: src.vec(&format!("{p}attn_norm.weight"), hidden).unwrap(),
                post_attention_norm: src.vec(&format!("{p}post_attention_norm.weight"), hidden).unwrap(),
                mixer,
                mlp: Mlp::load(src, &p, hidden, cfg.intermediate_size as usize).unwrap(),
                eps: cfg.rms_norm_eps,
            });
        }
        let output_norm = src.vec("output_norm.weight", hidden).unwrap();
        let lm_head = src.mat("output.weight", vocab, hidden).unwrap();
        Oracle { cfg, embed, layers, output_norm, lm_head }
    }

    fn new_states(&self) -> Vec<MixerState> {
        self.layers.iter().map(|l| l.new_state()).collect()
    }

    /// One token: returns the hidden state after every layer (n_layer entries) and the logits.
    fn step(&self, token: u32, pos: u32, states: &mut [MixerState], threads: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
        let hidden = self.cfg.hidden_size as usize;
        let mut h = self.embed[token as usize * hidden..(token as usize + 1) * hidden].to_vec();
        let mut per_layer = Vec::with_capacity(self.layers.len());
        for (layer, state) in self.layers.iter().zip(states.iter_mut()) {
            let mut y = vec![0f32; hidden];
            layer.forward_token(&h, pos, state, &mut y, threads);
            per_layer.push(y.clone());
            h = y;
        }
        let mut normed = vec![0f32; hidden];
        rmsnorm(&h, &self.output_norm, self.cfg.rms_norm_eps, &mut normed);
        let mut logits = vec![0f32; self.cfg.vocab_size as usize];
        matvec(&self.lm_head, Act::F32(&normed), &mut logits, threads);
        (per_layer, logits)
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

fn max_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}

fn tiny_gguf() -> std::path::PathBuf {
    let p = common::root().join("models").join("tiny").join("tiny-f32.gguf");
    assert!(p.exists(), "tiny GGUF missing at {}: run `python tools/make_tiny_checkpoint.py`", p.display());
    p
}

#[test]
fn tiny_oracle_matches_hf_reference() {
    let g = Gguf::open(tiny_gguf()).expect("open tiny gguf");
    let oracle = Oracle::load(&g);
    let hidden = oracle.cfg.hidden_size as usize;
    let dir = common::fixture("tiny").join("tiny");
    let m = common::json(&dir.join("manifest.json"));
    let k = m["budget_k_per_layer"].as_f64().unwrap();
    let n_greedy = m["greedy_tokens"].as_u64().unwrap() as usize;
    let arr = |f: &serde_json::Value| common::read_npy_f32(&dir.join(f["file"].as_str().unwrap()));
    let threads = 4;

    let mut total_greedy = 0;
    let mut matched_greedy = 0;
    let mut worst_layer_ratio = 0f64;
    let mut worst_layer_desc = String::new();
    for c in m["cases"].as_array().unwrap() {
        let name = c["case"].as_str().unwrap();
        let ids: Vec<u32> = c["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let t = ids.len();
        let mut states = oracle.new_states();
        // prompt: collect per-layer hidden at every position
        let mut per_layer_all: Vec<Vec<f32>> = vec![Vec::with_capacity(t * hidden); oracle.layers.len()];
        let mut last_logits = Vec::new();
        for (pos, &tok) in ids.iter().enumerate() {
            let (pl, logits) = oracle.step(tok, pos as u32, &mut states, threads);
            for (l, h) in pl.into_iter().enumerate() {
                per_layer_all[l].extend_from_slice(&h);
            }
            let want_arg = c["argmax_per_position"][pos].as_u64().unwrap() as u32;
            let got_arg = argmax(&logits);
            if got_arg != want_arg {
                println!("{name}: argmax at prompt position {pos} differs: got {got_arg} want {want_arg}");
            }
            last_logits = logits;
        }
        for l in 0..oracle.layers.len() {
            let want = arr(&c["hidden"][format!("layer_{l}")]);
            let max_abs = c["hidden"][format!("layer_{l}")]["max_abs"].as_f64().unwrap();
            let bud = k * (l as f64 + 1.0) * (hidden as f64).sqrt() * EPS * max_abs;
            let d = max_diff(&per_layer_all[l], &want);
            assert!(d <= bud, "{name}: hidden state after layer {l}: max abs diff {d:e} exceeds budget {bud:e}");
            if d / bud > worst_layer_ratio {
                worst_layer_ratio = d / bud;
                worst_layer_desc = format!("{name} layer {l}: diff {d:.3e} vs budget {bud:.3e}");
            }
        }
        let want_logits = arr(&c["logits_last"]);
        let lbud = k * (oracle.layers.len() as f64 + 2.0) * (hidden as f64).sqrt() * EPS * c["logits_max_abs"].as_f64().unwrap();
        let ld = max_diff(&last_logits, &want_logits);
        println!("{name}: logits max diff {ld:.3e} vs budget {lbud:.3e}; argmax {} want {} (top-2 margin {:.4})", argmax(&last_logits), c["top5_last"][0], c["margin_last"].as_f64().unwrap());
        assert!(ld <= lbud, "{name}: logits max abs diff {ld:e} exceeds budget {lbud:e}");
        // greedy continuation
        let want_greedy: Vec<u32> = c["greedy_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let mut got_greedy = Vec::with_capacity(n_greedy);
        let mut cur = last_logits;
        for step in 0..n_greedy {
            let nxt = argmax(&cur);
            got_greedy.push(nxt);
            let (_, logits) = oracle.step(nxt, (t + step) as u32, &mut states, threads);
            cur = logits;
        }
        let ok = got_greedy.iter().zip(&want_greedy).filter(|(a, b)| a == b).count();
        println!("{name}: greedy {ok}/{n_greedy} match; got {got_greedy:?}");
        total_greedy += n_greedy;
        matched_greedy += ok;
        assert_eq!(got_greedy, want_greedy, "{name}: greedy tokens differ");
    }
    println!("greedy {matched_greedy}/{total_greedy}; worst layer: {worst_layer_desc} ({worst_layer_ratio:.2} of budget)");
    assert_eq!(matched_greedy, total_greedy);
}

#[test]
fn tiny_oracle_is_thread_invariant() {
    let g = Gguf::open(tiny_gguf()).expect("open tiny gguf");
    let oracle = Oracle::load(&g);
    let dir = common::fixture("tiny").join("tiny");
    let m = common::json(&dir.join("manifest.json"));
    let c = &m["cases"][1];
    let ids: Vec<u32> = c["prompt_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let mut outs = Vec::new();
    for threads in [1usize, 6] {
        let mut states = oracle.new_states();
        let mut all = Vec::new();
        for (pos, &tok) in ids.iter().enumerate() {
            let (pl, logits) = oracle.step(tok, pos as u32, &mut states, threads);
            for h in pl {
                all.extend(h.iter().map(|v| v.to_bits()));
            }
            all.extend(logits.iter().map(|v| v.to_bits()));
        }
        outs.push(all);
    }
    assert_eq!(outs[0], outs[1], "tiny oracle: 6 threads differ from 1 thread");
}
