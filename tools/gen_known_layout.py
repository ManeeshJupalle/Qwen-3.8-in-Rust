"""Emit crates/cli/src/known.rs: the layout of the one supported GGUF, so `aqueduct doctor` can compute the
memory plan for a machine before the 17.8 GB file is downloaded.

Source: tests/fixtures/gguf_index.json (every tensor's name, type, offset, size; captured in Phase 0 from
the file itself) and tests/fixtures/gguf_metadata.json (the qwen35.* hyperparameters). The emitted table is
what `aqueduct_core::tier::PlanInput::new` would build from the file; the unit test at the bottom of the
emitted file re-derives it from the fixture with a second implementation, and compares it with the real
file whenever that is present.

    python tools/gen_known_layout.py > crates/cli/src/known.rs
"""
import io
import json
import re
import sys

INDEX = "tests/fixtures/gguf_index.json"
META = "tests/fixtures/gguf_metadata.json"
# mirrors aqueduct_core::tier::RESIDENT_SMALL (the per-layer tensors kept as f32 copies)
RESIDENT_SMALL = [
    "attn_norm.weight",
    "post_attention_norm.weight",
    "ssm_a",
    "ssm_dt.bias",
    "ssm_conv1d.weight",
    "ssm_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.shared_head_norm.weight",
]
SHA256 = "e103abf9d914d1d7b2f2592f055f2759a71195c350a01c135f71aaae86bca52b"

TEST = r'''
#[cfg(test)]
mod tests {
    use super::*;
    use aqueduct_core::tier::RESIDENT_SMALL;
    use std::path::{Path, PathBuf};

    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
    }

    fn layer_of(name: &str) -> Option<usize> {
        let rest = name.strip_prefix("blk.")?;
        let end = rest.find('.')?;
        rest[..end].parse().ok()
    }

    /// A second derivation of the table from the fixture, in Rust: spans, small bytes, the non-layer
    /// region and the sizes must agree with what the generator wrote.
    #[test]
    fn table_matches_the_gguf_index_fixture() {
        let text = std::fs::read_to_string(root().join("tests/fixtures/gguf_index.json")).expect("tests/fixtures/gguf_index.json");
        let j: serde_json::Value = serde_json::from_str(&text).expect("json");
        assert_eq!(j["file"].as_str().unwrap(), FILE_NAME);
        assert_eq!(j["file_size"].as_u64().unwrap(), FILE_SIZE);
        let tensors = j["tensors"].as_array().unwrap();
        let n_blocks = LAYERS.len();
        let mut start = vec![u64::MAX; n_blocks];
        let mut end = vec![0u64; n_blocks];
        let mut small = vec![0u64; n_blocks];
        let mut non_layer: Vec<(String, u64)> = Vec::new();
        let (mut nl_start, mut nl_end, mut nl_sum) = (u64::MAX, 0u64, 0u64);
        for t in tensors {
            let name = t["name"].as_str().unwrap();
            let (off, size, fin) = (t["byte_offset"].as_u64().unwrap(), t["byte_size"].as_u64().unwrap(), t["byte_end"].as_u64().unwrap());
            match layer_of(name) {
                Some(l) => {
                    assert!(l < n_blocks, "{name}");
                    start[l] = start[l].min(off);
                    end[l] = end[l].max(fin);
                    if RESIDENT_SMALL.iter().any(|s| name.ends_with(s)) {
                        small[l] += size;
                    }
                }
                None => {
                    non_layer.push((name.to_string(), size));
                    nl_start = nl_start.min(off);
                    nl_end = nl_end.max(fin);
                    nl_sum += size;
                }
            }
        }
        for l in 0..n_blocks {
            assert_eq!(LAYERS[l], (start[l], end[l], small[l]), "layer {l}");
        }
        let input = plan_input();
        assert_eq!(input.non_layer, non_layer);
        assert_eq!(input.non_layer_region, if nl_end - nl_start <= nl_sum + nl_sum / 8 { Some((nl_start, nl_end)) } else { None });
        assert_eq!(input.n_layer as usize + input.mtp.len(), n_blocks);
        assert_eq!(input.layer_span.len(), N_LAYER as usize);
        // every layer is contiguous with the next: the spans tile the data region up to the end of the file
        for l in 1..n_blocks {
            assert_eq!(LAYERS[l].0, LAYERS[l - 1].1, "gap before layer {l}");
        }
        assert_eq!(LAYERS[n_blocks - 1].1, FILE_SIZE);
        assert_eq!(LAYERS[0].0, nl_end);
    }

    /// With the real file on this machine, the table must be exactly what the engine builds from it.
    #[test]
    fn table_matches_the_real_file_when_present() {
        let path = std::env::var_os("AQUEDUCT_GGUF").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(crate::run::DEFAULT_GGUF));
        if !path.exists() {
            if std::env::var_os("AQUEDUCT_REQUIRE_MODEL").is_some() {
                panic!("primary GGUF missing at {} and AQUEDUCT_REQUIRE_MODEL is set", path.display());
            }
            eprintln!("SKIPPED: primary GGUF missing at {} (set AQUEDUCT_GGUF)", path.display());
            return;
        }
        let g = aqueduct_core::Gguf::open(&path).expect("open");
        let cfg = aqueduct_core::ModelConfig::from_gguf(&g).expect("config");
        let from_file = PlanInput::new(&g, &cfg);
        assert_eq!(format!("{:?}", from_file), format!("{:?}", plan_input()));
        assert_eq!(std::fs::metadata(&path).unwrap().len(), FILE_SIZE);
    }
}
'''


def layer_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def main():
    d = json.load(io.open(INDEX, encoding="utf-8"))
    meta = json.load(io.open(META, encoding="utf-8"))
    kv = meta["kv"] if "kv" in meta else meta

    def k(key):
        return kv[key]["value"]

    tensors = d["tensors"]
    layers = {}
    non_layer = []
    for t in tensors:
        l = layer_of(t["name"])
        if l is None:
            non_layer.append(t)
        else:
            layers.setdefault(l, []).append(t)
    block_count = k("qwen35.block_count")
    nextn = k("qwen35.nextn_predict_layers")
    n_layer = block_count - nextn
    assert len(layers) == block_count, (len(layers), block_count)
    spans = {}
    small = {}
    for l, ts in layers.items():
        spans[l] = (min(t["byte_offset"] for t in ts), max(t["byte_end"] for t in ts))
        small[l] = sum(t["byte_size"] for t in ts if any(t["name"].endswith(s) for s in RESIDENT_SMALL))
    nl_start = non_layer[0]["byte_offset"]
    nl_end = max(t["byte_end"] for t in non_layer)
    nl_sum = sum(t["byte_size"] for t in non_layer)
    region = (nl_start, nl_end) if nl_end - nl_start <= nl_sum + nl_sum // 8 else None

    interval = k("qwen35.full_attention_interval")
    n_attention = sum(1 for i in range(n_layer) if (i + 1) % interval == 0)
    n_deltanet = n_layer - n_attention
    dn_n_k = k("qwen35.ssm.group_count")
    dn_n_v = k("qwen35.ssm.time_step_rank")
    dn_d = k("qwen35.ssm.state_size")
    dn_qkv_dim = 2 * dn_n_k * dn_d + dn_n_v * dn_d
    assert dn_n_v * dn_d == k("qwen35.ssm.inner_size")
    vocab = k("qwen35.vocab_size") if "qwen35.vocab_size" in kv else non_layer_vocab(non_layer)

    out = []
    w = out.append
    w("//! The one supported GGUF, as `aqueduct doctor` needs it before the file exists on this machine: its")
    w("//! name, size, sha256, download URL, and the layout the memory plan is computed from. GENERATED by")
    w("//! `tools/gen_known_layout.py` from `tests/fixtures/gguf_index.json` and `tests/fixtures/gguf_metadata.json`")
    w("//! (both captured from the file itself in Phase 0); the tests below check this table against the fixture")
    w("//! always and against the real file whenever it is present. Do not edit by hand.")
    w("")
    w("use aqueduct_core::tier::PlanInput;")
    w("")
    w('pub const FILE_NAME: &str = "%s";' % d["file"])
    w('pub const REPO: &str = "bartowski/Qwen3.8-27B-GGUF";')
    w('pub const URL: &str = "https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/%s";' % d["file"])
    w("pub const FILE_SIZE: u64 = %d;" % d["file_size"])
    w("/// sha256 of the file (Hugging Face LFS object id; verified against the local copy on 2026-09-06).")
    w('pub const SHA256: &str = "%s";' % SHA256)
    w("pub const N_LAYER: u32 = %d;" % n_layer)
    w("")
    w("/// Per layer `0..%d`: (span start, span end) in the file, and the resident small-tensor bytes." % block_count)
    w("const LAYERS: [(u64, u64, u64); %d] = [" % block_count)
    for l in range(block_count):
        s, e = spans[l]
        w("    (%d, %d, %d), // blk.%d" % (s, e, small[l], l))
    w("];")
    w("")
    w("/// Non-layer tensors in file order: (name, bytes).")
    w("const NON_LAYER: [(&str, u64); %d] = [" % len(non_layer))
    for t in non_layer:
        w('    ("%s", %d),' % (t["name"], t["byte_size"]))
    w("];")
    w("")
    w("/// What `PlanInput::new(&gguf, &config)` builds from the file, without the file.")
    w("pub fn plan_input() -> PlanInput {")
    w("    let layer_span: Vec<(u64, u64)> = LAYERS[..N_LAYER as usize].iter().map(|&(s, e, _)| (s, e)).collect();")
    w("    let layer_small: Vec<u64> = LAYERS[..N_LAYER as usize].iter().map(|&(_, _, b)| b).collect();")
    w("    let mtp: Vec<(u32, u64, u64)> = (N_LAYER as usize..LAYERS.len()).map(|i| (i as u32, LAYERS[i].1 - LAYERS[i].0, LAYERS[i].0)).collect();")
    w("    PlanInput {")
    w("        non_layer: NON_LAYER.iter().map(|&(n, b)| (n.to_string(), b)).collect(),")
    if region:
        w("        non_layer_region: Some((%d, %d))," % region)
    else:
        w("        non_layer_region: None,")
    w("        mtp,")
    w("        layer_span,")
    w("        layer_small,")
    w("        n_layer: N_LAYER,")
    w("        n_deltanet: %d," % n_deltanet)
    w("        n_attention: %d," % n_attention)
    w("        hidden: %d," % k("qwen35.embedding_length"))
    w("        inter: %d," % k("qwen35.feed_forward_length"))
    w("        vocab: %d," % vocab)
    w("        dn_conv_dim: %d," % dn_qkv_dim)
    w("        dn_n_v: %d," % dn_n_v)
    w("        dn_dk: %d," % dn_d)
    w("        dn_dv: %d," % dn_d)
    w("        dn_kernel: %d," % k("qwen35.ssm.conv_kernel"))
    w("        n_head: %d," % k("qwen35.attention.head_count"))
    w("        n_head_kv: %d," % k("qwen35.attention.head_count_kv"))
    w("        head_dim: %d," % k("qwen35.attention.key_length"))
    w("        rope_dim: %d," % k("qwen35.rope.dimension_count"))
    w("    }")
    w("}")
    sys.stdout.write("\n".join(out) + "\n" + TEST)


def non_layer_vocab(non_layer):
    for t in non_layer:
        if t["name"] == "token_embd.weight":
            return t["shape_numpy_order"][0]
    raise SystemExit("no token_embd.weight")


if __name__ == "__main__":
    main()
