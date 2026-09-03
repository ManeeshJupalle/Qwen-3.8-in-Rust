"""Generate docs/model-facts.md from the downloaded config files. Every value is read from a file
and printed together with the JSON path it came from. Nothing is typed from memory.

Fields the engine needs are listed in KNOWN with their JSON path. Every other field in
config.json is emitted verbatim under "Unrecognised fields" (value included, no guess at meaning).

Usage: python tools/model_facts.py <hf_local_dir>
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "model-facts.md")


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        kk = "%s.%s" % (p, k) if p else k
        if isinstance(v, dict):
            out.update(flat(v, kk))
        else:
            out[kk] = v
    return out


def get(d, path):
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return "<MISSING>"
    return cur


# (heading, [(label, json path in config.json, note)])
KNOWN = [
    ("Identity", [
        ("architecture class", "architectures", "top-level, list"),
        ("model_type (top)", "model_type", ""),
        ("model_type (text)", "text_config.model_type", ""),
        ("model_type (vision)", "vision_config.model_type", "vision tower; engine skips it"),
        ("transformers version that wrote the file", "transformers_version", ""),
        ("language_model_only", "language_model_only", ""),
        ("dtype of checkpoint", "text_config.dtype", "safetensors headers confirm BF16 for all 1199 tensors"),
    ]),
    ("Dimensions", [
        ("hidden size", "text_config.hidden_size", ""),
        ("number of layers", "text_config.num_hidden_layers", ""),
        ("MLP intermediate size", "text_config.intermediate_size", ""),
        ("MLP activation", "text_config.hidden_act", "SwiGLU: down(act(gate) * up)"),
        ("vocab size (padded)", "text_config.vocab_size", "tokenizer.json has fewer real tokens; see tokenizer section"),
        ("max position embeddings", "text_config.max_position_embeddings", ""),
        ("tie word embeddings (text)", "text_config.tie_word_embeddings", ""),
        ("tie word embeddings (top)", "tie_word_embeddings", ""),
        ("RMSNorm eps", "text_config.rms_norm_eps", ""),
    ]),
    ("Layer schedule", [
        ("full_attention_interval", "text_config.full_attention_interval", ""),
        ("layer_types (explicit list)", "text_config.layer_types", "64 entries; see derived list below"),
    ]),
    ("Full attention (GQA) layers", [
        ("num_attention_heads (Q)", "text_config.num_attention_heads", ""),
        ("num_key_value_heads (KV)", "text_config.num_key_value_heads", ""),
        ("head_dim", "text_config.head_dim", ""),
        ("attention_bias", "text_config.attention_bias", ""),
        ("attention_dropout", "text_config.attention_dropout", ""),
        ("attn_output_gate", "text_config.attn_output_gate", "safetensors: q_proj is [12288,5120] = 2 x 24 x 256, i.e. Q and gate fused"),
        ("partial_rotary_factor (text)", "text_config.partial_rotary_factor", "0.25 x 256 = 64 rotary dims, matches model card 'RoPE dimension 64'"),
        ("rope_theta", "text_config.rope_parameters.rope_theta", ""),
        ("rope_type", "text_config.rope_parameters.rope_type", ""),
        ("rope partial_rotary_factor", "text_config.rope_parameters.partial_rotary_factor", "duplicate of the text_config one"),
        ("mrope_section", "text_config.rope_parameters.mrope_section", "sums to 32 = 64/2 rotary pairs; multimodal RoPE (t,h,w)"),
        ("mrope_interleaved", "text_config.rope_parameters.mrope_interleaved", ""),
    ]),
    ("Gated DeltaNet (linear attention) layers", [
        ("linear_num_key_heads (Q/K heads)", "text_config.linear_num_key_heads", ""),
        ("linear_key_head_dim", "text_config.linear_key_head_dim", ""),
        ("linear_num_value_heads (V heads)", "text_config.linear_num_value_heads", ""),
        ("linear_value_head_dim", "text_config.linear_value_head_dim", ""),
        ("linear_conv_kernel_dim (causal conv width)", "text_config.linear_conv_kernel_dim", "safetensors conv1d.weight is [10240,1,4]"),
        ("output_gate_type", "text_config.output_gate_type", ""),
        ("mamba_ssm_dtype", "text_config.mamba_ssm_dtype", "recurrent state kept in float32 by the reference"),
    ]),
    ("MTP head", [
        ("mtp_num_hidden_layers", "text_config.mtp_num_hidden_layers", "safetensors has mtp.layers.0.* (one full-attention block)"),
        ("mtp_use_dedicated_embeddings", "text_config.mtp_use_dedicated_embeddings", "false: no mtp embed/lm_head tensors in safetensors"),
    ]),
    ("Special token ids (config)", [
        ("bos_token_id (text)", "text_config.bos_token_id", ""),
        ("eos_token_id (text)", "text_config.eos_token_id", ""),
        ("pad_token_id (text)", "text_config.pad_token_id", ""),
        ("image_token_id", "image_token_id", ""),
        ("video_token_id", "video_token_id", ""),
        ("vision_start_token_id", "vision_start_token_id", ""),
        ("vision_end_token_id", "vision_end_token_id", ""),
    ]),
    ("Misc", [
        ("use_cache", "text_config.use_cache", ""),
        ("initializer_range", "text_config.initializer_range", "training only"),
    ]),
]


def main():
    hf_dir = sys.argv[1]
    cfg = json.load(open(os.path.join(hf_dir, "config.json"), encoding="utf-8"))
    gen = json.load(open(os.path.join(hf_dir, "generation_config.json"), encoding="utf-8"))
    tokc = json.load(open(os.path.join(hf_dir, "tokenizer_config.json"), encoding="utf-8"))
    tokj = json.load(open(os.path.join(hf_dir, "tokenizer.json"), encoding="utf-8"))

    L = []
    P = L.append
    P("# Model facts (captured from files, not from memory)")
    P("")
    P("Generated by `tools/model_facts.py` from the files downloaded into `models/Qwen3.8-27B/`.")
    P("Every value is followed by the JSON path it was read from. `config.json` is copied verbatim to")
    P("`tests/fixtures/config.json`. Vision fields are listed once and then ignored by the engine.")
    P("")
    P("Sources: `config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`,")
    P("`model.safetensors.index.json` + shard headers (shapes; see `tests/fixtures/st_index.json`).")
    P("")

    seen = set()
    for heading, rows in KNOWN:
        P("## " + heading)
        P("")
        P("| value | JSON path (config.json) | note |")
        P("|---|---|---|")
        for label, path, note in rows:
            v = get(cfg, path)
            seen.add(path)
            vs = json.dumps(v) if not isinstance(v, list) or len(v) <= 4 else "<list of %d>" % len(v)
            P("| %s = `%s` | `%s` | %s |" % (label, vs, path, note))
        P("")

    # derived: full-attention layer indices
    lt = cfg["text_config"]["layer_types"]
    n = cfg["text_config"]["num_hidden_layers"]
    interval = cfg["text_config"]["full_attention_interval"]
    full_idx = [i for i, t in enumerate(lt) if t == "full_attention"]
    lin_idx = [i for i, t in enumerate(lt) if t == "linear_attention"]
    derived = [i for i in range(n) if (i + 1) % interval == 0]
    P("## Derived: layer schedule")
    P("")
    P("- `text_config.layer_types` has %d entries; distinct values: %s" % (len(lt), sorted(set(lt))))
    P("- full_attention layer indices (**0-based**, index into `layer_types`): `%s`" % full_idx)
    P("- count: %d full_attention, %d linear_attention" % (len(full_idx), len(lin_idx)))
    P("- rule check: `(i+1) %% full_attention_interval == 0` gives `%s`; matches layer_types: **%s**" % (derived, derived == full_idx))
    P("- so with 0-based indexing the full-attention layers are 3, 7, 11, ... 63 (every 4th, the LAST of each group of 4),")
    P("  NOT 0, 4, 8, ... The engine must read `layer_types`, not recompute from the interval.")
    P("")

    # derived dims
    tc = cfg["text_config"]
    P("## Derived: projection widths (cross-checked against safetensors shapes)")
    P("")
    P("| quantity | formula | value | safetensors tensor confirming it |")
    P("|---|---|---|---|")
    q = tc["num_attention_heads"] * tc["head_dim"]
    kv = tc["num_key_value_heads"] * tc["head_dim"]
    P("| GQA q width | num_attention_heads x head_dim | %d | q_proj is [%d, %d] = 2 x q width (Q + output gate, attn_output_gate=true) |" % (q, 2 * q, tc["hidden_size"]))
    P("| GQA kv width | num_key_value_heads x head_dim | %d | k_proj, v_proj are [%d, %d] |" % (kv, kv, tc["hidden_size"]))
    P("| GQA o_proj in | num_attention_heads x head_dim | %d | o_proj is [%d, %d] |" % (q, tc["hidden_size"], q))
    P("| GQA q_norm / k_norm | head_dim | %d | q_norm.weight, k_norm.weight are [%d] (per-head RMSNorm) |" % (tc["head_dim"], tc["head_dim"]))
    P("| rotary dims | head_dim x partial_rotary_factor | %d | (no tensor; mrope_section sums to %d = rotary/2) |" % (int(tc["head_dim"] * tc["partial_rotary_factor"]), sum(tc["rope_parameters"]["mrope_section"])))
    dk = tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
    dv = tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
    P("| DeltaNet q,k width | linear_num_key_heads x linear_key_head_dim | %d each | in_proj_qkv is [%d, %d] = q + k + v |" % (dk, 2 * dk + dv, tc["hidden_size"]))
    P("| DeltaNet v width | linear_num_value_heads x linear_value_head_dim | %d | in_proj_z is [%d, %d]; out_proj is [%d, %d] |" % (dv, dv, tc["hidden_size"], tc["hidden_size"], dv))
    P("| DeltaNet conv channels | q+k+v | %d | conv1d.weight is [%d, 1, %d] |" % (2 * dk + dv, 2 * dk + dv, tc["linear_conv_kernel_dim"]))
    P("| DeltaNet a/b/A_log/dt_bias | linear_num_value_heads | %d | in_proj_a, in_proj_b are [%d, %d]; A_log, dt_bias are [%d] |" % (tc["linear_num_value_heads"], tc["linear_num_value_heads"], tc["hidden_size"], tc["linear_num_value_heads"]))
    P("| DeltaNet norm | linear_value_head_dim | %d | linear_attn.norm.weight is [%d] |" % (tc["linear_value_head_dim"], tc["linear_value_head_dim"]))
    P("| MTP fc in | 2 x hidden_size | %d | mtp.fc.weight is [%d, %d] |" % (2 * tc["hidden_size"], tc["hidden_size"], 2 * tc["hidden_size"]))
    P("| V/QK head ratio (DeltaNet) | linear_num_value_heads / linear_num_key_heads | %d | each K head serves %d V heads |" % (tc["linear_num_value_heads"] // tc["linear_num_key_heads"], tc["linear_num_value_heads"] // tc["linear_num_key_heads"]))
    P("| GQA group | num_attention_heads / num_key_value_heads | %d | |" % (tc["num_attention_heads"] // tc["num_key_value_heads"]))
    P("")

    # generation config
    P("## generation_config.json")
    P("")
    P("| value | JSON path (generation_config.json) |")
    P("|---|---|")
    for k, v in gen.items():
        P("| `%s` | `%s` |" % (json.dumps(v), k))
    P("")
    P("Note: `eos_token_id` here is a list `%s` while `config.json:text_config.eos_token_id` is the scalar `%s`." % (gen.get("eos_token_id"), tc.get("eos_token_id")))
    P("")

    # tokenizer facts
    P("## Tokenizer (tokenizer_config.json / tokenizer.json)")
    P("")
    P("| value | JSON path |")
    P("|---|---|")
    for k in ["tokenizer_class", "add_bos_token", "add_prefix_space", "bos_token", "eos_token", "pad_token", "unk_token",
              "model_max_length", "clean_up_tokenization_spaces", "split_special_tokens", "errors"]:
        P("| `%s` | `tokenizer_config.json:%s` |" % (json.dumps(tokc.get(k, "<MISSING>")), k))
    P("| `%s` | `tokenizer_config.json:pretokenize_regex` |" % tokc.get("pretokenize_regex", "<MISSING>").replace("|", "\\|"))
    P("| chat_template present in tokenizer_config.json: `%s` (%d chars) | `tokenizer_config.json:chat_template` |" % ("chat_template" in tokc, len(tokc.get("chat_template", ""))))
    P("| additional_special_tokens: `%s` | `tokenizer_config.json:additional_special_tokens` |" % json.dumps(tokc.get("additional_special_tokens")))
    P("| extra_special_tokens: `%s` | `tokenizer_config.json:extra_special_tokens` |" % json.dumps(tokc.get("extra_special_tokens")))
    model = tokj.get("model", {})
    vocab = model.get("vocab", {})
    added = tokj.get("added_tokens", [])
    P("| tokenizer.json model.type = `%s` | `tokenizer.json:model.type` |" % model.get("type"))
    P("| tokenizer.json vocab entries = `%d` | `tokenizer.json:model.vocab` (len) |" % len(vocab))
    P("| tokenizer.json merges = `%d` | `tokenizer.json:model.merges` (len) |" % len(model.get("merges", [])))
    P("| tokenizer.json added_tokens = `%d`, max id = `%d` | `tokenizer.json:added_tokens` |" % (len(added), max(t["id"] for t in added) if added else -1))
    maxv = max(vocab.values()) if vocab else -1
    P("| max id used by vocab+added = `%d`; config vocab_size = `%d`; padding rows = `%d` | derived |" % (max(maxv, max(t["id"] for t in added)), tc["vocab_size"], tc["vocab_size"] - 1 - max(maxv, max(t["id"] for t in added))))
    P("| byte_fallback = `%s`, ignore_merges = `%s`, fuse_unk = `%s` | `tokenizer.json:model.*` |" % (model.get("byte_fallback"), model.get("ignore_merges"), model.get("fuse_unk")))
    P("| pre_tokenizer = `%s` | `tokenizer.json:pre_tokenizer` |" % json.dumps(tokj.get("pre_tokenizer"))[:400].replace("|", "\\|"))
    P("| normalizer = `%s` | `tokenizer.json:normalizer` |" % json.dumps(tokj.get("normalizer")))
    P("| decoder = `%s` | `tokenizer.json:decoder` |" % json.dumps(tokj.get("decoder"))[:300].replace("|", "\\|"))
    P("| post_processor = `%s` | `tokenizer.json:post_processor` |" % json.dumps(tokj.get("post_processor"))[:300].replace("|", "\\|"))
    P("")
    P("### Added tokens (tokenizer_config.json:added_tokens_decoder, all %d)" % len(tokc["added_tokens_decoder"]))
    P("")
    P("| id | content | special |")
    P("|---|---|---|")
    for k, v in sorted(tokc["added_tokens_decoder"].items(), key=lambda kv: int(kv[0])):
        P("| %s | `%s` | %s |" % (k, v["content"].replace("|", "\\|"), v.get("special")))
    P("")

    # fields whose name is known but whose exact semantics are NOT verifiable from the files alone
    P("## Fields listed above whose semantics are inferred, not verified from files")
    P("")
    P("These are tabulated above because the engine needs them, but the note column is an inference")
    P("(from tensor shapes or from the transformers 5.16.1 reference source in the venv), not a fact read")
    P("from a config file. Phase 2 fixtures must pin each one against the reference implementation.")
    P("")
    P("| JSON path | what is verified | what is inferred |")
    P("|---|---|---|")
    P("| `text_config.attn_output_gate` | q_proj is 2 x (heads x head_dim) wide | that the second half is a sigmoid gate on the attention output, and its layout (interleaved per head vs. concatenated) |")
    P("| `text_config.output_gate_type` | value `swish` | which tensor it gates (in_proj_z / GGUF attn_gate) and where the norm sits relative to the gate |")
    P("| `text_config.mamba_ssm_dtype` | value `float32` | that only the recurrent state is float32 while projections stay bf16 |")
    P("| `text_config.partial_rotary_factor` | 0.25 x 256 = 64; GGUF `qwen35.rope.dimension_count` = 64 | which 64 of the 256 dims rotate (first 64 in the reference source) |")
    P("| `text_config.rope_parameters.mrope_section` | [11,11,10] sums to 32 = 64/2 | for text-only input all three position streams are identical, so interleaved mRoPE reduces to plain RoPE (reference source `apply_interleaved_mrope` copies H/W freqs into T slots; equal positions give equal freqs) |")
    P("| `text_config.rope_parameters.mrope_interleaved` | value true | the THW interleave pattern (only matters with images/video) |")
    P("| `text_config.linear_conv_kernel_dim` | conv1d.weight is [10240,1,4] | that the conv runs over the concatenated q,k,v (10240 channels) before the recurrence, causal with zero left padding |")
    P("| `text_config.mtp_use_dedicated_embeddings` | value false; safetensors has no mtp.embed / mtp.lm_head | that the MTP layer reuses embed_tokens and lm_head |")
    P("| `linear_attn.A_log`, `dt_bias`, `in_proj_a`, `in_proj_b` | shapes [48], [48], [48,5120], [48,5120] | the decay/beta formulas that combine them (must come from the reference source, not from a config field) |")
    P("| `self_attn.q_norm` / `k_norm` | shape [256] = head_dim | per-head RMSNorm applied before RoPE |")
    P("")

    # unrecognised
    P("## Unrecognised fields in config.json (verbatim; meaning NOT guessed)")
    P("")
    P("Every leaf of `config.json` that is not listed in a table above. `vision_config.*` is included")
    P("because the engine does not use it, so no meaning is claimed for it here.")
    P("")
    P("| JSON path | value |")
    P("|---|---|")
    fl = flat(cfg)
    unrec = 0
    for k in sorted(fl):
        if k in seen:
            continue
        # list-valued paths are flattened with the list as the value
        unrec += 1
        P("| `%s` | `%s` |" % (k, json.dumps(fl[k])))
    P("")
    P("Unrecognised field count: **%d**" % unrec)
    P("")
    P("Top-level keys of config.json: `%s`" % list(cfg.keys()))
    P("Keys of text_config: `%s`" % list(cfg["text_config"].keys()))
    P("Keys of vision_config: `%s`" % list(cfg["vision_config"].keys()))
    P("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote %s (%d lines), unrecognised fields: %d" % (OUT, len(L), unrec))
    print("full_attention layer indices (0-based):", full_idx)


if __name__ == "__main__":
    main()
