"""Build the tiny oracle checkpoint: the HF Qwen3.5 text model class with a tiny config that keeps the real graph
(9 layers = two full 3:1 blocks + 1, full_attention_interval 4, hidden 64, 2 KV heads, small DeltaNet heads),
random init with a fixed seed, saved as safetensors + config.json, then converted to an F32 GGUF with
llama.cpp's convert_hf_to_gguf.py so tensor names, permutations and metadata are exactly what the real file
went through.

Deviations from the task text, both forced by the converter (stated in the report):
  - vocab: the real tokenizer is reused and vocab_size is the real 248,320. The converter hashes the tokenizer
    to pick `tokenizer.ggml.pre` and asserts `max(vocab id) < vocab_size`; a 512-token tokenizer would be
    rejected as an unknown pre-tokenizer.
  - an MTP block (random weights, mtp.* tensors) is included because the converter asserts on
    mtp_num_hidden_layers > 0 without tensors, and the real config has mtp_num_hidden_layers = 1.

Usage: python tools/make_tiny_checkpoint.py [--out models/tiny] [--no-convert]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

REAL = os.path.join(ROOT, "models", "Qwen3.8-27B")
SEED = 20260905

# structural flags copied from the real config.json (text_config); sizes shrunk
TINY = dict(
    num_hidden_layers=9,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2 + ["linear_attention"],
    full_attention_interval=4,
    hidden_size=64,
    intermediate_size=96,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=6,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    mtp_num_hidden_layers=1,
)
COPY_FROM_REAL = ["attention_bias", "attention_dropout", "attn_output_gate", "bos_token_id", "eos_token_id", "hidden_act",
                  "initializer_range", "mamba_ssm_dtype", "max_position_embeddings", "mtp_use_dedicated_embeddings",
                  "output_gate_type", "pad_token_id", "rms_norm_eps", "tie_word_embeddings", "use_cache", "vocab_size"]


def build_config():
    real = json.load(open(os.path.join(REAL, "config.json"), encoding="utf-8"))["text_config"]
    kw = {k: real[k] for k in COPY_FROM_REAL}
    kw.update(TINY)
    rp = dict(real["rope_parameters"])
    rope_dim = int(TINY["head_dim"] * rp["partial_rotary_factor"])  # 4
    rp["mrope_section"] = [1, 1, 0]  # sums to rope_dim / 2 = 2, same T/H/W layout as [11, 11, 10]
    kw["rope_parameters"] = rp
    cfg = Qwen3_5TextConfig(**kw)
    cfg.architectures = ["Qwen3_5ForCausalLM"]
    return cfg, rope_dim


def mtp_tensors(cfg, gen):
    """Random MTP block with the real checkpoint's shapes scaled down: fc [hidden, 2*hidden], one attention block, norms."""
    h, inter, hd = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    def w(*shape, scale=0.02):
        return (torch.randn(*shape, generator=gen) * scale).contiguous()
    return {
        "mtp.fc.weight": w(h, 2 * h),
        "mtp.layers.0.input_layernorm.weight": w(h, scale=0.1),
        "mtp.layers.0.post_attention_layernorm.weight": w(h, scale=0.1),
        "mtp.layers.0.self_attn.q_proj.weight": w(2 * nh * hd, h),
        "mtp.layers.0.self_attn.k_proj.weight": w(nkv * hd, h),
        "mtp.layers.0.self_attn.v_proj.weight": w(nkv * hd, h),
        "mtp.layers.0.self_attn.o_proj.weight": w(h, nh * hd),
        "mtp.layers.0.self_attn.q_norm.weight": w(hd, scale=0.1),
        "mtp.layers.0.self_attn.k_norm.weight": w(hd, scale=0.1),
        "mtp.layers.0.mlp.gate_proj.weight": w(inter, h),
        "mtp.layers.0.mlp.up_proj.weight": w(inter, h),
        "mtp.layers.0.mlp.down_proj.weight": w(h, inter),
        "mtp.norm.weight": w(h, scale=0.1),
        "mtp.pre_fc_norm_embedding.weight": w(h, scale=0.1),
        "mtp.pre_fc_norm_hidden.weight": w(h, scale=0.1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "tiny"))
    ap.add_argument("--no-convert", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(SEED)
    cfg, rope_dim = build_config()
    cfg._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(cfg).eval()
    # HF init leaves zero-centered norms at 0 and A_log/dt_bias at their init; perturb norms so they are not trivial
    gen = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") and "linear_attn.norm" not in name:
                p.add_(torch.randn(p.shape, generator=gen) * 0.1)
            elif name.endswith("linear_attn.norm.weight"):
                p.copy_(1.0 + torch.randn(p.shape, generator=gen) * 0.1)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
    model.save_pretrained(a.out, safe_serialization=True)
    # add the MTP block to the single shard
    st = os.path.join(a.out, "model.safetensors")
    sd = load_file(st)
    sd.update(mtp_tensors(cfg, gen))
    save_file(sd, st, metadata={"format": "pt"})
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        shutil.copy(os.path.join(REAL, f), os.path.join(a.out, f))
    # config.json: make sure the converter sees the class name and flat text hparams
    cj = json.load(open(os.path.join(a.out, "config.json"), encoding="utf-8"))
    cj["architectures"] = ["Qwen3_5ForCausalLM"]
    json.dump(cj, open(os.path.join(a.out, "config.json"), "w", encoding="utf-8"), indent=2)
    n_params = sum(p.numel() for p in model.parameters())
    print("tiny model: %d params (%d without embed/lm_head), rope_dim %d, mtp tensors %d" % (
        n_params, n_params - 2 * cfg.vocab_size * cfg.hidden_size, rope_dim, len(mtp_tensors(cfg, gen))))
    print("saved", a.out)
    json.dump({"seed": SEED, "tiny": TINY, "copied_from_real": COPY_FROM_REAL, "rope_dim": rope_dim, "n_params": n_params,
               "mrope_section": [1, 1, 0]}, open(os.path.join(a.out, "tiny_manifest.json"), "w"), indent=1)
    if a.no_convert:
        return
    conv = os.path.join(ROOT, "models", "llama.cpp", "convert_hf_to_gguf.py")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "models", "llama.cpp", "gguf-py"))
    out_gguf = os.path.join(a.out, "tiny-f32.gguf")
    cmd = [sys.executable, conv, a.out, "--outfile", out_gguf, "--outtype", "f32"]
    print("running:", " ".join(cmd))
    log = subprocess.run(cmd, env=env, capture_output=True, text=True)
    os.makedirs(os.path.join(ROOT, "docs", "data"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "data", "tiny_convert.log"), "w", encoding="utf-8") as fh:
        fh.write("# %s\n# exit code %d\n" % (" ".join(cmd), log.returncode))
        fh.write(log.stdout)
        fh.write(log.stderr)
    print("converter exit code", log.returncode, "->", out_gguf, os.path.getsize(out_gguf) if os.path.exists(out_gguf) else "MISSING")
    print(log.stderr[-3000:])


if __name__ == "__main__":
    main()
