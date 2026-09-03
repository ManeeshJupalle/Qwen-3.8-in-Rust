"""Layer fixtures for Phase 2a from the HF Qwen3.5 submodules with random weights, emitted in GGUF layout.

Layers: gated_deltanet (Qwen3_5GatedDeltaNet), gqa_attention (Qwen3_5Attention), mlp (Qwen3_5MLP), plus the
full decoder layer (Qwen3_5DecoderLayer) for both block types. Each has a prefill case (T tokens, fresh state)
and a step case (one more token with the carried state / KV cache from the prefill), computed with a
DynamicCache exactly as the HF model does at decode time. Weights are converted to GGUF layout with the
same transforms as llama.cpp's converter (tools/fixtures/common.py, docs/deltanet.md).

Usage: python tools/fixtures/gen_layers.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Fixture, check_reorder_matches_converter, gguf_layout_deltanet, gguf_layout_norm  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as M  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

torch.manual_seed(20260904)
DEN = 1e-40

# Tiny config that keeps the real structure: 3 V heads per K head, partial rotary 0.25, zero-centered norms.
CFG = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
           linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=8, linear_value_head_dim=8,
           linear_conv_kernel_dim=4, hidden_act="silu", rms_norm_eps=1e-6, attn_output_gate=True, attention_bias=False,
           layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"], full_attention_interval=4,
           rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                            "mrope_section": [1, 1, 0], "mrope_interleaved": True},
           max_position_embeddings=4096, vocab_size=512, mamba_ssm_dtype="float32", output_gate_type="swish")


def config():
    c = Qwen3_5TextConfig(**CFG)
    c._attn_implementation = "eager"
    return c


def randomize(module, scale=0.3):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(0.01, 16)))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.5)
            elif "norm" in name and name.endswith("weight"):
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if "linear_attn.norm" in name or type(module).__name__ == "Qwen3_5RMSNormGated" else 0.0))
            else:
                p.copy_(torch.randn_like(p) * scale)


def hostile_x(t, hidden):
    x = torch.randn(1, t, hidden)
    x[0, 0, :4] = torch.tensor([DEN, -0.0, 5.0, -5.0])
    if t > 2:
        x[0, 2] = 0.7
    return x


def positions(cfg, t, past=0):
    ids = torch.arange(past, past + t).view(1, 1, -1).expand(4, 1, -1)
    return ids[0], ids[1:]


def add_arrays(fx, case, d):
    return {k: fx.array(case, k, v) for k, v in d.items()}


# ------------------------------------------------------------------------------------------- gated deltanet
def gen_deltanet_layer(fx, cfg):
    mod = M.Qwen3_5GatedDeltaNet(cfg, layer_idx=0).eval()
    randomize(mod)
    sd = {k: v for k, v in mod.state_dict().items()}
    gg = gguf_layout_deltanet(sd, "", cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim)
    t = 6
    x = hostile_x(t, cfg.hidden_size)
    cache = DynamicCache(config=cfg)
    mask = create_recurrent_attention_mask(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=cache, position_ids=positions(cfg, t)[0])
    with torch.no_grad():
        y = mod(x, cache_params=cache, attention_mask=mask)
    conv_state = cache.layers[0].conv_states[0].clone()  # (1, conv_dim, kernel-1)
    rec_state = cache.layers[0].recurrent_states[0].clone()  # (1, heads, dk, dv)
    x1 = hostile_x(1, cfg.hidden_size)
    with torch.no_grad():
        y1 = mod(x1, cache_params=cache, attention_mask=None)
    conv_state1 = cache.layers[0].conv_states[0].clone()
    rec_state1 = cache.layers[0].recurrent_states[0].clone()
    # states in GGUF head order: conv channels (q,k unchanged; v re-tiled) and recurrent heads re-tiled
    from common import reorder_v_heads
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    kd = nk * cfg.linear_key_head_dim
    def conv_gguf(s):
        s = s[0]
        return torch.cat([s[: 2 * kd], reorder_v_heads(s[2 * kd:], 0, nk, nv // nk, cfg.linear_value_head_dim)], dim=0)
    def rec_gguf(s):
        return reorder_v_heads(s[0], 0, nk, nv // nk, 1)
    c = "deltanet_layer"
    fx.add(c, kind="gated_deltanet", hidden=cfg.hidden_size, num_k_heads=nk, num_v_heads=nv, head_k=cfg.linear_key_head_dim,
           head_v=cfg.linear_value_head_dim, conv_kernel=cfg.linear_conv_kernel_dim, eps=cfg.rms_norm_eps, t=t,
           weights=add_arrays(fx, c, gg),
           prefill=dict(x=fx.array(c, "x", x[0]), y=fx.array(c, "y", y[0]), conv_state=fx.array(c, "conv_state", conv_gguf(conv_state)),
                        rec_state=fx.array(c, "rec_state", rec_gguf(rec_state)), max_abs=float(y.abs().max())),
           step=dict(x=fx.array(c, "x1", x1[0]), y=fx.array(c, "y1", y1[0]), conv_state=fx.array(c, "conv_state1", conv_gguf(conv_state1)),
                     rec_state=fx.array(c, "rec_state1", rec_gguf(rec_state1)), max_abs=float(y1.abs().max())),
           state_note="states are in GGUF head order (V heads re-tiled: position p = v_in_group * num_k + k)")
    return mod


# ------------------------------------------------------------------------------------------- gqa attention
def gen_attention_layer(fx, cfg):
    mod = M.Qwen3_5Attention(cfg, layer_idx=3).eval()
    randomize(mod)
    rot = M.Qwen3_5TextRotaryEmbedding(cfg)
    t = 6
    x = hostile_x(t, cfg.hidden_size)
    cache = DynamicCache(config=cfg)
    text_pos, pos3 = positions(cfg, t)
    mask = create_causal_mask(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=cache, position_ids=text_pos)
    with torch.no_grad():
        pe = rot(x, pos3)
        y, _ = mod(x, position_embeddings=pe, attention_mask=mask, past_key_values=cache)
        k_cache = cache.layers[3].keys.clone()
        v_cache = cache.layers[3].values.clone()
    x1 = hostile_x(1, cfg.hidden_size)
    text_pos1, pos31 = positions(cfg, 1, past=t)
    mask1 = create_causal_mask(config=cfg, inputs_embeds=x1, attention_mask=None, past_key_values=cache, position_ids=text_pos1)
    with torch.no_grad():
        pe1 = rot(x1, pos31)
        y1, _ = mod(x1, position_embeddings=pe1, attention_mask=mask1, past_key_values=cache)
    sd = mod.state_dict()
    gg = {
        "attn_q.weight": sd["q_proj.weight"], "attn_k.weight": sd["k_proj.weight"], "attn_v.weight": sd["v_proj.weight"],
        "attn_output.weight": sd["o_proj.weight"],
        "attn_q_norm.weight": gguf_layout_norm(sd["q_norm.weight"]), "attn_k_norm.weight": gguf_layout_norm(sd["k_norm.weight"]),
    }
    c = "attention_layer"
    fx.add(c, kind="gqa_attention", hidden=cfg.hidden_size, n_head=cfg.num_attention_heads, n_head_kv=cfg.num_key_value_heads,
           head_dim=cfg.head_dim, rope_dim=int(cfg.head_dim * 0.25), theta=10000000.0, eps=cfg.rms_norm_eps, t=t,
           weights=add_arrays(fx, c, gg),
           prefill=dict(x=fx.array(c, "x", x[0]), y=fx.array(c, "y", y[0]), k_cache=fx.array(c, "k_cache", k_cache[0]), v_cache=fx.array(c, "v_cache", v_cache[0]),
                        cos=fx.array(c, "cos", pe[0][0]), sin=fx.array(c, "sin", pe[1][0]), max_abs=float(y.abs().max())),
           step=dict(x=fx.array(c, "x1", x1[0]), y=fx.array(c, "y1", y1[0]), position=t, cos=fx.array(c, "cos1", pe1[0][0]), sin=fx.array(c, "sin1", pe1[1][0]),
                     max_abs=float(y1.abs().max())),
           layout_note="attn_q rows: per head [q(head_dim) | gate(head_dim)]; q_norm/k_norm before RoPE; RoPE on the first rope_dim dims; attn_output * sigmoid(gate) before o_proj; KV cache stores post-RoPE keys")
    return mod


# ------------------------------------------------------------------------------------------- mlp
def gen_mlp_layer(fx, cfg):
    mod = M.Qwen3_5MLP(cfg, cfg.intermediate_size).eval()
    randomize(mod)
    x = hostile_x(5, cfg.hidden_size)
    with torch.no_grad():
        y = mod(x)
    sd = mod.state_dict()
    gg = {"ffn_gate.weight": sd["gate_proj.weight"], "ffn_up.weight": sd["up_proj.weight"], "ffn_down.weight": sd["down_proj.weight"]}
    c = "mlp_layer"
    fx.add(c, kind="mlp", hidden=cfg.hidden_size, intermediate=cfg.intermediate_size, t=5, weights=add_arrays(fx, c, gg),
           prefill=dict(x=fx.array(c, "x", x[0]), y=fx.array(c, "y", y[0]), max_abs=float(y.abs().max())))


# ------------------------------------------------------------------------------------------- decoder layers
def gen_decoder_layers(fx, cfg):
    rot = M.Qwen3_5TextRotaryEmbedding(cfg)
    for idx, kind in ((0, "linear_attention"), (3, "full_attention")):
        layer = M.Qwen3_5DecoderLayer(cfg, idx).eval()
        randomize(layer)
        t = 6
        x = hostile_x(t, cfg.hidden_size)
        cache = DynamicCache(config=cfg)
        text_pos, pos3 = positions(cfg, t)
        masks = {"full_attention": create_causal_mask(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=cache, position_ids=text_pos),
                 "linear_attention": create_recurrent_attention_mask(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=cache, position_ids=text_pos)}
        with torch.no_grad():
            pe = rot(x, pos3)
            y = layer(x, position_embeddings=pe, attention_mask=masks[kind], position_ids=text_pos, past_key_values=cache, use_cache=True)
        x1 = hostile_x(1, cfg.hidden_size)
        text_pos1, pos31 = positions(cfg, 1, past=t)
        masks1 = {"full_attention": create_causal_mask(config=cfg, inputs_embeds=x1, attention_mask=None, past_key_values=cache, position_ids=text_pos1),
                  "linear_attention": None}
        with torch.no_grad():
            pe1 = rot(x1, pos31)
            y1 = layer(x1, position_embeddings=pe1, attention_mask=masks1[kind], position_ids=text_pos1, past_key_values=cache, use_cache=True)
        sd = layer.state_dict()
        gg = {"attn_norm.weight": gguf_layout_norm(sd["input_layernorm.weight"]),
              "post_attention_norm.weight": gguf_layout_norm(sd["post_attention_layernorm.weight"]),
              "ffn_gate.weight": sd["mlp.gate_proj.weight"], "ffn_up.weight": sd["mlp.up_proj.weight"], "ffn_down.weight": sd["mlp.down_proj.weight"]}
        if kind == "linear_attention":
            gg.update(gguf_layout_deltanet(sd, "linear_attn.", cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim))
        else:
            gg.update({"attn_q.weight": sd["self_attn.q_proj.weight"], "attn_k.weight": sd["self_attn.k_proj.weight"],
                       "attn_v.weight": sd["self_attn.v_proj.weight"], "attn_output.weight": sd["self_attn.o_proj.weight"],
                       "attn_q_norm.weight": gguf_layout_norm(sd["self_attn.q_norm.weight"]), "attn_k_norm.weight": gguf_layout_norm(sd["self_attn.k_norm.weight"])})
        c = "decoder_%s" % kind
        fx.add(c, kind="decoder_layer", layer_type=kind, layer_idx=idx, t=t, weights=add_arrays(fx, c, gg),
               prefill=dict(x=fx.array(c, "x", x[0]), y=fx.array(c, "y", y[0]), max_abs=float(y.abs().max())),
               step=dict(x=fx.array(c, "x1", x1[0]), y=fx.array(c, "y1", y1[0]), position=t, max_abs=float(y1.abs().max())))


def main():
    cfg = config()
    fx = Fixture("layers", subdir="layers")
    fx.meta["config"] = {k: v for k, v in CFG.items()}
    fx.meta["source"] = "modeling_qwen3_5.py: Qwen3_5GatedDeltaNet 387-548, Qwen3_5Attention 632-706, Qwen3_5MLP 707-722, Qwen3_5DecoderLayer 743-797; eager attention"
    fx.meta["budget_k"] = 16
    fx.meta["reorder_check"] = check_reorder_matches_converter()
    gen_deltanet_layer(fx, cfg)
    gen_attention_layer(fx, cfg)
    gen_mlp_layer(fx, cfg)
    gen_decoder_layers(fx, cfg)
    fx.write()
    print(fx.meta["reorder_check"])


if __name__ == "__main__":
    main()
