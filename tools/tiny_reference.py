"""HF reference for the tiny oracle: greedy 20 tokens on the 3 fixed prompts, per-layer hidden states at every
prompt position, and last-position logits, from the HF Qwen3_5ForCausalLM loaded from models/tiny
(float32, eager attention). Writes tests/fixtures/tiny/manifest.json + .npy files.

Hidden states are captured with forward hooks on each decoder layer (the residual stream after layer i),
plus the embedding output (layer index -1 -> "embed") and the final norm output. Greedy decoding is a manual
loop with a DynamicCache (argmax, no stop condition) so it mirrors the Rust oracle exactly.

Usage: python tools/tiny_reference.py [--model models/tiny]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "fixtures"))
from common import Fixture  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

GREEDY = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "tiny"))
    a = ap.parse_args()
    model = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    prompts = json.load(open(os.path.join(ROOT, "tests", "fixtures", "prompts.json")))["prompts"]
    fx = Fixture("tiny", subdir="tiny")
    fx.meta["source"] = "Qwen3_5ForCausalLM.from_pretrained(models/tiny), float32, eager attention; hooks on model.model.layers[i]"
    fx.meta["greedy_tokens"] = GREEDY
    fx.meta["budget_k_per_layer"] = 16
    fx.meta["budget_note"] = "hidden state after layer L: k * (L + 1) * sqrt(hidden) * eps * max|h_L| (rounding compounds per layer); logits: k * (n_layer + 2) * sqrt(hidden) * eps * max|logits|"
    fx.meta["tiny_config"] = {k: getattr(cfg, k) for k in ["num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                                                            "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim", "vocab_size"]}
    captured = {}

    def hook(i):
        def f(_m, _inp, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return f

    handles = [model.model.layers[i].register_forward_hook(hook(i)) for i in range(cfg.num_hidden_layers)]
    handles.append(model.model.embed_tokens.register_forward_hook(hook("embed")))
    handles.append(model.model.norm.register_forward_hook(hook("final_norm")))

    for p in prompts:
        ids = torch.tensor([p["ids"]], dtype=torch.long)
        t = ids.shape[1]
        captured.clear()
        cache = DynamicCache(config=cfg)
        with torch.no_grad():
            out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        logits = out.logits[0, -1].float()
        hidden = {}
        c = p["name"]
        for i in range(cfg.num_hidden_layers):
            h = captured[i][0]
            hidden["layer_%d" % i] = dict(fx.array(c, "hidden_layer_%d" % i, h), max_abs=float(h.abs().max()))
        emb = captured["embed"][0]
        hidden["embed"] = dict(fx.array(c, "hidden_embed", emb), max_abs=float(emb.abs().max()))
        fin = captured["final_norm"][0]
        hidden["final_norm"] = dict(fx.array(c, "hidden_final_norm", fin), max_abs=float(fin.abs().max()))
        argmax_per_pos = [int(x) for x in out.logits[0].argmax(-1)]
        # greedy continuation, one token at a time with the cache
        greedy = []
        step_argmax_logit = []
        cur = logits
        with torch.no_grad():
            for _ in range(GREEDY):
                nxt = int(cur.argmax())
                greedy.append(nxt)
                step_argmax_logit.append(float(cur[nxt]))
                o = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
                cur = o.logits[0, -1].float()
        top5 = [int(i) for i in torch.topk(logits, 5).indices]
        fx.add(c, prompt_ids=p["ids"], n_ids=t, hidden=hidden, logits_last=fx.array(c, "logits_last", logits),
               logits_max_abs=float(logits.abs().max()), argmax_per_position=argmax_per_pos, top5_last=top5,
               greedy_ids=greedy, greedy_top_logit=step_argmax_logit,
               margin_last=float(torch.topk(logits, 2).values[0] - torch.topk(logits, 2).values[1]))
        print("%-9s T=%d greedy=%s margin=%.4f" % (c, t, greedy, float(torch.topk(logits, 2).values[0] - torch.topk(logits, 2).values[1])))
    for h in handles:
        h.remove()
    fx.write()


if __name__ == "__main__":
    main()
