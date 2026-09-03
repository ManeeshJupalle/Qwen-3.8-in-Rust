"""HF transformers bf16 reference, ONE LAYER AT A TIME, from the safetensors shards.

Never calls from_pretrained on the full model. For each of the 64 text layers it builds the
transformers Qwen3_5DecoderLayer, loads that layer's tensors with safe_open, runs it on CPU,
records max-abs of the hidden state, frees it, and moves on. At the end: final norm, lm_head,
last-position logits (float32 .npy), argmax, top-10, in the same shapes as tools/ref_llamacpp.py.

Requires the bf16 shards in <hf_dir> (about 52 GiB; NOT downloaded in Phase 0 without approval).

STATUS: validated by tools/ref_forward_selftest.py on a tiny random Qwen3_5 model (bit-identical to the full
Qwen3_5TextModel forward in float32 and bfloat16 with eager attention). Not yet run on the real 27B shards.
Attention is forced to "eager" so results are deterministic; SDPA in bf16 differs from eager by bf16 rounding.

Usage:
  python tools/ref_forward.py <hf_dir> --prompt fib            # shortest prompt first, prints wall time
  python tools/ref_forward.py <hf_dir> --prompt capital --prompt sentence
  options: --greedy N   (N extra full recomputes; every token re-streams all shards, ~minutes each)
           --dump-hidden (also save per-layer hidden states for the whole prompt, ~50 MB per prompt)
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures")
OUT = os.path.join(FIX, "ref_hf")
PREFIX = "model.language_model."


class ShardStore:
    """Lazy safe_open over the shards named in model.safetensors.index.json."""

    def __init__(self, hf_dir):
        self.hf_dir = hf_dir
        self.weight_map = json.load(open(os.path.join(hf_dir, "model.safetensors.index.json")))["weight_map"]
        self.open = {}

    def get(self, name):
        shard = self.weight_map[name]
        if shard not in self.open:
            self.open[shard] = safe_open(os.path.join(self.hf_dir, shard), framework="pt", device="cpu")
        return self.open[shard].get_tensor(name)

    def layer_state_dict(self, i):
        pre = "%slayers.%d." % (PREFIX, i)
        return {k[len(pre):]: self.get(k) for k in self.weight_map if k.startswith(pre)}


def forward_logits(store, tcfg, ids, dump_dir=None, log=print, dtype=torch.bfloat16):
    """Full forward over all layers for token ids -> (logits[-1] float32, per-layer stats).
    dtype: compute/weight dtype (bf16 = the checkpoint's dtype; float32 for debugging the plumbing).
    Attention runs with the eager implementation (explicit softmax in float32), set on tcfg by the caller."""
    torch.set_default_dtype(dtype)
    T = len(ids)
    ids_t = torch.tensor([ids], dtype=torch.long)
    t0 = time.time()
    emb = store.get(PREFIX + "embed_tokens.weight").to(dtype)
    h = emb[ids_t]  # (1, T, hidden)
    del emb
    gc.collect()

    rotary = Qwen3_5TextRotaryEmbedding(tcfg)
    position_ids = torch.arange(T).view(1, 1, -1).expand(4, 1, -1)
    text_pos = position_ids[0]
    pos3 = position_ids[1:]
    pe = rotary(h, pos3)
    mask_kwargs = {"config": tcfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": None,
                   "position_ids": text_pos}
    masks = {"full_attention": create_causal_mask(**mask_kwargs),
             "linear_attention": create_recurrent_attention_mask(**mask_kwargs)}

    stats = []
    for i in range(tcfg.num_hidden_layers):
        t1 = time.time()
        layer = Qwen3_5DecoderLayer(tcfg, i)
        sd = {k: v.to(dtype) for k, v in store.layer_state_dict(i).items()}
        missing, unexpected = layer.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        layer.eval()
        t_load = time.time() - t1
        with torch.no_grad():
            h = layer(h, position_embeddings=pe, attention_mask=masks[tcfg.layer_types[i]],
                      position_ids=text_pos, past_key_values=None, use_cache=False)
        t_run = time.time() - t1 - t_load
        hf = h.float()
        rec = {"layer": i, "type": tcfg.layer_types[i], "max_abs": float(hf.abs().max()),
               "last_pos_max_abs": float(hf[0, -1].abs().max()), "mean_abs": float(hf.abs().mean()),
               "n_nan": int(torch.isnan(hf).sum()), "load_s": round(t_load, 2), "run_s": round(t_run, 2)}
        stats.append(rec)
        log("layer %2d %-16s max|h|=%10.4f last=%10.4f nan=%d load=%.1fs run=%.1fs" % (
            i, rec["type"], rec["max_abs"], rec["last_pos_max_abs"], rec["n_nan"], t_load, t_run))
        if dump_dir:
            np.save(os.path.join(dump_dir, "hidden_after_layer_%02d.npy" % i), hf[0].numpy())
        del layer, sd, hf
        gc.collect()

    norm = Qwen3_5RMSNorm(tcfg.hidden_size, eps=tcfg.rms_norm_eps)
    norm.load_state_dict({"weight": store.get(PREFIX + "norm.weight").to(dtype)}, strict=True)
    with torch.no_grad():
        h = norm(h)
        lm_head = store.get("lm_head.weight").to(dtype)  # (vocab, hidden)
        logits = (h[0, -1:] @ lm_head.T).float()[0]  # matmul in `dtype` like HF, then float32
    del lm_head
    gc.collect()
    total = time.time() - t0
    return logits.numpy().astype(np.float32), stats, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("--prompt", action="append", default=None, help="prompt name(s) from prompts.json; default: shortest")
    ap.add_argument("--greedy", type=int, default=0)
    ap.add_argument("--dump-hidden", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    cfg = AutoConfig.from_pretrained(a.hf_dir)
    tcfg = cfg.text_config
    tcfg._attn_implementation = "eager"  # explicit; standalone layers cannot dispatch sdpa
    prompts = json.load(open(os.path.join(FIX, "prompts.json")))["prompts"]
    if a.prompt:
        prompts = [p for p in prompts if p["name"] in a.prompt]
    else:
        prompts = [min(prompts, key=lambda p: p["n_ids"])]
    store = ShardStore(a.hf_dir)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf_dir)

    for p in prompts:
        ids = list(p["ids"])
        dump_dir = None
        if a.dump_hidden:
            dump_dir = os.path.join(OUT, p["name"] + "_hidden")
            os.makedirs(dump_dir, exist_ok=True)
        print("=== prompt %s (%d ids) ===" % (p["name"], len(ids)))
        logits, stats, secs = forward_logits(store, tcfg, ids, dump_dir)
        argmax = int(logits.argmax())
        top10 = [int(i) for i in np.argsort(-logits)[:10]]
        gen = []
        cur_ids = list(ids)
        cur = logits
        for _ in range(a.greedy):
            nxt = int(cur.argmax())
            gen.append(nxt)
            cur_ids.append(nxt)
            cur, _, s = forward_logits(store, tcfg, cur_ids, None, log=lambda *x: None)
            print("  greedy token %d -> %d %r (%.0fs)" % (len(gen), nxt, tok.decode([nxt]), s))
        if not np.isfinite(logits).all() or not np.any(logits) or float(logits.std()) < 1e-6:
            raise RuntimeError("%s: logits all-zero/constant/non-finite; refusing to write" % p["name"])
        np.save(os.path.join(OUT, p["name"] + ".logits.npy"), logits)
        rec = {
            "name": p["name"], "prompt_text": p["text"], "prompt_ids": ids, "n_prompt_ids": len(ids),
            "argmax": argmax, "argmax_text": tok.decode([argmax]),
            "top10": top10, "top10_logits": [float(logits[i]) for i in top10],
            "top10_text": [tok.decode([i]) for i in top10],
            "greedy_ids": gen, "greedy_text": tok.decode(gen) if gen else "",
            "logits_stats": {"max": float(logits.max()), "min": float(logits.min()), "mean": float(logits.mean()),
                             "std": float(logits.std()), "n_nan": int(np.isnan(logits).sum())},
            "per_layer": stats, "wall_seconds": round(secs, 1),
            "info": {"transformers": __import__("transformers").__version__, "torch": torch.__version__,
                     "dtype": "bfloat16 weights/activations, float32 logits; layers run one at a time from safetensors"},
        }
        with open(os.path.join(OUT, p["name"] + ".json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False)
        print("%-9s argmax=%d %r top10=%s greedy=%r wall=%.0fs" % (p["name"], argmax, rec["argmax_text"], top10, rec["greedy_text"], secs))


if __name__ == "__main__":
    main()
