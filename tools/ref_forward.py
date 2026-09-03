"""HF transformers reference forward, ONE LAYER AT A TIME, from either weight source:

  --weights bf16   the 18 bf16 safetensors shards in <hf_dir> (the checkpoint as published)
  --weights gguf   the bartowski Q4_K_M GGUF, every tensor dequantised to float32 with gguf-py and mapped back
                   to the HF names and layout (V heads un-tiled, A_log from ssm_a, zero-centered norms minus 1,
                   conv1d unsqueezed; tools/fixtures/common.py::hf_layout_deltanet). This is the Phase 2b parity
                   target: the engine reads the same bytes, so only rounding separates the two.

Never calls from_pretrained on the full model. Each of the 64 text layers is built as a standalone
Qwen3_5DecoderLayer, loaded, run on every prompt (so each layer's weights are read from disk once), freed.
Then final norm and lm_head (in row chunks, so the 248,320 x 5,120 matrix is never fully resident in f32).

Outputs, per prompt, in --out (default tests/fixtures/ref_<weights>/):
  <name>.json          argmax, top-10, argmax at every prompt position, per-layer max|h|, timing, provenance
  <name>.logits.npy    float32 logits at the last position (full vocab)
  with --dump-hidden:  <name>__hidden_embed.npy, <name>__hidden_after_layer_NN.npy, <name>__hidden_final_norm.npy
                       (T, hidden) float32: the residual stream after every layer, every position
  info.json            run-level provenance (weight source, dtype, versions, sizes)

--dtype f32 (default) keeps weights and activations in float32 (bf16 weights are upcast exactly). --dtype bf16
runs the checkpoint's native precision, eager attention with a float32 softmax, as HF would; only valid with
--weights bf16. Attention is always "eager" so results are deterministic.

STATUS: tools/ref_forward_selftest.py checks the bf16 path plumbing against the full Qwen3_5TextModel on a tiny
random model; tools/ref_gguf_selftest.py checks the gguf path on models/tiny/tiny-f32.gguf against the Phase 2a
tiny fixture (tests/fixtures/tiny/), i.e. the inverse layout transforms end to end.

Usage:
  python tools/ref_forward.py models/Qwen3.8-27B --weights gguf --dump-hidden
  python tools/ref_forward.py models/Qwen3.8-27B --weights bf16 --dump-hidden
  options: --prompt NAME (repeatable; default all three), --layers N (smoke test: stop after N layers, write
           nothing), --gguf PATH, --out DIR, --dtype f32|bf16
"""
import argparse
import gc
import json
import os
import sys
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
PREFIX = "model.language_model."
GGUF_DEFAULT = os.environ.get("AQUEDUCT_GGUF", r"C:\models\Qwen3.8-27B-Q4_K_M.gguf")  # on the SSD since Phase 3 (finding 34)
LM_HEAD_CHUNK_ROWS = 16384

sys.path.insert(0, os.path.join(ROOT, "tools", "fixtures"))
from common import check_gguf_hf_roundtrip, hf_layout_deltanet, hf_layout_norm  # noqa: E402


class ShardStore:
    """Lazy safe_open over the bf16 shards named in model.safetensors.index.json."""

    kind = "bf16"

    def __init__(self, hf_dir):
        self.hf_dir = hf_dir
        self.weight_map = json.load(open(os.path.join(hf_dir, "model.safetensors.index.json")))["weight_map"]
        self.open = {}

    def _shard(self, name):
        shard = self.weight_map[name]
        if shard not in self.open:
            self.open[shard] = safe_open(os.path.join(self.hf_dir, shard), framework="pt", device="cpu")
        return self.open[shard]

    def get(self, name):
        return self._shard(name).get_tensor(name)

    def layer_state_dict(self, i):
        pre = "%slayers.%d." % (PREFIX, i)
        return {k[len(pre):]: self.get(k) for k in self.weight_map if k.startswith(pre)}

    def embed_rows(self, ids):
        sl = self._shard(PREFIX + "embed_tokens.weight").get_slice(PREFIX + "embed_tokens.weight")
        return torch.stack([sl[i:i + 1][0] for i in ids])

    def final_norm(self):
        return self.get(PREFIX + "norm.weight")

    def lm_head_chunks(self, vocab):
        # the whole bf16 tensor (2.5 GB) is loaded once and sliced here: safetensors get_slice on it
        # terminated the process silently on Windows during the first bf16 run
        w = self.get("lm_head.weight")
        for a in range(0, vocab, LM_HEAD_CHUNK_ROWS):
            yield a, w[a:min(a + LM_HEAD_CHUNK_ROWS, vocab)]

    def describe(self):
        shards = sorted(set(self.weight_map.values()))
        return {"weights": "bf16", "hf_dir": self.hf_dir, "shards": len(shards),
                "shard_bytes": sum(os.path.getsize(os.path.join(self.hf_dir, s)) for s in shards)}


class GgufStore:
    """The bartowski GGUF, dequantised to float32 with gguf-py (models/llama.cpp/gguf-py, the converter's own
    package) and mapped back to HF names and layout. Only the rows needed are dequantised for the embedding
    and the lm_head."""

    kind = "gguf"

    def __init__(self, path, tcfg):
        gguf_py = os.path.join(ROOT, "models", "llama.cpp", "gguf-py")
        if os.path.isdir(gguf_py) and gguf_py not in sys.path:
            sys.path.insert(0, gguf_py)
        import gguf
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.gguf_module = gguf.__file__
        self.path = path
        self.dequantize = dequantize
        self.reader = GGUFReader(path)
        self.t = {t.name: t for t in self.reader.tensors}
        self.tcfg = tcfg
        self.a_log_exact = {}
        self.roundtrip = check_gguf_hf_roundtrip()

    def deq(self, name, rows=None):
        t = self.t[name]
        data = t.data if rows is None else t.data[rows]
        arr = np.ascontiguousarray(self.dequantize(data, t.tensor_type), dtype=np.float32)
        if not arr.flags.writeable:  # F32 tensors are views of the memory map; torch wants an owned buffer
            arr = arr.copy()
        return torch.from_numpy(arr)

    def ggml_type(self, name):
        return self.t[name].tensor_type.name

    def layer_state_dict(self, i):
        c = self.tcfg
        p = "blk.%d." % i
        sd = {
            "input_layernorm.weight": hf_layout_norm(self.deq(p + "attn_norm.weight")),
            "post_attention_layernorm.weight": hf_layout_norm(self.deq(p + "post_attention_norm.weight")),
            "mlp.gate_proj.weight": self.deq(p + "ffn_gate.weight"),
            "mlp.up_proj.weight": self.deq(p + "ffn_up.weight"),
            "mlp.down_proj.weight": self.deq(p + "ffn_down.weight"),
        }
        if c.layer_types[i] == "linear_attention":
            gg = {k: self.deq(p + k) for k in ("attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_a",
                                               "ssm_dt.bias", "ssm_norm.weight", "ssm_conv1d.weight", "ssm_out.weight")}
            dn, n_exact = hf_layout_deltanet(gg, c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim)
            self.a_log_exact[i] = [n_exact, c.linear_num_value_heads]
            sd.update(dn)
        else:
            sd.update({
                "self_attn.q_proj.weight": self.deq(p + "attn_q.weight"),
                "self_attn.k_proj.weight": self.deq(p + "attn_k.weight"),
                "self_attn.v_proj.weight": self.deq(p + "attn_v.weight"),
                "self_attn.o_proj.weight": self.deq(p + "attn_output.weight"),
                "self_attn.q_norm.weight": hf_layout_norm(self.deq(p + "attn_q_norm.weight")),
                "self_attn.k_norm.weight": hf_layout_norm(self.deq(p + "attn_k_norm.weight")),
            })
        return sd

    def embed_rows(self, ids):
        return self.deq("token_embd.weight", rows=np.asarray(ids, dtype=np.int64))

    def final_norm(self):
        return hf_layout_norm(self.deq("output_norm.weight"))

    def lm_head_chunks(self, vocab):
        for a in range(0, vocab, LM_HEAD_CHUNK_ROWS):
            yield a, self.deq("output.weight", rows=slice(a, min(a + LM_HEAD_CHUNK_ROWS, vocab)))

    def describe(self):
        return {"weights": "gguf", "gguf": os.path.basename(self.path), "gguf_bytes": os.path.getsize(self.path),
                "gguf_module": self.gguf_module, "general.name": str(bytes(self.reader.fields["general.name"].parts[-1]), "utf-8"),
                "layout_roundtrip_check": self.roundtrip, "a_log_exact_per_layer": self.a_log_exact,
                "embed_type": self.ggml_type("token_embd.weight"), "output_type": self.ggml_type("output.weight")}


def run_prompts(store, tcfg, prompts, dtype, dump_dir=None, n_layers=None, log=print):
    """Forward every (name, ids) in `prompts` through the stack, one layer at a time, all prompts per layer.
    Returns ({name: {"logits": float32 (vocab,), "argmax_per_position": [..], "stats": [...]}}, wall seconds).
    Attention is eager (set on tcfg by the caller); masks and rotary embeddings are built as Qwen3_5TextModel does."""
    n_layers = tcfg.num_hidden_layers if n_layers is None else n_layers
    torch.set_default_dtype(dtype)
    t0 = time.time()
    try:
        rotary = Qwen3_5TextRotaryEmbedding(tcfg)
        P = {}
        for name, ids in prompts:
            T = len(ids)
            h = store.embed_rows(ids).to(dtype).unsqueeze(0)  # (1, T, hidden)
            position_ids = torch.arange(T).view(1, 1, -1).expand(4, 1, -1)
            text_pos = position_ids[0]
            pe = rotary(h, position_ids[1:])
            mask_kwargs = {"config": tcfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": None, "position_ids": text_pos}
            masks = {"full_attention": create_causal_mask(**mask_kwargs), "linear_attention": create_recurrent_attention_mask(**mask_kwargs)}
            P[name] = {"ids": list(ids), "h": h, "pe": pe, "masks": masks, "text_pos": text_pos, "stats": []}
            if dump_dir:
                np.save(os.path.join(dump_dir, "%s__hidden_embed.npy" % name), h[0].float().numpy())
        gc.collect()

        for i in range(n_layers):
            t1 = time.time()
            layer = Qwen3_5DecoderLayer(tcfg, i)
            sd = {k: v.to(dtype) for k, v in store.layer_state_dict(i).items()}
            missing, unexpected = layer.load_state_dict(sd, strict=True)
            assert not missing and not unexpected, (missing, unexpected)
            layer.eval()
            t_load = time.time() - t1
            for name, p in P.items():
                t2 = time.time()
                with torch.no_grad():
                    p["h"] = layer(p["h"], position_embeddings=p["pe"], attention_mask=p["masks"][tcfg.layer_types[i]],
                                   position_ids=p["text_pos"], past_key_values=None, use_cache=False)
                hf = p["h"].float()
                rec = {"layer": i, "type": tcfg.layer_types[i], "max_abs": float(hf.abs().max()),
                       "last_pos_max_abs": float(hf[0, -1].abs().max()), "mean_abs": float(hf.abs().mean()),
                       "n_nan": int(torch.isnan(hf).sum()), "load_s": round(t_load, 2), "run_s": round(time.time() - t2, 2)}
                p["stats"].append(rec)
                log("layer %2d %-16s %-9s max|h|=%10.4f last=%10.4f nan=%d load=%.1fs run=%.1fs" % (
                    i, rec["type"], name, rec["max_abs"], rec["last_pos_max_abs"], rec["n_nan"], t_load, rec["run_s"]))
                if dump_dir:
                    np.save(os.path.join(dump_dir, "%s__hidden_after_layer_%02d.npy" % (name, i)), hf[0].numpy())
            del layer, sd
            gc.collect()

        norm = Qwen3_5RMSNorm(tcfg.hidden_size, eps=tcfg.rms_norm_eps)
        norm.load_state_dict({"weight": store.final_norm().to(dtype)}, strict=True)
        vocab = tcfg.vocab_size
        for name, p in P.items():
            with torch.no_grad():
                p["hn"] = norm(p["h"])
            if dump_dir:
                np.save(os.path.join(dump_dir, "%s__hidden_final_norm.npy" % name), p["hn"][0].float().numpy())
            p["logits"] = torch.empty(len(p["ids"]), vocab, dtype=torch.float32)
        t3 = time.time()
        for a, w in store.lm_head_chunks(vocab):
            wd = w.to(dtype)
            for name, p in P.items():
                with torch.no_grad():
                    p["logits"][:, a:a + wd.shape[0]] = (p["hn"][0] @ wd.T).float()  # matmul in `dtype` like HF
            del w, wd
        gc.collect()
        log("lm_head: %d rows in chunks of %d, %.1fs" % (vocab, LM_HEAD_CHUNK_ROWS, time.time() - t3))
        out = {}
        for name, p in P.items():
            out[name] = {"logits": p["logits"][-1].numpy().astype(np.float32), "argmax_per_position": [int(x) for x in p["logits"].argmax(-1)],
                         "stats": p["stats"]}
    finally:
        torch.set_default_dtype(torch.float32)
    return out, time.time() - t0


def forward_logits(store, tcfg, ids, dump_dir=None, log=print, dtype=torch.bfloat16):
    """Single-prompt wrapper kept for tools/ref_forward_selftest.py: (logits[-1] float32, per-layer stats, seconds)."""
    res, secs = run_prompts(store, tcfg, [("p", ids)], dtype, dump_dir=None, log=log)
    return res["p"]["logits"], res["p"]["stats"], secs


def text_config(hf_dir):
    cfg = AutoConfig.from_pretrained(hf_dir)
    tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    tcfg._attn_implementation = "eager"  # explicit; standalone layers cannot dispatch sdpa
    return tcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", help="directory with config.json and tokenizer.json (and the shards for --weights bf16)")
    ap.add_argument("--weights", choices=["bf16", "gguf"], default="bf16")
    ap.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
    ap.add_argument("--gguf", default=GGUF_DEFAULT)
    ap.add_argument("--out", default=None, help="output directory (default tests/fixtures/ref_<weights>)")
    ap.add_argument("--prompt", action="append", default=None, help="prompt name(s) from prompts.json; default: all")
    ap.add_argument("--layers", type=int, default=None, help="smoke test: run only the first N layers, write nothing")
    ap.add_argument("--dump-hidden", action="store_true")
    a = ap.parse_args()
    if a.weights == "gguf" and a.dtype != "f32":
        ap.error("--weights gguf is float32 only (the dequantised weights must not be rounded again)")
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    out_dir = a.out or os.path.join(FIX, "ref_" + a.weights)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    tcfg = text_config(a.hf_dir)
    prompts = json.load(open(os.path.join(FIX, "prompts.json")))["prompts"]
    if a.prompt:
        prompts = [p for p in prompts if p["name"] in a.prompt]
    t0 = time.time()
    store = GgufStore(a.gguf, tcfg) if a.weights == "gguf" else ShardStore(a.hf_dir)
    print("weight source: %s (%.1fs to open)" % (json.dumps(store.describe(), default=str), time.time() - t0))

    smoke = a.layers is not None and a.layers < tcfg.num_hidden_layers
    if not smoke:
        os.makedirs(out_dir, exist_ok=True)
    dump_dir = out_dir if (a.dump_hidden and not smoke) else None
    res, secs = run_prompts(store, tcfg, [(p["name"], p["ids"]) for p in prompts], dtype, dump_dir=dump_dir, n_layers=a.layers)
    if smoke:
        print("smoke run of %d layers done in %.0fs; nothing written" % (a.layers, secs))
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf_dir)
    info = {"weights": a.weights, "dtype": a.dtype, "source": store.describe(), "transformers": __import__("transformers").__version__,
            "torch": torch.__version__, "n_layers": tcfg.num_hidden_layers, "prompts": [p["name"] for p in prompts],
            "dump_hidden": bool(dump_dir), "wall_seconds": round(secs, 1),
            "note": "raw prompt ids, no BOS, no template; eager attention; layers built standalone and run one at a time; "
                    "hidden dumps are the residual stream after each layer at every position, float32"}
    for p in prompts:
        r = res[p["name"]]
        logits = r["logits"]
        if not np.isfinite(logits).all() or not np.any(logits) or float(logits.std()) < 1e-6:
            raise RuntimeError("%s: logits all-zero/constant/non-finite; refusing to write" % p["name"])
        argmax = int(logits.argmax())
        top10 = [int(i) for i in np.argsort(-logits)[:10]]
        np.save(os.path.join(out_dir, p["name"] + ".logits.npy"), logits)
        rec = {
            "name": p["name"], "prompt_text": p["text"], "prompt_ids": list(p["ids"]), "n_prompt_ids": len(p["ids"]),
            "argmax": argmax, "argmax_text": tok.decode([argmax]),
            "top10": top10, "top10_logits": [float(logits[i]) for i in top10], "top10_text": [tok.decode([i]) for i in top10],
            "argmax_per_position": r["argmax_per_position"],
            "logits_stats": {"max": float(logits.max()), "min": float(logits.min()), "mean": float(logits.mean()),
                             "std": float(logits.std()), "n_nan": int(np.isnan(logits).sum()), "max_abs": float(np.abs(logits).max())},
            "per_layer": r["stats"],
            "hidden_files": None if not dump_dir else {"embed": "%s__hidden_embed.npy" % p["name"],
                                                       "after_layer": "%s__hidden_after_layer_NN.npy" % p["name"],
                                                       "final_norm": "%s__hidden_final_norm.npy" % p["name"]},
            "info": info,
        }
        with open(os.path.join(out_dir, p["name"] + ".json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False)
        print("%-9s argmax=%d %r top10=%s max|logit|=%.3f" % (p["name"], argmax, rec["argmax_text"], top10, rec["logits_stats"]["max_abs"]))
    if dump_dir:
        files = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
        info["dump_bytes"] = sum(os.path.getsize(os.path.join(out_dir, f)) for f in files)
        info["dump_files"] = len(files)
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1, default=str)
    print("wrote %s in %.0fs" % (out_dir, secs))


if __name__ == "__main__":
    main()
