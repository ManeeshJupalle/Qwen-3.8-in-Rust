"""Finish a tools/ref_forward.py run whose per-layer dumps exist but whose lm_head pass did not complete:
recompute the logits at every prompt position from <name>__hidden_final_norm.npy and lm_head (bf16 shards or
GGUF), in float32, and write <name>.logits.npy, <name>.json and info.json exactly as ref_forward.py would.

Usage: python tools/ref_logits_from_dump.py <hf_dir> --weights bf16|gguf --out DIR [--gguf PATH]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_forward as RF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("--weights", choices=["bf16", "gguf"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", default=RF.GGUF_DEFAULT)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tcfg = RF.text_config(a.hf_dir)
    store = RF.GgufStore(a.gguf, tcfg) if a.weights == "gguf" else RF.ShardStore(a.hf_dir)
    prompts = json.load(open(os.path.join(RF.FIX, "prompts.json")))["prompts"]
    vocab = tcfg.vocab_size
    hn = {p["name"]: torch.from_numpy(np.load(os.path.join(a.out, "%s__hidden_final_norm.npy" % p["name"]))) for p in prompts}
    logits = {n: torch.empty(h.shape[0], vocab, dtype=torch.float32) for n, h in hn.items()}
    t0 = time.time()
    for s, w in store.lm_head_chunks(vocab):
        wd = w.to(torch.float32)
        for n, h in hn.items():
            with torch.no_grad():
                logits[n][:, s:s + wd.shape[0]] = h @ wd.T
        del w, wd
    print("lm_head pass: %.0fs" % (time.time() - t0))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf_dir)
    info = {"weights": a.weights, "dtype": "f32", "source": store.describe(), "transformers": __import__("transformers").__version__,
            "torch": torch.__version__, "n_layers": tcfg.num_hidden_layers, "prompts": [p["name"] for p in prompts], "dump_hidden": True,
            "note": "per-layer dumps from tools/ref_forward.py; logits recomputed from the final-norm dump by tools/ref_logits_from_dump.py "
                    "(float32 matmul, lm_head in row chunks); raw prompt ids, no BOS, no template; eager attention"}
    for p in prompts:
        name = p["name"]
        lg = logits[name]
        last = lg[-1].numpy().astype(np.float32)
        if not np.isfinite(last).all() or not np.any(last) or float(last.std()) < 1e-6:
            raise RuntimeError("%s: logits all-zero/constant/non-finite; refusing to write" % name)
        argmax = int(last.argmax())
        top10 = [int(i) for i in np.argsort(-last)[:10]]
        np.save(os.path.join(a.out, name + ".logits.npy"), last)
        per_layer = []
        for i in range(tcfg.num_hidden_layers):
            h = np.load(os.path.join(a.out, "%s__hidden_after_layer_%02d.npy" % (name, i)))
            per_layer.append({"layer": i, "type": tcfg.layer_types[i], "max_abs": float(np.abs(h).max()), "last_pos_max_abs": float(np.abs(h[-1]).max()),
                              "mean_abs": float(np.abs(h).mean()), "n_nan": int(np.isnan(h).sum())})
        rec = {"name": name, "prompt_text": p["text"], "prompt_ids": list(p["ids"]), "n_prompt_ids": len(p["ids"]),
               "argmax": argmax, "argmax_text": tok.decode([argmax]), "top10": top10, "top10_logits": [float(last[i]) for i in top10],
               "top10_text": [tok.decode([i]) for i in top10], "argmax_per_position": [int(x) for x in lg.argmax(-1)],
               "logits_stats": {"max": float(last.max()), "min": float(last.min()), "mean": float(last.mean()), "std": float(last.std()),
                                "n_nan": int(np.isnan(last).sum()), "max_abs": float(np.abs(last).max())},
               "per_layer": per_layer,
               "hidden_files": {"embed": "%s__hidden_embed.npy" % name, "after_layer": "%s__hidden_after_layer_NN.npy" % name,
                                "final_norm": "%s__hidden_final_norm.npy" % name},
               "info": info}
        with open(os.path.join(a.out, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False)
        print("%-9s argmax=%d %r top10=%s max|logit|=%.3f" % (name, argmax, rec["argmax_text"], top10, rec["logits_stats"]["max_abs"]))
    files = [f for f in os.listdir(a.out) if f.endswith(".npy")]
    info["dump_bytes"] = sum(os.path.getsize(os.path.join(a.out, f)) for f in files)
    info["dump_files"] = len(files)
    with open(os.path.join(a.out, "info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
