"""Quantisation noise floor: the HF reference from the bf16 checkpoint vs the same reference from the dequantised
Q4_K_M GGUF, per layer and at the logits, on the three fixed prompts. Writes docs/data/quant_noise_floor.txt.

Both directories come from tools/ref_forward.py (--weights bf16 / --weights gguf, --dump-hidden, float32 compute):
same code, same activations precision, only the weight bytes differ, so every difference below is weight
quantisation. llama.cpp's logits on the same GGUF (tests/fixtures/ref_llamacpp) are added as a third column at
the logits: it also reads the quantised bytes but quantises activations to 8 bits, so its distance from ref_gguf
is a preview of the engine's own activation-quantisation noise.

Usage: python tools/noise_floor.py [--bf16 DIR] [--gguf DIR] [--llamacpp DIR] [--out FILE]
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures")
OUT = os.path.join(ROOT, "docs", "data", "quant_noise_floor.txt")


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def logit_row(label, x, y):
    ax, ay = int(x.argmax()), int(y.argmax())
    tx, ty = set(np.argsort(-x)[:10].tolist()), set(np.argsort(-y)[:10].tolist())
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    dl = np.abs(log_softmax(x) - log_softmax(y))
    return "  %-28s argmax %s (%d vs %d) | top10 overlap %2d/10 | raw max|diff| %.4f mean|diff| %.5f | logsoftmax max|diff| %.4f mean|diff| %.5f" % (
        label, "MATCH" if ax == ay else "DIFF ", ax, ay, len(tx & ty), d.max(), d.mean(), dl.max(), dl.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", default=os.path.join(FIX, "ref_bf16"))
    ap.add_argument("--gguf", default=os.path.join(FIX, "ref_gguf"))
    ap.add_argument("--llamacpp", default=os.path.join(FIX, "ref_llamacpp"))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    ib = json.load(open(os.path.join(a.bf16, "info.json")))
    ig = json.load(open(os.path.join(a.gguf, "info.json")))
    prompts = json.load(open(os.path.join(FIX, "prompts.json")))["prompts"]
    n_layers = ig["n_layers"]
    L = ["# Quantisation noise floor: HF reference from bf16 shards vs HF reference from the dequantised bartowski Q4_K_M GGUF",
         "# tools/noise_floor.py; sources: tools/ref_forward.py --weights bf16 --dtype %s (%s) and --weights gguf (%s)" % (
             ib["dtype"], ib["source"].get("hf_dir", ""), ig["source"].get("gguf", "")),
         "# both float32 compute, eager attention, raw prompt ids; per layer: max|bf16 - gguf| over all positions, relative to max|bf16|, and",
         "# the RMS of the difference relative to the RMS of the bf16 hidden state (rms_rel); 'cos' is the cosine between the two hidden states",
         ""]
    per_layer_worst = np.zeros(n_layers)
    for p in prompts:
        name = p["name"]
        L.append("== prompt %s (%d ids) ==" % (name, p["n_ids"]))
        L.append("  layer type              max|diff|   max|bf16|   rel      rms_rel   cos")
        for i in range(n_layers):
            hb = np.load(os.path.join(a.bf16, "%s__hidden_after_layer_%02d.npy" % (name, i))).astype(np.float64)
            hg = np.load(os.path.join(a.gguf, "%s__hidden_after_layer_%02d.npy" % (name, i))).astype(np.float64)
            d = hb - hg
            mx = float(np.abs(hb).max())
            rel = float(np.abs(d).max()) / mx
            rms_rel = float(np.sqrt((d ** 2).mean()) / np.sqrt((hb ** 2).mean()))
            cos = float((hb * hg).sum() / (np.linalg.norm(hb) * np.linalg.norm(hg)))
            per_layer_worst[i] = max(per_layer_worst[i], rel)
            typ = json.load(open(os.path.join(a.gguf, name + ".json")))["per_layer"][i]["type"] if i == 0 else typ_list[i]
            if i == 0:
                typ_list = [r["type"] for r in json.load(open(os.path.join(a.gguf, name + ".json")))["per_layer"]]
            L.append("  %5d %-16s %10.4f %10.3f  %.2e  %.2e  %.6f" % (i, typ_list[i], float(np.abs(d).max()), mx, rel, rms_rel, cos))
        hb = np.load(os.path.join(a.bf16, "%s__hidden_final_norm.npy" % name)).astype(np.float64)
        hg = np.load(os.path.join(a.gguf, "%s__hidden_final_norm.npy" % name)).astype(np.float64)
        d = hb - hg
        L.append("  final norm: max|diff| %.4f max|bf16| %.3f rel %.2e rms_rel %.2e" % (
            float(np.abs(d).max()), float(np.abs(hb).max()), float(np.abs(d).max() / np.abs(hb).max()), float(np.sqrt((d ** 2).mean()) / np.sqrt((hb ** 2).mean()))))
        lb = np.load(os.path.join(a.bf16, name + ".logits.npy"))
        lg = np.load(os.path.join(a.gguf, name + ".logits.npy"))
        L.append(logit_row("logits bf16 vs gguf-dequant", lb, lg))
        fl = os.path.join(a.llamacpp, name + ".logits.npy")
        if os.path.exists(fl):
            ll = np.load(fl)
            L.append(logit_row("logits bf16 vs llama.cpp", lb, ll))
            L.append(logit_row("logits gguf-dequant vs llama.cpp", lg, ll))
        L.append("")
    L.append("worst relative max|diff| per layer over the 3 prompts (bf16 vs gguf-dequant):")
    L.append("  " + " ".join("%d:%.1e" % (i, v) for i, v in enumerate(per_layer_worst)))
    L.append("")
    L.append("Reading: this is the distance between the published bf16 model and the Q4_K_M file the engine loads, computed with")
    L.append("identical float32 code. An engine reading the GGUF should sit far inside these numbers against ref_gguf (its parity")
    L.append("target) and about here against the bf16 model; matching bf16 more closely than this is impossible with these weights.")
    text = "\n".join(L) + "\n"
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
