"""Compare two reference logit sets (e.g. llama.cpp vs HF bf16) on the three fixed prompts and write
docs/data/quant_noise_floor.txt.

For each prompt: argmax match, top-10 overlap (|A intersect B| / 10), max|diff|, mean|diff|, plus the same
numbers after log-softmax (so the comparison is invariant to a per-position logit offset).

Usage: python tools/cmp_logits.py <dir_a> <dir_b> [--label-a llamacpp --label-b hf]
Each dir holds <name>.logits.npy and <name>.json written by tools/ref_llamacpp.py / tools/ref_forward.py.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    a = ap.parse_args()
    prompts = json.load(open(os.path.join(FIX, "prompts.json")))["prompts"]
    L = ["# quant noise floor: %s (%s) vs %s (%s)" % (a.label_a, a.dir_a, a.label_b, a.dir_b), ""]
    n_argmax = 0
    worst_max = 0.0
    for p in prompts:
        fa = os.path.join(a.dir_a, p["name"] + ".logits.npy")
        fb = os.path.join(a.dir_b, p["name"] + ".logits.npy")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            L.append("%-9s MISSING: %s or %s" % (p["name"], fa, fb))
            continue
        x = np.load(fa).astype(np.float64)
        y = np.load(fb).astype(np.float64)
        assert x.shape == y.shape, (x.shape, y.shape)
        ax, ay = int(x.argmax()), int(y.argmax())
        tx = set(np.argsort(-x)[:10].tolist())
        ty = set(np.argsort(-y)[:10].tolist())
        d = np.abs(x - y)
        lx, ly = log_softmax(x), log_softmax(y)
        dl = np.abs(lx - ly)
        n_argmax += int(ax == ay)
        worst_max = max(worst_max, float(d.max()))
        L.append("%-9s argmax %s: %d vs %d | top10 overlap %d/10 | raw max|diff| %.4f mean|diff| %.5f | "
                 "logsoftmax max|diff| %.4f mean|diff| %.5f | top1 logprob %.4f vs %.4f" % (
                     p["name"], "MATCH" if ax == ay else "DIFF", ax, ay, len(tx & ty), d.max(), d.mean(),
                     dl.max(), dl.mean(), lx[ax], ly[ay]))
    L.append("")
    L.append("argmax match: %d/%d   worst raw max|diff|: %.4f" % (n_argmax, len(prompts), worst_max))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
