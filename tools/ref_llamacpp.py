"""llama.cpp reference logits on the exact GGUF the engine will load (via llama-cpp-python, CPU).

For each prompt in tests/fixtures/prompts.json:
  - feeds the RAW ids (no BOS, no chat template) through one forward pass
  - saves full float32 logits at the last position:  tests/fixtures/ref_llamacpp/<name>.logits.npy
  - saves argmax, top-10 ids, 16 greedy tokens (+ decoded text), timing and llama.cpp build info:
                                                      tests/fixtures/ref_llamacpp/<name>.json
  - appends a human-readable line to                 tests/fixtures/ref_llamacpp/summary.txt

Usage: python tools/ref_llamacpp.py <model.gguf> [--threads N] [--greedy 16] [--out DIR]
Refuses to write logits that are all-zero, constant, or non-finite.
"""
import argparse
import json
import os
import time

import numpy as np
import llama_cpp
from llama_cpp import Llama

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures")
OUT = os.path.join(FIX, "ref_llamacpp")


def check_logits(arr, what):
    """Refuse to write a logits vector that is all-zero or contains NaN/Inf (a silent readout failure)."""
    if not np.isfinite(arr).all():
        raise RuntimeError("%s: logits contain %d non-finite values; refusing to write" % (what, int((~np.isfinite(arr)).sum())))
    if not np.any(arr):
        raise RuntimeError("%s: logits are all zero (readout failure); refusing to write" % what)
    if float(arr.std()) < 1e-6:
        raise RuntimeError("%s: logits are constant (std=%g); refusing to write" % (what, float(arr.std())))


def last_logits(llm, n_vocab):
    """Logits of the last evaluated token. With logits_all=False llama.cpp computes logits only for the
    final token of the batch and llama-cpp-python 0.3.x does NOT copy them into Llama.scores, so read the
    context's logit buffer directly (row 0 is the only row)."""
    ptr = llm._ctx.get_logits()
    arr = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).astype(np.float32, copy=True)
    assert arr.shape == (n_vocab,)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--greedy", type=int, default=16)
    ap.add_argument("--n-ctx", type=int, default=512)
    ap.add_argument("--out", default=OUT, help="output directory (default tests/fixtures/ref_llamacpp)")
    a = ap.parse_args()
    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)

    prompts = json.load(open(os.path.join(FIX, "prompts.json")))["prompts"]

    t0 = time.time()
    llm = Llama(model_path=a.gguf, n_ctx=a.n_ctx, n_threads=a.threads, n_batch=a.n_ctx,
                logits_all=False, verbose=True, seed=0, use_mmap=True)
    t_load = time.time() - t0
    n_vocab = llm.n_vocab()
    info = {
        "llama_cpp_python": llama_cpp.__version__,
        "gguf": os.path.basename(a.gguf),
        "gguf_bytes": os.path.getsize(a.gguf),
        "n_vocab": n_vocab,
        "n_ctx": a.n_ctx,
        "threads": a.threads,
        "load_seconds": round(t_load, 2),
        "greedy_tokens": a.greedy,
        "note": "raw ids fed with Llama.eval (no BOS, no template); logits read from llama_get_logits (last evaluated position); "
                "greedy = argmax of the returned logits, appended one token at a time with KV cache",
    }
    summary = []
    for p in prompts:
        ids = list(p["ids"])
        llm.reset()
        t1 = time.time()
        llm.eval(ids)
        t_prompt = time.time() - t1
        last = last_logits(llm, n_vocab)
        assert last.shape == (n_vocab,), last.shape
        argmax = int(last.argmax())
        top10 = [int(i) for i in np.argsort(-last)[:10]]
        gen = []
        t2 = time.time()
        cur = last
        for _ in range(a.greedy):
            nxt = int(cur.argmax())
            gen.append(nxt)
            llm.eval([nxt])
            cur = last_logits(llm, n_vocab)
        t_gen = time.time() - t2
        text = llm.detokenize(gen).decode("utf-8", "replace") if gen else ""
        check_logits(last, p["name"])
        np.save(os.path.join(out_dir, p["name"] + ".logits.npy"), last)
        rec = {
            "name": p["name"],
            "prompt_text": p["text"],
            "prompt_ids": ids,
            "n_prompt_ids": len(ids),
            "argmax": argmax,
            "argmax_text": llm.detokenize([argmax]).decode("utf-8", "replace"),
            "top10": top10,
            "top10_logits": [float(last[i]) for i in top10],
            "top10_text": [llm.detokenize([i]).decode("utf-8", "replace") for i in top10],
            "greedy_ids": gen,
            "greedy_text": text,
            "logits_stats": {"max": float(last.max()), "min": float(last.min()), "mean": float(last.mean()),
                             "std": float(last.std()), "n_nan": int(np.isnan(last).sum())},
            "prompt_seconds": round(t_prompt, 3),
            "gen_seconds": round(t_gen, 3),
            "ms_per_generated_token": round(1000 * t_gen / max(1, a.greedy), 1),
        }
        rec.update({"info": info})
        with open(os.path.join(out_dir, p["name"] + ".json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False)
        line = "%-9s argmax=%d %r top10=%s greedy=%r prompt=%.1fs gen=%.1fs (%.0f ms/tok)" % (
            p["name"], argmax, rec["argmax_text"], top10, text, t_prompt, t_gen, rec["ms_per_generated_token"])
        print(line)
        summary.append(line)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("# tools/ref_llamacpp.py  %s  llama-cpp-python %s  threads=%d\n" % (info["gguf"], info["llama_cpp_python"], a.threads))
        fh.write("\n".join(summary) + "\n")
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)


if __name__ == "__main__":
    main()
