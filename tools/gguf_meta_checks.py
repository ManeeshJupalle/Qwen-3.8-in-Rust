"""Two checks on a captured GGUF metadata fixture (tests/fixtures/gguf_metadata*.json):

  1. a compact one-line-per-key summary (long arrays elided)      -> <out_dir>/gguf_metadata_summary.txt
  2. GGUF tokenizer/template vs the HF files: chat template diff, vocab id-for-id comparison, token types,
     special ids, merges                                          -> <out_dir>/gguf_vs_hf_tokenizer.txt

Usage: python tools/gguf_meta_checks.py tests/fixtures/gguf_metadata.json models/Qwen3.8-27B docs/data
"""
import collections
import difflib
import json
import os
import sys


def summary(m, path):
    lines = ["# compact view of the GGUF KV metadata (arrays longer than 16 elided); tools/gguf_meta_checks.py", ""]
    for k, v in m.items():
        val = v["value"]
        if isinstance(val, list) and len(val) > 16:
            s = "<array len %d, first 8: %s>" % (len(val), val[:8])
        else:
            s = json.dumps(val, ensure_ascii=False)
        if len(s) > 300:
            s = s[:300] + "...(truncated)"
        lines.append("%-45s %-12s %s" % (k, ",".join(v["types"]), s))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return lines


def tokenizer_check(m, hf_dir, path):
    tj = json.load(open(os.path.join(hf_dir, "tokenizer.json"), encoding="utf-8"))
    hf_tpl = open(os.path.join(hf_dir, "chat_template.jinja"), encoding="utf-8").read()
    L = ["# GGUF tokenizer/template vs HF files; tools/gguf_meta_checks.py", ""]
    P = L.append
    gg_tpl = m["tokenizer.chat_template"]["value"] if "tokenizer.chat_template" in m else None
    if gg_tpl is None:
        P("GGUF has no tokenizer.chat_template key")
    else:
        P("chat template identical to HF chat_template.jinja: %s (gguf %d chars, hf %d chars)" % (gg_tpl == hf_tpl, len(gg_tpl), len(hf_tpl)))
        if gg_tpl != hf_tpl:
            d = list(difflib.unified_diff(hf_tpl.splitlines(), gg_tpl.splitlines(), "HF chat_template.jinja",
                                          "GGUF tokenizer.chat_template", lineterm="", n=1))
            P("unified diff (%d lines):" % len(d))
            L.extend("  " + x for x in d)
    P("")
    toks = m["tokenizer.ggml.tokens"]["value"]
    tt = m["tokenizer.ggml.token_type"]["value"]
    vocab = tj["model"]["vocab"]
    added = {t["id"]: t for t in tj["added_tokens"]}
    inv = {v: k for k, v in vocab.items()}
    for i, t in added.items():
        inv[i] = t["content"]
    P("gguf tokens: %d ; hf vocab: %d ; hf added_tokens: %d ; hf max id+1: %d" % (len(toks), len(vocab), len(added), max(inv) + 1))
    mism = [(i, toks[i], inv.get(i)) for i in range(len(toks)) if toks[i] != inv.get(i)]
    pad = [i for i, t, h in mism if h is None]
    real = [(i, t, h) for i, t, h in mism if h is not None]
    P("ids where gguf token != hf token (by id): %d" % len(mism))
    P("  of which hf has no token (padding ids): %d, range %s..%s, gguf strings e.g. %s" % (
        len(pad), min(pad) if pad else None, max(pad) if pad else None, [toks[i] for i in pad[:3]]))
    P("  real mismatches: %d %s" % (len(real), real[:10]))
    P("token_type histogram: %s" % dict(collections.Counter(tt)))
    P("ids with token_type != 1 (first 40): %s" % [(i, tt[i], toks[i]) for i in range(len(tt)) if tt[i] != 1][:40])
    P("")
    P("tokenizer.* keys present: %s" % [k for k in m if k.startswith("tokenizer.")])
    for k in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token", "tokenizer.ggml.add_space_prefix"):
        P("%s present: %s%s" % (k, k in m, (" = %s" % m[k]["value"]) if k in m else ""))
    P("bos=%s eos=%s pad=%s" % (m.get("tokenizer.ggml.bos_token_id", {}).get("value"),
                               m.get("tokenizer.ggml.eos_token_id", {}).get("value"),
                               m.get("tokenizer.ggml.padding_token_id", {}).get("value")))
    hf_merges = [(x if isinstance(x, str) else " ".join(x)) for x in tj["model"]["merges"]]
    gg_merges = m["tokenizer.ggml.merges"]["value"]
    P("merges: gguf %d vs hf %d; identical: %s" % (len(gg_merges), len(hf_merges), gg_merges == hf_merges))
    P("tokenizer.ggml.model=%s pre=%s" % (m["tokenizer.ggml.model"]["value"], m.get("tokenizer.ggml.pre", {}).get("value")))
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return L


def main():
    meta_path, hf_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    m = json.load(open(meta_path, encoding="utf-8"))
    s = summary(m, os.path.join(out_dir, "gguf_metadata_summary.txt"))
    print("\n".join(s[:60]))
    print("...")
    t = tokenizer_check(m, hf_dir, os.path.join(out_dir, "gguf_vs_hf_tokenizer.txt"))
    print("\n".join(x for x in t if not x.startswith("  ")))


if __name__ == "__main__":
    main()
