"""Freeze rule-5 regression ceilings for one K-quant activation format from a parity log (Phase 3.5).

Reads the output of `crates/core/tests/real_layers.rs real_layers_0_63_and_logits` (run with
`AQUEDUCT_ACT=<format>` and no frozen file for that format, so it is a measurement run) and writes the
ceilings fixture the harness gates that format against: per layer the measured relative error of every
prompt (the harness takes `factor` x the max over prompts as the ceiling, rule 5's "per layer"), and per
prompt the last-position logits max|diff| vs ref_gguf (ceiling = `factor` x that) and vs llama.cpp (for
reference only). Values keep 3 significant digits.

Usage: python tools/freeze_ceilings.py <parity.log> --act q8k [--factor 2] [--out tests/fixtures/q8k_ceilings.json]
"""
import argparse
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LAYER_RE = re.compile(r"^layer\s+(\d+)\s+(\w+)\s+\|")
PROMPT_RE = re.compile(r"\|\s+(\w+)\s+f32 ")
REL_RE = re.compile(r"q8 \S+ \((\S+) rel, vs f32")
LOGITS_RE = re.compile(
    r"^(\w+)\s+logits q8: vs ref_gguf max\|diff\| ([\d.]+) argmax (\w+) top10 (\d+)/10 \| vs llama\.cpp max\|diff\| ([\d.]+) argmax (\w+) top10 (\d+)/10"
)


def sig3(v):
    return float("%.3g" % v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--act", required=True, choices=["q8_0", "q8k"])
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "tests", "fixtures", "q8_ceilings.json" if a.act == "q8_0" else "q8k_ceilings.json")

    layers = {}
    logits = {}
    with open(a.log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LAYER_RE.match(line)
            if m:
                l, kind = int(m.group(1)), m.group(2)
                prompts = PROMPT_RE.findall(line)
                rels = [float(x) for x in REL_RE.findall(line)]
                if len(prompts) != len(rels) or not prompts:
                    raise SystemExit("layer %d: %d prompts but %d q8 entries in: %s" % (l, len(prompts), len(rels), line.strip()))
                layers[l] = {"layer": l, "kind": kind, "rel": {p: sig3(r) for p, r in zip(prompts, rels)}}
                continue
            m = LOGITS_RE.match(line)
            if m:
                logits[m.group(1)] = {
                    "vs_ref_gguf_max_diff": float(m.group(2)),
                    "vs_ref_gguf_argmax": m.group(3),
                    "vs_ref_gguf_top10": int(m.group(4)),
                    "vs_llamacpp_max_diff": float(m.group(5)),
                    "vs_llamacpp_argmax": m.group(6),
                    "vs_llamacpp_top10": int(m.group(7)),
                }
    if sorted(layers) != list(range(len(layers))) or not layers:
        raise SystemExit("layers found: %s" % sorted(layers))
    if len(logits) != 3:
        raise SystemExit("logits lines found for %s (need the 3 prompts)" % sorted(logits))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    doc = {
        "source": "%s (commit %s): engine q8 mode with %s activations vs tests/fixtures/ref_gguf, max|diff| / max|ref| per layer and prompt (crates/core/tests/real_layers.rs real_layers_0_63_and_logits, AVX2 kernels)"
        % (os.path.relpath(a.log, ROOT).replace("\\", "/"), commit, "ggml Q8_K (256-value blocks, f32 scale)" if a.act == "q8k" else "Q8_0-grain (32-value blocks, f16 scale)"),
        "act": a.act,
        "rule": "rule 5 (amended in Phase 3.5: ceilings are per activation format, each frozen from that format's own measurement): ceiling per layer = factor * max over the three prompts of the measured relative error; crates/core/tests/real_layers.rs fails if the q8 mode's relative error after any layer, on any prompt, exceeds that layer's ceiling, or if the last-position logits max|diff| vs ref_gguf exceeds factor * measured for that prompt",
        "factor": a.factor,
        "measured": {"layers": [layers[l] for l in sorted(layers)], "logits": logits},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    worst = max((max(v["rel"].values()), l) for l, v in layers.items())
    print("wrote %s: %d layers, worst measured rel %.3g at layer %d; logits vs ref_gguf %s" % (out, len(layers), worst[0], worst[1], {k: v["vs_ref_gguf_max_diff"] for k, v in logits.items()}))


if __name__ == "__main__":
    main()
