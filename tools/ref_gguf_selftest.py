"""Self-test for the `--weights gguf` path of tools/ref_forward.py: run it on models/tiny/tiny-f32.gguf (the tiny
oracle built through llama.cpp's converter) and compare per-layer hidden states and last-position logits with
tests/fixtures/tiny/tiny/manifest.json (tools/tiny_reference.py, produced by Qwen3_5ForCausalLM from the HF
checkpoint the GGUF was converted from). This checks the inverse layout transforms (V-head un-tiling, A_log from
ssm_a, norms minus 1, conv1d unsqueeze) end to end: an F32 GGUF read back through them must reproduce the HF
model up to float32 rounding (expected ~1e-7 or bit-exact).

Usage: python tools/ref_gguf_selftest.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_forward as RF  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = os.path.join(ROOT, "models", "tiny")
FXD = os.path.join(ROOT, "tests", "fixtures", "tiny", "tiny")


def main():
    tcfg = RF.text_config(TINY)
    store = RF.GgufStore(os.path.join(TINY, "tiny-f32.gguf"), tcfg)
    m = json.load(open(os.path.join(FXD, "manifest.json")))
    prompts = [(c["case"], c["prompt_ids"]) for c in m["cases"]]
    worst = 0.0
    ok = True
    import tempfile
    d = tempfile.mkdtemp(prefix="aq_gguf_selftest_")
    res, secs = RF.run_prompts(store, tcfg, prompts, torch.float32, dump_dir=d, log=lambda *x: None)
    for c in m["cases"]:
        name = c["case"]
        for i in range(tcfg.num_hidden_layers):
            got = np.load(os.path.join(d, "%s__hidden_after_layer_%02d.npy" % (name, i)))
            want = np.load(os.path.join(FXD, c["hidden"]["layer_%d" % i]["file"]))
            diff = float(np.abs(got - want).max())
            worst = max(worst, diff / max(c["hidden"]["layer_%d" % i]["max_abs"], 1e-30))
            if diff > 1e-5 * c["hidden"]["layer_%d" % i]["max_abs"]:
                ok = False
                print("%s layer %d: max|diff| %.3e (max|ref| %.3e)" % (name, i, diff, c["hidden"]["layer_%d" % i]["max_abs"]))
        got = np.load(os.path.join(d, "%s__hidden_embed.npy" % name))
        want = np.load(os.path.join(FXD, c["hidden"]["embed"]["file"]))
        assert np.array_equal(got, want), "%s: embedding rows differ" % name
        lg = res[name]["logits"]
        want_l = np.load(os.path.join(FXD, c["logits_last"]["file"]))
        dl = float(np.abs(lg - want_l).max())
        argmax_ok = int(lg.argmax()) == c["top5_last"][0]
        ok = ok and argmax_ok and dl <= 1e-5 * c["logits_max_abs"]
        print("%-9s logits max|diff| %.3e (max|ref| %.3f) argmax %d vs %d; argmax per position %s" % (
            name, dl, c["logits_max_abs"], int(lg.argmax()), c["top5_last"][0],
            "match" if res[name]["argmax_per_position"] == c["argmax_per_position"] else "DIFFER"))
        ok = ok and res[name]["argmax_per_position"] == c["argmax_per_position"]
    print("worst per-layer relative diff %.3e; A_log exact: %s" % (worst, store.a_log_exact))
    print("GGUF PATH SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
