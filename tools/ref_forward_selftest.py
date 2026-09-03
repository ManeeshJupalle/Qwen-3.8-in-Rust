"""Self-test for tools/ref_forward.py on a tiny random Qwen3_5 text model (no real weights needed).

Builds a small Qwen3_5TextModel (same graph: 3:1 DeltaNet/attention pattern, MTP-free), writes its weights as
two safetensors shards + model.safetensors.index.json under the HF naming (model.language_model.*, lm_head),
then checks that ref_forward.forward_logits (layer-by-layer from the shards) reproduces the full model's
last-position logits. Validates the loading/masking/rotary plumbing before the 52 GiB shards exist.

Usage: python tools/ref_forward_selftest.py [scratch_dir] [float32|bfloat16]
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_forward as RF  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    scratch = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="aq_selftest_")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[sys.argv[2] if len(sys.argv) > 2 else "bfloat16"]
    os.makedirs(scratch, exist_ok=True)
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(os.path.join(ROOT, "models", "Qwen3.8-27B"))
    t = cfg.text_config
    # shrink every dimension but keep the structure and the real layer_types pattern
    t.num_hidden_layers = 8
    t.layer_types = t.layer_types[:8]
    t.hidden_size = 64
    t.intermediate_size = 96
    t.num_attention_heads = 4
    t.num_key_value_heads = 2
    t.head_dim = 16
    t.linear_num_key_heads = 2
    t.linear_num_value_heads = 4
    t.linear_key_head_dim = 8
    t.linear_value_head_dim = 8
    t.vocab_size = 512
    t.max_position_embeddings = 256
    t._attn_implementation = "eager"
    torch.set_default_dtype(dtype)
    model = Qwen3_5TextModel(t).eval()
    lm_head = torch.nn.Linear(t.hidden_size, t.vocab_size, bias=False)
    torch.set_default_dtype(torch.float32)

    # write shards under HF naming
    sd = {"model.language_model." + k: v.contiguous() for k, v in model.state_dict().items()}
    sd["lm_head.weight"] = lm_head.weight.data.contiguous()
    keys = sorted(sd)
    half = len(keys) // 2
    shards = {"model-00001-of-00002.safetensors": keys[:half], "model-00002-of-00002.safetensors": keys[half:]}
    wm = {}
    for shard, ks in shards.items():
        save_file({k: sd[k] for k in ks}, os.path.join(scratch, shard))
        for k in ks:
            wm[k] = shard
    json.dump({"metadata": {"total_size": sum(v.numel() * 2 for v in sd.values())}, "weight_map": wm},
              open(os.path.join(scratch, "model.safetensors.index.json"), "w"), indent=1)
    cfg.save_pretrained(scratch)

    ids = [3, 77, 250, 9, 411, 12, 500, 31, 2, 64, 199]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
        ref = lm_head(out.last_hidden_state[0, -1:]).float()[0].numpy()

    store = RF.ShardStore(scratch)
    tcfg = AutoConfig.from_pretrained(scratch).text_config
    tcfg._attn_implementation = "eager"
    logits, stats, secs = RF.forward_logits(store, tcfg, ids, None, log=lambda *x: None, dtype=dtype)
    d = np.abs(logits - ref)
    print("selftest[%s]: %d layers, T=%d, max|diff|=%.3e mean|diff|=%.3e (|ref| rms %.3e) argmax %d vs %d, %.1fs" % (
        str(dtype).split(".")[-1], len(stats), len(ids), d.max(), d.mean(), float(np.sqrt(np.mean(ref ** 2))),
        int(logits.argmax()), int(ref.argmax()), secs))
    print("per-layer max|h|:", [round(s["max_abs"], 3) for s in stats])
    tol = 1e-4 if dtype == torch.float32 else 5e-2
    ok = d.max() < tol and int(logits.argmax()) == int(ref.argmax())
    print("SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
