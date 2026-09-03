"""Tokenize the three fixed reference prompts with HF `tokenizers` and write tests/fixtures/prompts.json.

The ids are RAW text ids: no chat template, no BOS (tokenizer_config.json says add_bos_token=false),
add_special_tokens=False. Every reference (llama.cpp, HF) must be fed exactly these ids.

Usage: python tools/prompts.py <path/to/tokenizer.json>
"""
import hashlib
import json
import os
import sys

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures", "prompts.json")

PROMPTS = [
    ("capital", "The capital of France is"),
    ("fib", "def fibonacci(n):"),
    ("sentence", "Rivers carve valleys slowly over thousands of years, carrying sediment downstream until it "
                 "settles in deltas where the water finally meets the sea, building new land while the tides "
                 "push back against the shore."),
]


def main():
    tok_path = sys.argv[1]
    tok = Tokenizer.from_file(tok_path)
    out = {
        "tokenizer_json_sha256": hashlib.sha256(open(tok_path, "rb").read()).hexdigest(),
        "add_special_tokens": False,
        "bos_prepended": False,
        "prompts": [],
    }
    for name, text in PROMPTS:
        ids = tok.encode(text, add_special_tokens=False).ids
        out["prompts"].append({
            "name": name,
            "text": text,
            "text_utf8_hex": text.encode("utf-8").hex(),
            "ids": ids,
            "n_ids": len(ids),
        })
        print("%-9s n_ids=%3d ids=%s" % (name, len(ids), ids))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
